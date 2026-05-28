#!/usr/bin/env python3
"""
dose_predict_3d_v1_4.py
=======================
v1.4 (ControlNet-lite penalty adapter) full-volume sliding-window
inference. Same prediction interface as ``dose_predict_3d.py`` but
additionally loads the 3D penalty volume per patient and forwards it
to ``UNetModel_PenaltyAdapter_v1_4``.

Penalty I/O mirrors the v1.4 dataset:
    --penalty_mode single  -> penalty == dose_synthetic.nii.gz (P=1).
                              The synthetic dose volume already consumed
                              by the SYN stream is *re-used* as the
                              penalty input — no separate file is read.
    --penalty_mode multi   -> 4 files (S_target / A_oar / B_boundary / P_fused, P=4)
                              file names overridable via --penalty_files

Penalty dropout is automatically disabled at inference (model is in
eval mode; the dropout switch in v1.4 forward only fires when
``self.training`` is True).
"""
import sys
import gc

sys.path.append("../")
sys.path.append("./")

import os
import argparse
import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm

from guided_diffusion.dose_loader_3d import OPENKBP_MASK_NAMES
from guided_diffusion.dose_loader_3d_v1_4 import DEFAULT_MULTI_PENALTY_FILES
from guided_diffusion.unet_3d_v1_4 import UNetModel_PenaltyAdapter_v1_4
from flow_matching import FlowMatchingV14


def denormalize_dose(dose, dose_max=80.0, dose_norm_factor=40.0):
    dose = (dose + 1.0) * dose_norm_factor
    dose = np.clip(dose, 0, dose_max)
    return dose


def normalize_ct(img):
    img = np.clip(img, 0, 2500)
    img = img / 1250.0 - 1.0
    return img


def normalize_dose_input(img):
    img = np.clip(img, 0, 80)
    img = img / 40.0 - 1.0
    return img


def normalize_penalty(arr):
    amax = float(np.max(np.abs(arr)))
    if amax > 1e-6:
        arr = arr / amax
    return arr


def _offsets(dim_size, patch_size, overlap_ratio=0.25):
    if patch_size >= dim_size:
        return [0]
    step = max(1, int(patch_size * (1.0 - overlap_ratio)))
    offsets = list(range(0, dim_size - patch_size, step))
    last = dim_size - patch_size
    if not offsets or offsets[-1] != last:
        offsets.append(last)
    return sorted(set(offsets))


def predict_full_volume(
    flow_model,
    ct_volume,        # (D, H, W)
    syn_volume,       # (D, H, W)
    dis_volume,       # (11, D, H, W)
    penalty_volume,   # (P, D, H, W)
    patch_size=(128, 128, 128),
    steps=30,
    device='cuda',
):
    D, H, W = ct_volume.shape
    patch_d, patch_h, patch_w = patch_size

    pred_dose = np.zeros((D, H, W), dtype=np.float32)
    count_map = np.zeros((D, H, W), dtype=np.float32)

    d_offsets = _offsets(D, patch_d)
    h_offsets = _offsets(H, patch_h)
    w_offsets = _offsets(W, patch_w)

    flow_model.eval()
    with torch.no_grad():
        for d in d_offsets:
            for h in h_offsets:
                for w in w_offsets:
                    d_end = min(d + patch_d, D)
                    h_end = min(h + patch_h, H)
                    w_end = min(w + patch_w, W)

                    ct_patch = ct_volume[d:d_end, h:h_end, w:w_end]
                    syn_patch = syn_volume[d:d_end, h:h_end, w:w_end]
                    dis_patch = dis_volume[:, d:d_end, h:h_end, w:w_end]
                    pen_patch = penalty_volume[:, d:d_end, h:h_end, w:w_end]

                    if ct_patch.shape != patch_size:
                        pad_d = patch_d - ct_patch.shape[0]
                        pad_h = patch_h - ct_patch.shape[1]
                        pad_w = patch_w - ct_patch.shape[2]
                        ct_patch = np.pad(ct_patch, ((0, pad_d), (0, pad_h), (0, pad_w)),
                                          'constant', constant_values=-1.0)
                        syn_patch = np.pad(syn_patch, ((0, pad_d), (0, pad_h), (0, pad_w)),
                                           'constant', constant_values=-1.0)
                        dis_patch = np.pad(dis_patch, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
                                           'constant', constant_values=0.0)
                        pen_patch = np.pad(pen_patch, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
                                           'constant', constant_values=0.0)

                    ct_t = torch.from_numpy(ct_patch).unsqueeze(0).unsqueeze(0).float().to(device)
                    syn_t = torch.from_numpy(syn_patch).unsqueeze(0).unsqueeze(0).float().to(device)
                    dis_t = torch.from_numpy(dis_patch).unsqueeze(0).float().to(device)
                    pen_t = torch.from_numpy(pen_patch).unsqueeze(0).float().to(device)

                    pred = flow_model.sample(ct_t, syn_t, dis_t, pen_t, steps=steps)
                    pred = pred.squeeze(0).squeeze(0).cpu().numpy()  # (D, H, W) of patch

                    del ct_t, syn_t, dis_t, pen_t
                    torch.cuda.empty_cache()

                    actual_d = d_end - d
                    actual_h = h_end - h
                    actual_w = w_end - w
                    pred_crop = denormalize_dose(pred[:actual_d, :actual_h, :actual_w])

                    pred_dose[d:d_end, h:h_end, w:w_end] += pred_crop
                    count_map[d:d_end, h:h_end, w:w_end] += 1.0

                    del pred

        count_map[count_map == 0] = 1.0
        pred_dose = pred_dose / count_map

    return pred_dose


def _load_penalty_volume(patient_dir, ref_shape, penalty_files, normalize=True):
    """Mirrors Dataset_PSDM_3D_Train_v1_4._load_penalty_volume."""
    channels = []
    for fname in penalty_files:
        fpath = os.path.join(patient_dir, fname)
        if os.path.exists(fpath):
            arr = nib.load(fpath).get_fdata().astype(np.float32).transpose(2, 0, 1)
            if arr.shape != ref_shape:
                arr = np.zeros(ref_shape, dtype=np.float32)
        else:
            arr = np.zeros(ref_shape, dtype=np.float32)
        if normalize:
            arr = normalize_penalty(arr)
        channels.append(arr)
    return np.stack(channels, axis=0)


def main():
    parser = argparse.ArgumentParser(description='v1.4 ControlNet-lite penalty adapter inference')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--patch_size', type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model_channels', type=int, default=32)
    parser.add_argument('--channel_mult', type=int, nargs='+', default=[1, 2, 4, 4])

    # Penalty branch
    parser.add_argument('--penalty_mode', type=str, choices=['single', 'multi'], default='single')
    parser.add_argument('--penalty_files', type=str, nargs='*', default=None)
    parser.add_argument('--penalty_channels', type=int, default=None)
    parser.add_argument('--penalty_model_channels', type=int, default=None)
    parser.add_argument('--penalty_num_res_blocks', type=int, default=1)

    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    patch_size = tuple(args.patch_size)
    if any(s % 16 != 0 for s in patch_size):
        raise ValueError(f"patch_size 必须为 16 的倍数: {patch_size}")
    if args.model_channels % 32 != 0:
        raise ValueError(f"--model_channels 必须为 32 的倍数: {args.model_channels}")

    # Resolve penalty source.
    if args.penalty_mode == 'single':
        penalty_files = ()  # informational only; penalty == dose_synthetic
        penalty_channels = 1
        print("[v1.4] Penalty mode=single (penalty == dose_synthetic.nii.gz, reused from SYN)")
    else:
        if args.penalty_files:
            penalty_files = tuple(args.penalty_files)
        else:
            penalty_files = DEFAULT_MULTI_PENALTY_FILES
        penalty_channels = args.penalty_channels if args.penalty_channels is not None else len(penalty_files)
        print(f"[v1.4] Penalty mode=multi  files={list(penalty_files)} channels={penalty_channels}")

    dis_channels = len(OPENKBP_MASK_NAMES)

    model = UNetModel_PenaltyAdapter_v1_4(
        image_size=patch_size,
        in_channels=1,
        ct_channels=1,
        syn_channels=1,
        dis_channels=dis_channels,
        model_channels=args.model_channels,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        channel_mult=tuple(args.channel_mult),
        dims=3,
        # Inference: penalty dropout is harmless here but force-disable for clarity.
        penalty_channels=penalty_channels,
        penalty_model_channels=args.penalty_model_channels,
        penalty_num_res_blocks=args.penalty_num_res_blocks,
        use_penalty_dropout=False,
    )

    print(f"Loading checkpoint: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location='cpu')
    if any(k.startswith('module.') for k in ckpt.keys()):
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"[Load] missing={len(missing)} unexpected={len(unexpected)}")
    del ckpt
    gc.collect()

    model = model.to(device)
    model.eval()
    torch.cuda.empty_cache()
    print("Model ready.")

    flow_model = FlowMatchingV14(model)

    patient_list = sorted([p for p in os.listdir(args.data_dir)
                           if os.path.isdir(os.path.join(args.data_dir, p))])
    print(f"Found {len(patient_list)} patients in {args.data_dir}")

    mask_names = list(OPENKBP_MASK_NAMES)

    for patient_id in tqdm(patient_list, desc="Patients"):
        patient_dir = os.path.join(args.data_dir, patient_id)
        out_patient_dir = os.path.join(args.output_dir, patient_id)
        out_path = os.path.join(out_patient_dir, 'dose.nii.gz')
        if os.path.exists(out_path):
            print(f"Skipping {patient_id} (already exists)")
            continue
        os.makedirs(out_patient_dir, exist_ok=True)

        ct_path = os.path.join(patient_dir, 'ct.nii.gz')
        if not os.path.exists(ct_path):
            print(f"Warning: CT not found for {patient_id}; skipping")
            continue

        ct_nii = nib.load(ct_path)
        ct_data = ct_nii.get_fdata().astype(np.float32).transpose(2, 0, 1)
        ct_norm = normalize_ct(ct_data)

        syn_path = os.path.join(patient_dir, 'dose_synthetic.nii.gz')
        if os.path.exists(syn_path):
            syn_data = nib.load(syn_path).get_fdata().astype(np.float32).transpose(2, 0, 1)
            syn_data = normalize_dose_input(syn_data)
        else:
            syn_data = np.zeros_like(ct_norm)

        empty_shape = ct_norm.shape
        mask_channels = []
        for name in mask_names:
            fpath = os.path.join(patient_dir, '{}.nii.gz'.format(name))
            if os.path.exists(fpath):
                m = nib.load(fpath).get_fdata().astype(np.float32).transpose(2, 0, 1)
                if m.shape != empty_shape:
                    m = np.zeros(empty_shape, dtype=np.float32)
            else:
                m = np.zeros(empty_shape, dtype=np.float32)
            mask_channels.append((m > 0).astype(np.float32))
        dis_volume = np.stack(mask_channels, axis=0)

        if args.penalty_mode == 'single':
            # Single-channel penalty == synthetic dose (already loaded and
            # normalized into ``syn_data``). Reuse exactly the same volume.
            penalty_volume = syn_data[np.newaxis, ...].copy()
        else:
            penalty_volume = _load_penalty_volume(patient_dir, empty_shape, penalty_files)

        print(f"Predicting {patient_id}...")
        pred_dose = predict_full_volume(
            flow_model,
            ct_norm, syn_data, dis_volume, penalty_volume,
            patch_size=patch_size,
            steps=args.steps,
            device=device,
        )

        pred_dose = pred_dose.transpose(1, 2, 0)  # (D, H, W) -> (H, W, D)
        pred_nii = nib.Nifti1Image(pred_dose, ct_nii.affine, ct_nii.header)
        nib.save(pred_nii, out_path)
        print(f"Saved: {out_path}")

    print(f"\nAll predictions saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
