#!/usr/bin/env python3
"""
dose_predict_3d_v2_0.py
=======================
v2.0 (Dynamic Relation-to-Field Constraint Router) full-volume
sliding-window inference.

Same I/O contract as ``dose_predict_3d_v1_5.py`` — only the model class,
the v2.0 ablation configuration and the optional aux-visualisation dump
differ. Penalty dropout is force-disabled at inference; the constraint
compiler and dynamic router stay active so high-risk regions receive the
learned dynamic guidance.

Optional ``--save_aux``: for the first ``--aux_num`` patients, run a
single forward on the central 128^3 patch (at ``--aux_t``) with the
model's ``collect_aux`` flag on and dump, as NIfTI:
    * M_risk_base (base risk map prior)
    * compiled constraint field channels [C_pos, C_neg, C_boundary, C_fused]
    * per-level router weights (upsampled to the patch size)
so you can verify the router really uses the prior more strongly in
OAR / boundary / conflict regions.
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
import torch.nn.functional as F
from tqdm import tqdm

from guided_diffusion.dose_loader_3d import OPENKBP_MASK_NAMES
from guided_diffusion.dose_loader_3d_v1_5 import (
    DEFAULT_V15_PENALTY_FILES,
    infer_penalty_name,
)
from guided_diffusion.unet_3d_v2_0 import (
    UNetModel_DynamicConstraintRouter_v2_0,
    resolve_v2_mode,
    V2_MODES,
)
from guided_diffusion.unet_3d_v1_5 import DEFAULT_RISK_USE_CHANNELS
from flow_matching_v2_0 import FlowMatchingV20


def denormalize_dose(dose, dose_max=80.0, dose_norm_factor=40.0):
    dose = (dose + 1.0) * dose_norm_factor
    return np.clip(dose, 0, dose_max)


def normalize_ct(img):
    img = np.clip(img, 0, 2500)
    return img / 1250.0 - 1.0


def normalize_dose_input(img):
    img = np.clip(img, 0, 80)
    return img / 40.0 - 1.0


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


def predict_full_volume(flow_model, ct_volume, syn_volume, dis_volume, penalty_volume,
                        patch_size=(128, 128, 128), steps=30, device='cuda'):
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
                    d_end, h_end, w_end = min(d + patch_d, D), min(h + patch_h, H), min(w + patch_w, W)
                    ct_patch = ct_volume[d:d_end, h:h_end, w:w_end]
                    syn_patch = syn_volume[d:d_end, h:h_end, w:w_end]
                    dis_patch = dis_volume[:, d:d_end, h:h_end, w:w_end]
                    pen_patch = penalty_volume[:, d:d_end, h:h_end, w:w_end]

                    if ct_patch.shape != patch_size:
                        pad_d = patch_d - ct_patch.shape[0]
                        pad_h = patch_h - ct_patch.shape[1]
                        pad_w = patch_w - ct_patch.shape[2]
                        ct_patch = np.pad(ct_patch, ((0, pad_d), (0, pad_h), (0, pad_w)), 'constant', constant_values=-1.0)
                        syn_patch = np.pad(syn_patch, ((0, pad_d), (0, pad_h), (0, pad_w)), 'constant', constant_values=-1.0)
                        dis_patch = np.pad(dis_patch, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), 'constant', constant_values=0.0)
                        pen_patch = np.pad(pen_patch, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), 'constant', constant_values=0.0)

                    ct_t = torch.from_numpy(ct_patch).unsqueeze(0).unsqueeze(0).float().to(device)
                    syn_t = torch.from_numpy(syn_patch).unsqueeze(0).unsqueeze(0).float().to(device)
                    dis_t = torch.from_numpy(dis_patch).unsqueeze(0).float().to(device)
                    pen_t = torch.from_numpy(pen_patch).unsqueeze(0).float().to(device)

                    pred = flow_model.sample(ct_t, syn_t, dis_t, pen_t, steps=steps)
                    pred = pred.squeeze(0).squeeze(0).cpu().numpy()

                    del ct_t, syn_t, dis_t, pen_t
                    torch.cuda.empty_cache()

                    actual_d, actual_h, actual_w = d_end - d, h_end - h, w_end - w
                    pred_crop = denormalize_dose(pred[:actual_d, :actual_h, :actual_w])
                    pred_dose[d:d_end, h:h_end, w:w_end] += pred_crop
                    count_map[d:d_end, h:h_end, w:w_end] += 1.0
                    del pred

        count_map[count_map == 0] = 1.0
        pred_dose = pred_dose / count_map
    return pred_dose


def _load_penalty_volume(patient_dir, ref_shape, penalty_files, normalize=True):
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


def _center_patch(vol, patch, channel_first=False):
    """Center crop/pad a volume to ``patch`` size."""
    if channel_first:
        C = vol.shape[0]
        spatial = vol.shape[1:]
    else:
        spatial = vol.shape
    starts = [max(0, (s - p) // 2) for s, p in zip(spatial, patch)]
    sl = tuple(slice(st, st + p) for st, p in zip(starts, patch))
    if channel_first:
        out = vol[(slice(None),) + sl]
        pad = [(0, 0)] + [(0, max(0, p - o)) for p, o in zip(patch, out.shape[1:])]
    else:
        out = vol[sl]
        pad = [(0, max(0, p - o)) for p, o in zip(patch, out.shape)]
    if any(p[1] > 0 for p in pad):
        out = np.pad(out, pad, 'constant', constant_values=0.0)
    return out


@torch.no_grad()
def save_aux_for_patch(raw_model, flow_model, ct, syn, dis, penalty, out_dir, affine,
                       aux_t=0.5, patch_size=(128, 128, 128), device='cuda'):
    """Run one forward on a central patch with collect_aux and dump maps."""
    os.makedirs(out_dir, exist_ok=True)
    ct_p = _center_patch(ct, patch_size)
    syn_p = _center_patch(syn, patch_size)
    dis_p = _center_patch(dis, patch_size, channel_first=True)
    pen_p = _center_patch(penalty, patch_size, channel_first=True)

    ct_t = torch.from_numpy(ct_p).unsqueeze(0).unsqueeze(0).float().to(device)
    syn_t = torch.from_numpy(syn_p).unsqueeze(0).unsqueeze(0).float().to(device)
    dis_t = torch.from_numpy(dis_p).unsqueeze(0).float().to(device)
    pen_t = torch.from_numpy(pen_p).unsqueeze(0).float().to(device)

    b = 1
    t = torch.ones(b, device=device) * aux_t
    x = torch.randn(b, 1, *patch_size, device=device)

    raw_model.collect_aux = True
    _ = raw_model(x, t * 1000.0, ct_t, syn_t, dis_t, pen_t)
    aux = raw_model._aux
    raw_model.collect_aux = False

    def _save(arr_t, name):
        arr = arr_t.squeeze(0).cpu().numpy()  # (C, d,h,w) or (1,d,h,w)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        arr = np.asarray(arr, dtype=np.float32)
        # move to (H, W, D) for nibabel if 3D
        if arr.ndim == 3:
            nib.save(nib.Nifti1Image(arr.transpose(1, 2, 0), affine), os.path.join(out_dir, name))
        else:
            for c in range(arr.shape[0]):
                nib.save(nib.Nifti1Image(arr[c].transpose(1, 2, 0), affine),
                         os.path.join(out_dir, name.replace('.nii.gz', f'_c{c}.nii.gz')))

    if 'm_risk' in aux:
        _save(aux['m_risk'], 'm_risk_base.nii.gz')
    if 'compiled' in aux:
        _save(aux['compiled'], 'compiled_field.nii.gz')
    if 'router_weights' in aux:
        for li, w in enumerate(aux['router_weights']):
            w_up = F.interpolate(w, size=patch_size, mode='trilinear', align_corners=False)
            _save(w_up, f'router_weight_l{li}.nii.gz')

    del ct_t, syn_t, dis_t, pen_t
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='v2.0 dynamic constraint router inference')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--patch_size', type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model_channels', type=int, default=32)
    parser.add_argument('--channel_mult', type=int, nargs='+', default=[1, 2, 4, 4])

    # Penalty branch
    parser.add_argument('--penalty_mode', type=str, choices=['single', 'multi'], default='multi')
    parser.add_argument('--penalty_files', type=str, nargs='*', default=None)
    parser.add_argument('--penalty_channels', type=int, default=None)
    parser.add_argument('--penalty_channel_names', type=str, nargs='*', default=None)
    parser.add_argument('--penalty_model_channels', type=int, default=None)
    parser.add_argument('--penalty_num_res_blocks', type=int, default=1)

    # v1.5 risk prior
    parser.add_argument('--disable_risk_aware_injection', action='store_true')
    parser.add_argument('--risk_base', type=float, default=0.2)
    parser.add_argument('--risk_use_channels', type=str, nargs='+', default=list(DEFAULT_RISK_USE_CHANNELS))
    parser.add_argument('--risk_hard', action='store_true')

    # v2.0 modules / ablation
    parser.add_argument('--mode', type=str, choices=list(V2_MODES), default='full_v2_0')
    parser.add_argument('--disable_constraint_compiler', action='store_true')
    parser.add_argument('--disable_dynamic_router', action='store_true')
    parser.add_argument('--disable_relation_embedding', action='store_true')
    parser.add_argument('--compiler_out_channels', type=int, default=4)
    parser.add_argument('--compiler_width', type=int, default=None)
    parser.add_argument('--compiler_num_res_blocks', type=int, default=2)
    parser.add_argument('--relation_dim', type=int, default=128)
    parser.add_argument('--compiler_use_cond_context', action='store_true')
    parser.add_argument('--router_time_independent', action='store_true')
    parser.add_argument('--router_delta_scale', type=float, default=0.5)
    parser.add_argument('--router_hidden', type=int, default=None)

    # Aux visualisation
    parser.add_argument('--save_aux', action='store_true')
    parser.add_argument('--aux_num', type=int, default=3)
    parser.add_argument('--aux_t', type=float, default=0.5)

    args = parser.parse_args()

    mode, use_compiler, use_router = resolve_v2_mode(
        args.mode,
        use_constraint_compiler=not args.disable_constraint_compiler,
        use_dynamic_router=not args.disable_dynamic_router,
    )
    use_relation = (not args.disable_relation_embedding) and use_compiler

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

    if args.penalty_mode == 'single':
        penalty_files = ()
        penalty_channels = 1
        penalty_channel_names = args.penalty_channel_names or ['P_fused']
        print("[v2.0] Penalty mode=single (penalty == dose_synthetic.nii.gz == P_fused coarse prior)")
    else:
        penalty_files = tuple(args.penalty_files) if args.penalty_files else DEFAULT_V15_PENALTY_FILES
        penalty_channels = args.penalty_channels if args.penalty_channels is not None else len(penalty_files)
        if args.penalty_channel_names:
            penalty_channel_names = list(args.penalty_channel_names)
        else:
            penalty_channel_names = [infer_penalty_name(f) for f in penalty_files]
        print(f"[v2.0] Penalty mode=multi files={list(penalty_files)} names={penalty_channel_names}")

    use_risk = (not args.disable_risk_aware_injection)
    print(f"[v2.0] mode={mode} compiler={use_compiler} router={use_router} relation={use_relation}")

    dis_channels = len(OPENKBP_MASK_NAMES)

    model = UNetModel_DynamicConstraintRouter_v2_0(
        image_size=patch_size,
        in_channels=1, ct_channels=1, syn_channels=1, dis_channels=dis_channels,
        model_channels=args.model_channels, out_channels=1,
        num_res_blocks=2, attention_resolutions=(8, 16),
        channel_mult=tuple(args.channel_mult), dims=3,
        penalty_channels=penalty_channels,
        penalty_model_channels=args.penalty_model_channels,
        penalty_num_res_blocks=args.penalty_num_res_blocks,
        use_penalty_dropout=False,
        use_risk_aware_injection=use_risk,
        risk_base=args.risk_base,
        risk_use_channels=tuple(args.risk_use_channels),
        risk_soft=(not args.risk_hard),
        penalty_channel_names=tuple(penalty_channel_names),
        mode=mode,
        use_relation_embedding=use_relation,
        compiler_out_channels=args.compiler_out_channels,
        compiler_width=args.compiler_width,
        compiler_num_res_blocks=args.compiler_num_res_blocks,
        relation_dim=args.relation_dim,
        compiler_use_cond_context=args.compiler_use_cond_context,
        router_time_dependent=(not args.router_time_independent),
        router_delta_scale=args.router_delta_scale,
        router_hidden=args.router_hidden,
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

    flow_model = FlowMatchingV20(model)

    patient_list = sorted([p for p in os.listdir(args.data_dir)
                           if os.path.isdir(os.path.join(args.data_dir, p))])
    print(f"Found {len(patient_list)} patients in {args.data_dir}")
    mask_names = list(OPENKBP_MASK_NAMES)

    for p_idx, patient_id in enumerate(tqdm(patient_list, desc="Patients")):
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
            syn_data = normalize_dose_input(
                nib.load(syn_path).get_fdata().astype(np.float32).transpose(2, 0, 1))
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
            penalty_volume = syn_data[np.newaxis, ...].copy()
        else:
            penalty_volume = _load_penalty_volume(patient_dir, empty_shape, penalty_files)

        if args.save_aux and p_idx < args.aux_num:
            try:
                save_aux_for_patch(
                    model, flow_model, ct_norm, syn_data, dis_volume, penalty_volume,
                    out_dir=os.path.join(out_patient_dir, 'aux'), affine=ct_nii.affine,
                    aux_t=args.aux_t, patch_size=patch_size, device=device,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[aux] skip {patient_id}: {e}")

        pred_dose = predict_full_volume(
            flow_model, ct_norm, syn_data, dis_volume, penalty_volume,
            patch_size=patch_size, steps=args.steps, device=device,
        )

        pred_dose = pred_dose.transpose(1, 2, 0)
        nib.save(nib.Nifti1Image(pred_dose, ct_nii.affine, ct_nii.header), out_path)

    print(f"\nAll predictions saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
