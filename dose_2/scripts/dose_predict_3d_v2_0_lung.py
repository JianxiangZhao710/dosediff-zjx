#!/usr/bin/env python3
"""
dose_predict_3d_v2_0_lung.py
============================
v2.0 sliding-window inference on the lung-cancer dataset.

Mirrors the lung training input contract:
    ct          : CT
    syn         : final_total mapped to [-1,1] (coarse dose-attention prior)
    dis (11ch)  : Target (PTV, ch0) + 6 OARs (ch3-8); Body excluded
    penalty(4ch): [S_target, A_oar, B_boundary, final_total->P_fused]

Variable per-case size is handled by 3D sliding-window over D/H/W with
edge padding (no resampling). Predicted dose is written back in the
patient's native RD geometry using the CT affine/header.

Example
-------
    cd /root/dose-zjx/dose_2
    python scripts/dose_predict_3d_v2_0_lung.py \
        --model_path trained_models/v2_0_lung/<run>/ema_best_mae.pth \
        --data_dir   /root/autodl-tmp/lung_cancer_processed/processed/test \
        --output_dir predictions/v2_0_lung_test \
        --steps 30 --gpu 0
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

from guided_diffusion.dose_loader_3d_lung import (
    DEFAULT_LUNG_PENALTY_FILES,
    DEFAULT_LUNG_PENALTY_NAMES,
    DEFAULT_LUNG_PTV_SOURCE,
    DEFAULT_LUNG_OAR_SOURCES,
    DIS_CHANNELS,
)
from guided_diffusion.unet_3d_v2_0 import (
    UNetModel_DynamicConstraintRouter_v2_0,
    resolve_v2_mode,
    V2_MODES,
)
from guided_diffusion.unet_3d_v1_5 import DEFAULT_RISK_USE_CHANNELS
from flow_matching_v2_0 import FlowMatchingV20


def denormalize_dose(dose, dose_max=80.0, norm=40.0):
    return np.clip((dose + 1.0) * norm, 0, dose_max)


def normalize_ct(img):
    return np.clip(img, 0, 2500) / 1250.0 - 1.0


def load_dhw(path, ref_shape=None, binary=False):
    if not os.path.exists(path):
        return None if ref_shape is None else np.zeros(ref_shape, dtype=np.float32)
    arr = nib.load(path).get_fdata().astype(np.float32).transpose(2, 0, 1)
    if ref_shape is not None and arr.shape != ref_shape:
        return np.zeros(ref_shape, dtype=np.float32)
    if binary:
        arr = (arr > 0).astype(np.float32)
    return arr


def max_abs_norm(arr):
    amax = float(np.max(np.abs(arr)))
    return arr / amax if amax > 1e-6 else arr


def _offsets(dim_size, patch_size, overlap_ratio=0.25):
    if patch_size >= dim_size:
        return [0]
    step = max(1, int(patch_size * (1.0 - overlap_ratio)))
    offsets = list(range(0, dim_size - patch_size, step))
    last = dim_size - patch_size
    if not offsets or offsets[-1] != last:
        offsets.append(last)
    return sorted(set(offsets))


def predict_full_volume(flow_model, ct, syn, dis, penalty, patch_size, steps, device):
    D, H, W = ct.shape
    pd, ph, pw = patch_size
    pred_dose = np.zeros((D, H, W), dtype=np.float32)
    count = np.zeros((D, H, W), dtype=np.float32)

    flow_model.eval()
    with torch.no_grad():
        for d in _offsets(D, pd):
            for h in _offsets(H, ph):
                for w in _offsets(W, pw):
                    de, he, we = min(d + pd, D), min(h + ph, H), min(w + pw, W)
                    ct_p = ct[d:de, h:he, w:we]
                    syn_p = syn[d:de, h:he, w:we]
                    dis_p = dis[:, d:de, h:he, w:we]
                    pen_p = penalty[:, d:de, h:he, w:we]

                    if ct_p.shape != tuple(patch_size):
                        pad = (patch_size[0] - ct_p.shape[0],
                               patch_size[1] - ct_p.shape[1],
                               patch_size[2] - ct_p.shape[2])
                        ct_p = np.pad(ct_p, ((0, pad[0]), (0, pad[1]), (0, pad[2])), constant_values=-1.0)
                        syn_p = np.pad(syn_p, ((0, pad[0]), (0, pad[1]), (0, pad[2])), constant_values=-1.0)
                        dis_p = np.pad(dis_p, ((0, 0), (0, pad[0]), (0, pad[1]), (0, pad[2])), constant_values=0.0)
                        pen_p = np.pad(pen_p, ((0, 0), (0, pad[0]), (0, pad[1]), (0, pad[2])), constant_values=0.0)

                    ct_t = torch.from_numpy(ct_p).unsqueeze(0).unsqueeze(0).float().to(device)
                    syn_t = torch.from_numpy(syn_p).unsqueeze(0).unsqueeze(0).float().to(device)
                    dis_t = torch.from_numpy(dis_p).unsqueeze(0).float().to(device)
                    pen_t = torch.from_numpy(pen_p).unsqueeze(0).float().to(device)

                    pred = flow_model.sample(ct_t, syn_t, dis_t, pen_t, steps=steps)
                    pred = pred.squeeze(0).squeeze(0).cpu().numpy()
                    del ct_t, syn_t, dis_t, pen_t
                    torch.cuda.empty_cache()

                    ad, ah, aw = de - d, he - h, we - w
                    pred_dose[d:de, h:he, w:we] += denormalize_dose(pred[:ad, :ah, :aw])
                    count[d:de, h:he, w:we] += 1.0
                    del pred
        count[count == 0] = 1.0
        pred_dose /= count
    return pred_dose


def build_dis(patient_dir, ref_shape, ptv_source, oar_sources):
    dis = np.zeros((DIS_CHANNELS,) + ref_shape, dtype=np.float32)
    dis[0] = load_dhw(os.path.join(patient_dir, ptv_source + ".nii.gz"), ref_shape, binary=True)
    for i, src in enumerate(oar_sources):
        dis[3 + i] = load_dhw(os.path.join(patient_dir, src + ".nii.gz"), ref_shape, binary=True)
    return dis


def main():
    p = argparse.ArgumentParser(description="v2.0 lung dose inference.")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model_channels", type=int, default=32)
    p.add_argument("--channel_mult", type=int, nargs="+", default=[1, 2, 4, 4])

    p.add_argument("--penalty_files", type=str, nargs="*", default=list(DEFAULT_LUNG_PENALTY_FILES))
    p.add_argument("--penalty_channel_names", type=str, nargs="*", default=list(DEFAULT_LUNG_PENALTY_NAMES))
    p.add_argument("--penalty_model_channels", type=int, default=None)
    p.add_argument("--penalty_num_res_blocks", type=int, default=1)

    p.add_argument("--disable_risk_aware_injection", action="store_true")
    p.add_argument("--risk_base", type=float, default=0.2)
    p.add_argument("--risk_use_channels", type=str, nargs="+", default=list(DEFAULT_RISK_USE_CHANNELS))
    p.add_argument("--risk_hard", action="store_true")

    p.add_argument("--mode", type=str, choices=list(V2_MODES), default="full_v2_0")
    p.add_argument("--disable_constraint_compiler", action="store_true")
    p.add_argument("--disable_dynamic_router", action="store_true")
    p.add_argument("--disable_relation_embedding", action="store_true")
    p.add_argument("--compiler_out_channels", type=int, default=4)
    p.add_argument("--compiler_width", type=int, default=None)
    p.add_argument("--compiler_num_res_blocks", type=int, default=2)
    p.add_argument("--relation_dim", type=int, default=128)
    p.add_argument("--compiler_use_cond_context", action="store_true")
    p.add_argument("--router_time_independent", action="store_true")
    p.add_argument("--router_delta_scale", type=float, default=0.5)
    p.add_argument("--router_hidden", type=int, default=None)

    p.add_argument("--ptv_source", type=str, default=DEFAULT_LUNG_PTV_SOURCE)
    p.add_argument("--oar_sources", type=str, nargs="+", default=list(DEFAULT_LUNG_OAR_SOURCES))
    args = p.parse_args()

    mode, use_compiler, use_router = resolve_v2_mode(
        args.mode,
        use_constraint_compiler=not args.disable_constraint_compiler,
        use_dynamic_router=not args.disable_dynamic_router,
    )
    use_relation = (not args.disable_relation_embedding) and use_compiler

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    patch_size = tuple(args.patch_size)
    if any(s % 16 != 0 for s in patch_size):
        raise ValueError(f"patch_size 必须为 16 的倍数: {patch_size}")
    if args.model_channels % 32 != 0:
        raise ValueError(f"--model_channels 必须为 32 的倍数: {args.model_channels}")

    penalty_files = tuple(args.penalty_files)
    penalty_channels = len(penalty_files)
    final_total_file = penalty_files[-1]
    print(f"[v2.0-lung] mode={mode} compiler={use_compiler} router={use_router} "
          f"relation={use_relation} penalty={list(penalty_files)}")

    use_risk = not args.disable_risk_aware_injection
    model = UNetModel_DynamicConstraintRouter_v2_0(
        image_size=patch_size,
        in_channels=1, ct_channels=1, syn_channels=1, dis_channels=DIS_CHANNELS,
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
        penalty_channel_names=tuple(args.penalty_channel_names),
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
    ckpt = torch.load(args.model_path, map_location="cpu")
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"[Load] missing={len(missing)} unexpected={len(unexpected)}")
    del ckpt
    gc.collect()

    model = model.to(device).eval()
    torch.cuda.empty_cache()
    flow_model = FlowMatchingV20(model)

    patient_list = sorted([d for d in os.listdir(args.data_dir)
                           if os.path.isdir(os.path.join(args.data_dir, d))])
    print(f"Found {len(patient_list)} patients in {args.data_dir}")

    for patient_id in tqdm(patient_list, desc="Patients"):
        pdir = os.path.join(args.data_dir, patient_id)
        out_pdir = os.path.join(args.output_dir, patient_id)
        out_path = os.path.join(out_pdir, "dose.nii.gz")
        if os.path.exists(out_path):
            continue
        ct_path = os.path.join(pdir, "ct.nii.gz")
        if not os.path.exists(ct_path):
            print(f"Warning: CT not found for {patient_id}; skipping")
            continue
        os.makedirs(out_pdir, exist_ok=True)

        ct_nii = nib.load(ct_path)
        ct = normalize_ct(ct_nii.get_fdata().astype(np.float32).transpose(2, 0, 1))
        ref_shape = ct.shape

        ft = load_dhw(os.path.join(pdir, final_total_file), ref_shape)
        syn = ft * 2.0 - 1.0

        dis = build_dis(pdir, ref_shape, args.ptv_source, args.oar_sources)

        pen = []
        for fname in penalty_files:
            pen.append(max_abs_norm(load_dhw(os.path.join(pdir, fname), ref_shape)))
        penalty = np.stack(pen, axis=0)

        pred = predict_full_volume(flow_model, ct, syn, dis, penalty,
                                   patch_size=patch_size, steps=args.steps, device=device)
        pred = pred.transpose(1, 2, 0)
        nib.save(nib.Nifti1Image(pred, ct_nii.affine, ct_nii.header), out_path)

    print(f"\nAll predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
