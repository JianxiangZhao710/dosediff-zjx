#!/usr/bin/env python3
"""One-off: dump compiled_field_c* for a single lung case (center 128^3 patch)."""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from guided_diffusion.dose_loader_3d_lung import (
    DEFAULT_LUNG_PENALTY_FILES,
    DEFAULT_LUNG_PENALTY_NAMES,
    DEFAULT_LUNG_PTV_SOURCE,
    DEFAULT_LUNG_OAR_SOURCES,
    DIS_CHANNELS,
)
from guided_diffusion.unet_3d_v1_5 import DEFAULT_RISK_USE_CHANNELS

CHANNEL_NAMES = ("C_pos", "C_neg", "C_boundary", "C_fused")


def _build_model(version, patch_size):
    penalty_files = DEFAULT_LUNG_PENALTY_FILES
    common = dict(
        image_size=patch_size,
        in_channels=1, ct_channels=1, syn_channels=1, dis_channels=DIS_CHANNELS,
        model_channels=32, out_channels=1,
        num_res_blocks=2, attention_resolutions=(8, 16),
        channel_mult=(1, 2, 4, 4), dims=3,
        penalty_channels=len(penalty_files),
        penalty_num_res_blocks=1,
        use_penalty_dropout=False,
        use_risk_aware_injection=True,
        risk_base=0.2,
        risk_use_channels=tuple(DEFAULT_RISK_USE_CHANNELS),
        penalty_channel_names=tuple(DEFAULT_LUNG_PENALTY_NAMES),
    )
    if version == "v2_2":
        from guided_diffusion.unet_3d_v2_2 import UNetModel_DynamicConstraintRouter_v2_2
        return UNetModel_DynamicConstraintRouter_v2_2(**common, mode="full_v2_2"), penalty_files
    if version == "v2_1":
        from guided_diffusion.unet_3d_v2_1 import UNetModel_DynamicConstraintRouter_v2_1
        return UNetModel_DynamicConstraintRouter_v2_1(**common, mode="full_v2_1"), penalty_files
    from guided_diffusion.unet_3d_v2_0 import UNetModel_DynamicConstraintRouter_v2_0
    return UNetModel_DynamicConstraintRouter_v2_0(**common, mode="full_v2_0"), penalty_files


def normalize_ct(img):
    return np.clip(img, 0, 2500) / 1250.0 - 1.0


def load_dhw(path, ref_shape=None, binary=False):
    if not os.path.exists(path):
        return np.zeros(ref_shape, dtype=np.float32) if ref_shape is not None else None
    arr = nib.load(path).get_fdata().astype(np.float32).transpose(2, 0, 1)
    if ref_shape is not None and arr.shape != ref_shape:
        return np.zeros(ref_shape, dtype=np.float32)
    if binary:
        arr = (arr > 0).astype(np.float32)
    return arr


def max_abs_norm(arr):
    amax = float(np.max(np.abs(arr)))
    return arr / amax if amax > 1e-6 else arr


def center_patch(vol, patch_size, channel_first=False):
    if channel_first:
        c, d, h, w = vol.shape
        cd = max(0, (d - patch_size[0]) // 2)
        ch = max(0, (h - patch_size[1]) // 2)
        cw = max(0, (w - patch_size[2]) // 2)
        out = vol[:, cd:cd + patch_size[0], ch:ch + patch_size[1], cw:cw + patch_size[2]]
        pad = (
            (0, 0),
            (0, max(0, patch_size[0] - out.shape[1])),
            (0, max(0, patch_size[1] - out.shape[2])),
            (0, max(0, patch_size[2] - out.shape[3])),
        )
        if any(p[1] > 0 for p in pad):
            out = np.pad(out, pad, mode="constant", constant_values=0.0)
        return out
    d, h, w = vol.shape
    cd = max(0, (d - patch_size[0]) // 2)
    ch = max(0, (h - patch_size[1]) // 2)
    cw = max(0, (w - patch_size[2]) // 2)
    out = vol[cd:cd + patch_size[0], ch:ch + patch_size[1], cw:cw + patch_size[2]]
    pad = (
        (0, max(0, patch_size[0] - out.shape[0])),
        (0, max(0, patch_size[1] - out.shape[1])),
        (0, max(0, patch_size[2] - out.shape[2])),
    )
    if any(p[1] > 0 for p in pad):
        out = np.pad(out, pad, mode="constant", constant_values=0.0)
    return out


def save_nifti(arr, path, affine):
    if arr.ndim == 3:
        nib.save(nib.Nifti1Image(arr.transpose(1, 2, 0).astype(np.float32), affine), path)
    else:
        for c in range(arr.shape[0]):
            p = path.replace(".nii.gz", f"_c{c}.nii.gz")
            nib.save(nib.Nifti1Image(arr[c].transpose(1, 2, 0).astype(np.float32), affine), p)


def masked_mean(field, mask):
    m = mask > 0
    if m.sum() == 0:
        return float("nan")
    return float(field[m].mean())


def masked_corr(a, b, mask):
    m = mask > 0
    if m.sum() < 10:
        return float("nan")
    x, y = a[m].ravel(), b[m].ravel()
    if x.std() < 1e-8 or y.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _load_model(model_path, patch_size, device, version="v2_0"):
    model, penalty_files = _build_model(version, patch_size)
    ckpt = torch.load(model_path, map_location="cpu")
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    return model.to(device).eval(), penalty_files


@torch.no_grad()
def inspect_one(model, penalty_files, patient_dir, output_dir, patch_size, device, aux_t, save_nii=True):
    os.makedirs(output_dir, exist_ok=True)

    ct_nii = nib.load(os.path.join(patient_dir, "ct.nii.gz"))
    ct = normalize_ct(ct_nii.get_fdata().astype(np.float32).transpose(2, 0, 1))
    ref = ct.shape

    ft = load_dhw(os.path.join(patient_dir, penalty_files[-1]), ref)
    syn = ft * 2.0 - 1.0

    dis = np.zeros((DIS_CHANNELS,) + ref, dtype=np.float32)
    dis[0] = load_dhw(os.path.join(patient_dir, DEFAULT_LUNG_PTV_SOURCE + ".nii.gz"), ref, binary=True)
    for i, src in enumerate(DEFAULT_LUNG_OAR_SOURCES):
        dis[3 + i] = load_dhw(os.path.join(patient_dir, src + ".nii.gz"), ref, binary=True)

    pen = [max_abs_norm(load_dhw(os.path.join(patient_dir, f), ref)) for f in penalty_files]
    penalty = np.stack(pen, axis=0)

    ptv = dis[0]
    oar = np.clip(dis[3:10].sum(axis=0), 0, 1)
    body = load_dhw(os.path.join(patient_dir, "resolved/resolved_Body.nii.gz"), ref, binary=True)

    ct_p = center_patch(ct, patch_size)
    syn_p = center_patch(syn, patch_size)
    dis_p = center_patch(dis, patch_size, channel_first=True)
    pen_p = center_patch(penalty, patch_size, channel_first=True)
    ptv_p = center_patch(ptv, patch_size)
    oar_p = center_patch(oar, patch_size)
    body_p = center_patch(body, patch_size)
    # match inference padding for CT/SYN
    if ct_p.shape != patch_size:
        pad = tuple(p - s for p, s in zip(patch_size, ct_p.shape))
        ct_p = np.pad(ct_p, ((0, pad[0]), (0, pad[1]), (0, pad[2])), constant_values=-1.0)
        syn_p = np.pad(syn_p, ((0, pad[0]), (0, pad[1]), (0, pad[2])), constant_values=-1.0)

    ct_t = torch.from_numpy(ct_p).unsqueeze(0).unsqueeze(0).float().to(device)
    syn_t = torch.from_numpy(syn_p).unsqueeze(0).unsqueeze(0).float().to(device)
    dis_t = torch.from_numpy(dis_p).unsqueeze(0).float().to(device)
    pen_t = torch.from_numpy(pen_p).unsqueeze(0).float().to(device)
    x = torch.randn(1, 1, *patch_size, device=device)
    t = torch.ones(1, device=device) * aux_t

    model.collect_aux = True
    _ = model(x, t * 1000.0, ct_t, syn_t, dis_t, pen_t)
    aux = model._aux
    model.collect_aux = False

    compiled = aux["compiled"].squeeze(0).cpu().numpy()  # (4, D, H, W)
    affine = ct_nii.affine
    if save_nii:
        save_nifti(compiled, os.path.join(output_dir, "compiled_field.nii.gz"), affine)
        if "m_risk" in aux:
            save_nifti(aux["m_risk"].squeeze(0).cpu().numpy(), os.path.join(output_dir, "m_risk_base.nii.gz"), affine)
        save_nifti(pen_p, os.path.join(output_dir, "penalty_basis.nii.gz"), affine)

    basis_names = ("S_target", "A_oar", "B_boundary", "P_fused")
    pid = os.path.basename(patient_dir)
    print(f"\n=== Patient: {pid} | vol={ref} | patch={patch_size} | t={aux_t} ===")
    print(f"{'channel':<12} {'global_mean':>12} {'PTV_mean':>12} {'OAR_mean':>12} {'BG_mean':>12} {'corr_PTV':>10} {'corr_OAR':>10}")
    bg_mask = (body_p > 0) & (ptv_p == 0) & (oar_p == 0)

    rows = []
    for i, name in enumerate(CHANNEL_NAMES):
        ch = compiled[i]
        row = {
            "patient": pid, "channel": name, "kind": "compiled",
            "global_mean": ch.mean(),
            "ptv_mean": masked_mean(ch, ptv_p),
            "oar_mean": masked_mean(ch, oar_p),
            "bg_mean": masked_mean(ch, bg_mask),
            "corr_ptv": masked_corr(ch, ptv_p, body_p),
            "corr_oar": masked_corr(ch, oar_p, body_p),
            "nonzero_frac": float((np.abs(ch) > 1e-6).mean()),
        }
        rows.append(row)
        print(
            f"{name:<12} {row['global_mean']:12.4f} {row['ptv_mean']:12.4f} "
            f"{row['oar_mean']:12.4f} {row['bg_mean']:12.4f} "
            f"{row['corr_ptv']:10.3f} {row['corr_oar']:10.3f}"
        )

    print("\n--- penalty basis (input) ---")
    for i, name in enumerate(basis_names):
        ch = pen_p[i]
        row = {
            "patient": pid, "channel": name, "kind": "basis",
            "global_mean": ch.mean(),
            "ptv_mean": masked_mean(ch, ptv_p),
            "oar_mean": masked_mean(ch, oar_p),
            "bg_mean": masked_mean(ch, bg_mask),
            "corr_ptv": masked_corr(ch, ptv_p, body_p),
            "corr_oar": masked_corr(ch, oar_p, body_p),
            "nonzero_frac": float((np.abs(ch) > 1e-6).mean()),
        }
        rows.append(row)
        print(
            f"{name:<12} {row['global_mean']:12.4f} {row['ptv_mean']:12.4f} "
            f"{row['oar_mean']:12.4f} {row['bg_mean']:12.4f} "
            f"{row['corr_ptv']:10.3f} {row['corr_oar']:10.3f}"
        )

    if save_nii:
        print(f"\nSaved NIfTI to: {output_dir}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--patient_dir", type=str)
    g.add_argument("--data_dir", type=str, help="batch mode: process sorted subdirs")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_patients", type=int, default=5)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--aux_t", type=float, default=0.5)
    p.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--no_save_nii", action="store_true")
    p.add_argument("--version", type=str, choices=("v2_0", "v2_1", "v2_2"), default="v2_0",
                   help="model/compiler version (v2_2 = basis gating + bounded boundary)")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    patch_size = tuple(args.patch_size)
    os.makedirs(args.output_dir, exist_ok=True)

    model, penalty_files = _load_model(args.model_path, patch_size, device, version=args.version)
    model.eval()

    if args.patient_dir:
        patients = [args.patient_dir]
    else:
        patients = sorted(
            os.path.join(args.data_dir, d)
            for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d))
        )[: args.max_patients]

    all_rows = []
    for patient_dir in patients:
        pid = os.path.basename(patient_dir)
        out = os.path.join(args.output_dir, pid)
        rows = inspect_one(
            model, penalty_files, patient_dir, out, patch_size, device, args.aux_t,
            save_nii=not args.no_save_nii,
        )
        all_rows.extend(rows)

    # summary CSV
    import csv
    csv_path = os.path.join(args.output_dir, "compiled_field_summary.csv")
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n[Summary] {len(patients)} patients -> {csv_path}")


if __name__ == "__main__":
    main()
