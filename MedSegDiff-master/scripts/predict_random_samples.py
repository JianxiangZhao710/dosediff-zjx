#!/usr/bin/env python3
"""
从数据目录随机抽取若干病例，用 Flow Matching 模型预测剂量，
并将预测结果与真值 dose 一并保存到输出目录。
"""
import sys
import os
import argparse
import random
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))

import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm

from guided_diffusion.unet_3d import UNetModel_MS_Former_3D
from flow_matching import FlowMatching


def denormalize_dose(dose, dose_max=80.0, dose_norm_factor=40.0):
    dose = (dose + 1.0) * dose_norm_factor
    dose = np.clip(dose, 0, dose_max)
    return dose


def normalize_ct(img):
    img = np.clip(img, 0, 2500)
    img = img / 1250.0 - 1.0
    return img


def compute_sliding_starts(size, patch, step):
    """Return start indices that always include a final patch flush with the boundary."""
    if size <= patch:
        return [0]

    starts = list(range(0, size - patch + 1, step))
    last_start = size - patch
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def build_blend_weight(patch_size, min_weight=0.2):
    """Separable center-weighted map to reduce stitching artifacts."""
    axes = []
    for length in patch_size:
        if length <= 1:
            axes.append(np.ones((length,), dtype=np.float32))
            continue

        axis = np.hanning(length).astype(np.float32)
        axis = np.maximum(axis, min_weight)
        axes.append(axis)

    weight = (
        axes[0][:, None, None]
        * axes[1][None, :, None]
        * axes[2][None, None, :]
    )
    weight = weight / np.max(weight)
    return weight.astype(np.float32)


def predict_full_volume(
    model,
    ct_volume,
    cond_volume,
    patch_size=(64, 128, 128),
    batch_size=1,
    steps=50,
    device="cuda",
):
    """滑动窗口预测；cond_volume: (5, D, H, W) one-hot"""
    D, H, W = ct_volume.shape
    patch_d, patch_h, patch_w = patch_size

    pred_dose = np.zeros((D, H, W), dtype=np.float32)
    count_map = np.zeros((D, H, W), dtype=np.float32)

    step_d = max(1, int(patch_d * 0.75))
    step_h = max(1, int(patch_h * 0.75))
    step_w = max(1, int(patch_w * 0.75))
    d_starts = compute_sliding_starts(D, patch_d, step_d)
    h_starts = compute_sliding_starts(H, patch_h, step_h)
    w_starts = compute_sliding_starts(W, patch_w, step_w)
    blend_weight = build_blend_weight(patch_size)

    model.eval()
    with torch.no_grad():
        patches = []
        positions = []

        for d in d_starts:
            for h in h_starts:
                for w in w_starts:
                    d_end = min(d + patch_d, D)
                    h_end = min(h + patch_h, H)
                    w_end = min(w + patch_w, W)

                    ct_patch = ct_volume[d:d_end, h:h_end, w:w_end]
                    cond_patch = cond_volume[:, d:d_end, h:h_end, w:w_end]

                    if ct_patch.shape != patch_size:
                        pad_d = patch_d - ct_patch.shape[0]
                        pad_h = patch_h - ct_patch.shape[1]
                        pad_w = patch_w - ct_patch.shape[2]
                        ct_patch = np.pad(
                            ct_patch,
                            ((0, pad_d), (0, pad_h), (0, pad_w)),
                            "constant",
                            constant_values=-1.0,
                        )
                        cond_patch = np.pad(
                            cond_patch,
                            ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
                            "constant",
                            constant_values=0.0
                        )

                    patches.append((ct_patch, cond_patch))
                    positions.append((d, h, w, d_end, h_end, w_end))

        for i in tqdm(range(0, len(patches), batch_size), desc="Patches"):
            batch_patches = patches[i : i + batch_size]
            batch_positions = positions[i : i + batch_size]

            for patch_idx, (ct_patch, cond_patch) in enumerate(batch_patches):
                d, h, w, d_end, h_end, w_end = batch_positions[patch_idx]

                ct_tensor = torch.from_numpy(ct_patch).unsqueeze(0).unsqueeze(0).float().to(device)
                cond_tensor = torch.from_numpy(cond_patch).unsqueeze(0).float().to(device)

                with torch.cuda.device(device):
                    pred_patch = model.sample(ct_tensor, cond_tensor, steps=steps)
                    pred_patch = pred_patch.squeeze(0).squeeze(0).cpu().numpy()

                del ct_tensor, cond_tensor
                torch.cuda.empty_cache()

                actual_d = d_end - d
                actual_h = h_end - h
                actual_w = w_end - w
                pred_patch_crop = pred_patch[:actual_d, :actual_h, :actual_w]
                pred_patch_crop = denormalize_dose(pred_patch_crop)
                weight_crop = blend_weight[:actual_d, :actual_h, :actual_w]

                pred_dose[d:d_end, h:h_end, w:w_end] += pred_patch_crop * weight_crop
                count_map[d:d_end, h:h_end, w:w_end] += weight_crop
                del pred_patch

        count_map[count_map == 0] = 1.0
        pred_dose = pred_dose / count_map

    return pred_dose


def list_valid_patients(data_root):
    need = ("ct.nii.gz", "radio_biology_map.nii.gz", "dose_resampled.nii.gz")
    out = []
    for name in sorted(os.listdir(data_root)):
        pdir = os.path.join(data_root, name)
        if not os.path.isdir(pdir):
            continue
        if all(os.path.isfile(os.path.join(pdir, f)) for f in need):
            out.append(name)
    return out


def build_cond_volume(patient_dir):
    """(D,H,W) -> one-hot (5, D, H, W)"""
    radio_path = os.path.join(patient_dir, "radio_biology_map.nii.gz")
    radio = nib.load(radio_path).get_fdata().astype(np.float32)
    radio = radio.transpose(2, 0, 1)
    radio = np.round(radio).astype(np.int64)
    radio = np.clip(radio, 0, 4)
    num_classes = 5
    return np.eye(num_classes)[radio].transpose(3, 0, 1, 2).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/zhaojianxiang/dosediff-zjx/MedSegDiff-master/scripts/trained_models/MedSegDiff_Flow_3D_bs3_epoch600/model_epoch100.pth",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data0/zhaojianxiang/dosedata/test_128",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data0/zhaojianxiang/dosedata/dose_test",
    )
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 128, 128])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="物理 GPU 编号（默认 1，便于与占用 0/2/3 的训练并行）",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    valid = list_valid_patients(args.data_dir)
    if len(valid) < args.num_samples:
        raise RuntimeError(
            f"有效病例仅 {len(valid)} 个，少于需要的 {args.num_samples}"
        )

    chosen = random.sample(valid, args.num_samples)
    print(f"随机抽取 {args.num_samples} 例 (seed={args.seed}): {chosen}")

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = os.path.join(args.output_dir, "sample_manifest.txt")
    with open(manifest, "w", encoding="utf-8") as f:
        f.write(f"seed={args.seed}\nmodel={args.model_path}\n")
        for pid in chosen:
            f.write(pid + "\n")
    print(f"已写入清单: {manifest}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        torch.cuda.empty_cache()

    patch_size = tuple(args.patch_size)
    if any(s % 16 != 0 for s in patch_size):
        raise ValueError(f"patch_size 必须为 16 的倍数，当前: {patch_size}")
    dis_channels = 5
    model = UNetModel_MS_Former_3D(
        image_size=patch_size,
        in_channels=1,
        ct_channels=1,
        dis_channels=dis_channels,
        model_channels=64,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        channel_mult=(1, 2, 4, 4),
        dims=3,
    )

    ckpt = torch.load(args.model_path, map_location="cpu")
    if any(k.startswith("module.") for k in ckpt.keys()):
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    del ckpt
    model = model.to(device)
    model.eval()
    flow_model = FlowMatching(model)

    src_gt = "dose_resampled.nii.gz"

    for patient_id in chosen:
        patient_dir = os.path.join(args.data_dir, patient_id)
        out_dir = os.path.join(args.output_dir, patient_id)
        os.makedirs(out_dir, exist_ok=True)

        ct_path = os.path.join(patient_dir, "ct.nii.gz")
        ct_nii = nib.load(ct_path)
        ct_data = ct_nii.get_fdata().astype(np.float32).transpose(2, 0, 1)
        ct_data = normalize_ct(ct_data)

        cond_vol = build_cond_volume(patient_dir)

        print(f"\n>>> 预测 {patient_id} ...")
        pred_dose = predict_full_volume(
            flow_model,
            ct_data,
            cond_vol,
            patch_size=patch_size,
            batch_size=1,
            steps=args.steps,
            device=device,
        )
        pred_dose_hw_d = pred_dose.transpose(1, 2, 0)
        pred_nii = nib.Nifti1Image(pred_dose_hw_d, ct_nii.affine, ct_nii.header)
        pred_path = os.path.join(out_dir, "dose_predicted.nii.gz")
        nib.save(pred_nii, pred_path)
        print(f"保存预测: {pred_path}")

        gt_src = os.path.join(patient_dir, src_gt)
        gt_dst = os.path.join(out_dir, "dose_gt.nii.gz")
        shutil.copy2(gt_src, gt_dst)
        print(f"复制真值: {gt_dst}")

        # 同时保留与训练一致的文件名，便于对比
        shutil.copy2(gt_src, os.path.join(out_dir, "dose_resampled.nii.gz"))

    print(f"\n完成。输出根目录: {args.output_dir}")


if __name__ == "__main__":
    main()
