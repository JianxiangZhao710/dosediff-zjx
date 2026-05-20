#!/usr/bin/env python3
"""
分析当前预测目录中的 dose_predicted.nii.gz 与 dose_gt.nii.gz。

功能：
1. 检查 pred / gt 的 shape、spacing、affine 是否一致
2. 统计每个病例的 MAE、RMSE、峰值剂量、非零体素数量
3. 标记“真值接近 0，但预测出较大剂量团块”的可疑病例
4. 可选读取 radio_biology_map，帮助判断条件输入是否在诱导模板化输出
"""
import argparse
import csv
import os

import nibabel as nib
import numpy as np


def load_array(path):
    nii = nib.load(path)
    return nii, nii.get_fdata().astype(np.float32)


def safe_unique_counts(arr):
    labels, counts = np.unique(np.round(arr).astype(np.int32), return_counts=True)
    return ";".join(f"{int(k)}:{int(v)}" for k, v in zip(labels, counts))


def compute_case_metrics(pred, gt, dose_thresholds):
    diff = pred - gt
    abs_diff = np.abs(diff)

    metrics = {
        "mae": float(abs_diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "pred_mean": float(pred.mean()),
        "gt_mean": float(gt.mean()),
        "pred_max": float(pred.max()),
        "gt_max": float(gt.max()),
    }

    for thr in dose_thresholds:
        pred_count = int((pred > thr).sum())
        gt_count = int((gt > thr).sum())
        inter = int(np.logical_and(pred > thr, gt > thr).sum())
        denom = pred_count + gt_count
        dice = (2.0 * inter / denom) if denom > 0 else 1.0
        key = str(thr).replace(".", "p")
        metrics[f"pred_vox_gt_{key}Gy"] = pred_count
        metrics[f"gt_vox_gt_{key}Gy"] = gt_count
        metrics[f"dice_gt_{key}Gy"] = float(dice)

    return metrics


def analyze_pair_dir(args):
    case_ids = sorted(
        case_id
        for case_id in os.listdir(args.prediction_dir)
        if case_id.startswith("_")
        and os.path.isdir(os.path.join(args.prediction_dir, case_id))
    )
    if not case_ids:
        raise RuntimeError(f"未在 {args.prediction_dir} 下找到病例目录")

    rows = []
    geometry_issues = []
    suspicious_cases = []

    for case_id in case_ids:
        case_dir = os.path.join(args.prediction_dir, case_id)
        pred_path = os.path.join(case_dir, args.pred_name)
        gt_path = os.path.join(case_dir, args.gt_name)
        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            continue

        pred_nii, pred = load_array(pred_path)
        gt_nii, gt = load_array(gt_path)

        same_shape = pred.shape == gt.shape
        same_zooms = tuple(pred_nii.header.get_zooms()[:3]) == tuple(gt_nii.header.get_zooms()[:3])
        same_affine = bool(np.allclose(pred_nii.affine, gt_nii.affine))
        if not (same_shape and same_zooms and same_affine):
            geometry_issues.append(case_id)

        row = {
            "case_id": case_id,
            "shape_ok": same_shape,
            "spacing_ok": same_zooms,
            "affine_ok": same_affine,
        }
        row.update(compute_case_metrics(pred, gt, args.thresholds))

        if args.input_dir:
            radio_path = os.path.join(args.input_dir, case_id, args.radio_name)
            if os.path.exists(radio_path):
                _, radio = load_array(radio_path)
                row["radio_labels"] = safe_unique_counts(radio)
            else:
                row["radio_labels"] = ""

        if row["gt_max"] < args.low_gt_max and row["pred_max"] > args.high_pred_max:
            suspicious_cases.append(case_id)
            row["flag"] = "hallucinated_template"
        else:
            row["flag"] = ""

        rows.append(row)

    if not rows:
        raise RuntimeError("未找到有效的 pred/gt 成对文件")

    rows_sorted = sorted(rows, key=lambda item: item["mae"], reverse=True)

    print("=" * 80)
    print(f"病例数: {len(rows_sorted)}")
    print(f"预测目录: {args.prediction_dir}")
    print(f"几何异常病例数: {len(geometry_issues)}")
    print(f"可疑幻觉病例数: {len(suspicious_cases)}")
    print("=" * 80)

    maes = np.array([r["mae"] for r in rows_sorted], dtype=np.float32)
    rmses = np.array([r["rmse"] for r in rows_sorted], dtype=np.float32)
    print(f"MAE  mean/std : {maes.mean():.4f} / {maes.std():.4f}")
    print(f"RMSE mean/std : {rmses.mean():.4f} / {rmses.std():.4f}")
    for thr in args.thresholds:
        key = str(thr).replace(".", "p")
        dices = np.array([r[f'dice_gt_{key}Gy'] for r in rows_sorted], dtype=np.float32)
        print(f"Dice@>{thr:g}Gy mean : {dices.mean():.4f}")

    if geometry_issues:
        print("\n几何异常病例:")
        for case_id in geometry_issues:
            print(f"  {case_id}")

    if suspicious_cases:
        print("\n可疑幻觉病例:")
        for case_id in suspicious_cases:
            row = next(r for r in rows_sorted if r["case_id"] == case_id)
            print(
                f"  {case_id} | gt_max={row['gt_max']:.3f} | "
                f"pred_max={row['pred_max']:.3f} | mae={row['mae']:.3f}"
            )

    print("\nMAE 最差的前 5 例:")
    for row in rows_sorted[:5]:
        print(
            f"  {row['case_id']} | mae={row['mae']:.3f} | rmse={row['rmse']:.3f} | "
            f"gt_max={row['gt_max']:.3f} | pred_max={row['pred_max']:.3f} | flag={row['flag'] or '-'}"
        )

    if args.csv_out:
        fieldnames = list(rows_sorted[0].keys())
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_sorted)
        print(f"\n已写出 CSV: {args.csv_out}")


def parse_args():
    parser = argparse.ArgumentParser(description="分析当前 dose 预测结果")
    parser.add_argument(
        "--prediction_dir",
        type=str,
        default="/data0/zhaojianxiang/dosedata/dose_test",
        help="包含每个病例子目录的预测结果目录",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/data0/zhaojianxiang/dosedata/test_128",
        help="原始输入目录，用于读取 radio_biology_map（可选）",
    )
    parser.add_argument(
        "--pred_name",
        type=str,
        default="dose_predicted.nii.gz",
        help="预测 dose 文件名",
    )
    parser.add_argument(
        "--gt_name",
        type=str,
        default="dose_gt.nii.gz",
        help="真值 dose 文件名",
    )
    parser.add_argument(
        "--radio_name",
        type=str,
        default="radio_biology_map.nii.gz",
        help="条件图文件名",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[1.0, 5.0, 10.0],
        help="用于统计体积重叠的剂量阈值列表",
    )
    parser.add_argument(
        "--low_gt_max",
        type=float,
        default=1.0,
        help="真值最大剂量低于该值时，视为接近空白",
    )
    parser.add_argument(
        "--high_pred_max",
        type=float,
        default=20.0,
        help="若同时预测最大剂量高于该值，则标记为可疑幻觉",
    )
    parser.add_argument(
        "--csv_out",
        type=str,
        default="",
        help="可选，输出每例统计 CSV 路径",
    )
    return parser.parse_args()


if __name__ == "__main__":
    analyze_pair_dir(parse_args())
