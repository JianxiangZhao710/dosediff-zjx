#!/usr/bin/env python3
"""
eval_lung_v2_0.py
=================
Compute Dose Score (3D dose MAE) and DVH Score for the lung-cancer test set.

Adapted from ``MedSegDiff-master/evaluate_openKBP.py`` for the lung dataset
layout:

* Predicted dose : ``<prediction_dir>/<patient>/dose.nii.gz`` (already in Gy,
  written by ``dose_predict_3d_v2_0_lung.py``; no denormalisation needed).
* Ground-truth   : ``<gt_dir>/<patient>/dose.nii.gz`` (native RD geometry, Gy).
* Structures     : ``<gt_dir>/<patient>/resolved/resolved_<name>.nii.gz``.
* Dose region    : Body mask (``resolved_Body``) replaces the OpenKBP
  ``possible_dose_mask`` for the Dose Score region.

Targets (DVH ``target`` mode -> D1/D95/D99):
    Target_Total_from_RS

OARs (DVH ``OAR`` mode -> D_0.1_cc / Dmean):
    Lung_L, Lung_R, Heart, Esophagus, SpinalCord, Trachea
"""
import argparse
import os

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

TARGET_STRUCTURES = ["Target_Total_from_RS"]
OAR_STRUCTURES = ["Lung_L", "Lung_R", "Heart", "Esophagus", "SpinalCord", "Trachea"]
BODY_STRUCTURE = "Body"


def denormalize_dose(dose, dose_max=80.0, dose_norm_factor=40.0):
    dose = (dose + 1.0) * dose_norm_factor
    return np.clip(dose, 0, dose_max)


def get_3D_Dose_dif(pred, gt, possible_dose_mask=None):
    if possible_dose_mask is not None:
        pred = pred[possible_dose_mask > 0]
        gt = gt[possible_dose_mask > 0]
    return float(np.mean(np.abs(pred - gt)))


def get_DVH_metrics(_dose, _mask, mode, spacing=None):
    output = {}
    _roi_dose = _dose[_mask > 0]
    if mode == "target":
        if len(_roi_dose) == 0:
            return {"D1": 0.0, "D95": 0.0, "D99": 0.0}
        output["D1"] = np.percentile(_roi_dose, 99)
        output["D95"] = np.percentile(_roi_dose, 5)
        output["D99"] = np.percentile(_roi_dose, 1)
    elif mode == "OAR":
        if spacing is None:
            raise Exception("calculate OAR metrics need spacing")
        _roi_size = len(_roi_dose)
        if _roi_size == 0:
            return {"D_0.1_cc": 0.0, "mean": 0.0}
        _voxel_size = np.prod(spacing)
        voxels_in_tenth_of_cc = np.maximum(1, np.round(100 / _voxel_size))
        fractional_volume_to_evaluate = 100 - voxels_in_tenth_of_cc / _roi_size * 100
        fractional_volume_to_evaluate = np.clip(fractional_volume_to_evaluate, 0, 100)
        output["D_0.1_cc"] = np.percentile(_roi_dose, fractional_volume_to_evaluate)
        output["mean"] = float(np.mean(_roi_dose))
    else:
        raise Exception("Unknown mode!")
    return output


def _resolved_path(gt_dir, patient_id, name):
    return os.path.join(gt_dir, patient_id, "resolved", f"resolved_{name}.nii.gz")


def get_Dose_score_and_DVH_score(
    prediction_dir,
    gt_dir,
    denormalize=False,
    dose_max=80.0,
    dose_norm_factor=40.0,
    return_detail=False,
):
    list_dose_dif = []
    list_DVH_dif = []
    dvh_dif_dict = {}

    patient_ids = sorted([
        p for p in os.listdir(prediction_dir)
        if os.path.isdir(os.path.join(prediction_dir, p))
    ])

    for patient_id in tqdm(patient_ids):
        pred_path = os.path.join(prediction_dir, patient_id, "dose.nii.gz")
        gt_path = os.path.join(gt_dir, patient_id, "dose.nii.gz")
        if not (os.path.exists(pred_path) and os.path.exists(gt_path)):
            print(f"  [skip] missing dose for {patient_id}")
            continue

        pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
        if denormalize:
            pred = denormalize_dose(pred, dose_max=dose_max, dose_norm_factor=dose_norm_factor)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))

        # Dose Score region: Body mask (fallback to all voxels if absent).
        body_path = _resolved_path(gt_dir, patient_id, BODY_STRUCTURE)
        if os.path.exists(body_path):
            dose_mask = sitk.GetArrayFromImage(sitk.ReadImage(body_path))
        else:
            dose_mask = np.ones_like(gt)
        list_dose_dif.append(get_3D_Dose_dif(pred, gt, dose_mask))

        for structure_name in TARGET_STRUCTURES + OAR_STRUCTURES:
            structure_file = _resolved_path(gt_dir, patient_id, structure_name)
            if not os.path.exists(structure_file):
                continue
            structure_nii = sitk.ReadImage(structure_file, sitk.sitkUInt8)
            structure = sitk.GetArrayFromImage(structure_nii)
            spacing = structure_nii.GetSpacing()
            mode = "target" if structure_name in TARGET_STRUCTURES else "OAR"

            pred_DVH = get_DVH_metrics(pred, structure, mode=mode, spacing=spacing)
            gt_DVH = get_DVH_metrics(gt, structure, mode=mode, spacing=spacing)

            dvh_dif_dict.setdefault(structure_name, {})
            for metric in gt_DVH.keys():
                dvh_dif_dict[structure_name].setdefault(metric, [])
                err = abs(gt_DVH[metric] - pred_DVH[metric])
                dvh_dif_dict[structure_name][metric].append(err)
                list_DVH_dif.append(err)

    dose_mae_mean = float(np.mean(list_dose_dif))
    dose_mae_std = float(np.std(list_dose_dif))
    dvh_score_mean = float(np.mean(list_DVH_dif))
    dvh_score_std = float(np.std(list_DVH_dif))

    dvh_results = {}
    for structure_name, metrics_dict in dvh_dif_dict.items():
        dvh_results[structure_name] = {
            metric: {"mean": float(np.mean(errs)), "std": float(np.std(errs))}
            for metric, errs in metrics_dict.items()
        }

    if return_detail:
        return dose_mae_mean, dose_mae_std, dvh_score_mean, dvh_score_std, dvh_results
    return dose_mae_mean, dose_mae_std, dvh_score_mean, dvh_score_std


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 lung 数据集上的剂量预测结果")
    parser.add_argument("--prediction_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--denormalize", type=int, default=0,
                        help="预测是否需反归一化(1=是)。v2.0 lung 推理已输出 Gy，默认 0")
    parser.add_argument("--dose_max", type=float, default=80.0)
    parser.add_argument("--dose_norm_factor", type=float, default=40.0)
    args = parser.parse_args()

    print("=" * 80)
    print("开始评估 lung 剂量预测结果...")
    print(f"预测目录: {args.prediction_dir}")
    print(f"GT  目录: {args.gt_dir}")
    print(f"反归一化: {'是' if args.denormalize else '否 (预测已为 Gy)'}")
    print("=" * 80)

    dose_mean, dose_std, dvh_mean, dvh_std, dvh_results = get_Dose_score_and_DVH_score(
        args.prediction_dir, args.gt_dir,
        denormalize=bool(args.denormalize),
        dose_max=args.dose_max,
        dose_norm_factor=args.dose_norm_factor,
        return_detail=True,
    )

    print("\n" + "=" * 80)
    print("Dose Score (Body 区域 3D 剂量 MAE):")
    print("=" * 80)
    print(f"Dose_score: {dose_mean:.4f} Gy")
    print(f"Dose_std  : {dose_std:.4f} Gy")

    print("\n" + "=" * 80)
    print("DVH Score (所有结构所有指标的 MAE):")
    print("=" * 80)
    print(f"DVH_score : {dvh_mean:.4f} Gy")
    print(f"DVH_std   : {dvh_std:.4f} Gy")

    print("\n" + "=" * 80)
    print("各结构 DVH 指标的绝对误差详情:")
    print("=" * 80)
    for structure_name, metrics_dict in sorted(dvh_results.items()):
        print(f"\n【{structure_name}】")
        for metric, stats in sorted(metrics_dict.items()):
            print(f"  {metric:10s}: 平均值 = {stats['mean']:8.4f} Gy, 标准差 = {stats['std']:8.4f} Gy")

    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)
