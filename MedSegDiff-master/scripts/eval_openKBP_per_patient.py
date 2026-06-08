#!/usr/bin/env python3
"""OpenKBP per-patient evaluation: dose score, DVH score, worst-N listing."""
import os
import sys
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluate_openKBP import (
    denormalize_dose,
    get_3D_Dose_dif,
    get_DVH_metrics,
)

STRUCTURES = [
    'Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid',
    'Esophagus', 'Larynx', 'Mandible', 'PTV70', 'PTV63', 'PTV56',
]


def evaluate_per_patient(prediction_dir, gt_dir, denormalize=True,
                         dose_max=80.0, dose_norm_factor=40.0):
    patient_ids = sorted([
        p for p in os.listdir(prediction_dir)
        if os.path.isdir(os.path.join(prediction_dir, p))
    ])
    rows = []
    for patient_id in tqdm(patient_ids, desc='Evaluating'):
        pred = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(prediction_dir, patient_id, 'dose.nii.gz'))
        )
        if denormalize:
            pred = denormalize_dose(pred, dose_max=dose_max, dose_norm_factor=dose_norm_factor)

        gt = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(gt_dir, patient_id, 'dose.nii.gz'))
        )
        mask = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(gt_dir, patient_id, 'Mask_possible_dose_mask.nii.gz'))
        )
        dose_score = float(get_3D_Dose_dif(pred, gt, mask))

        dvh_errs = []
        for structure_name in STRUCTURES:
            structure_file = os.path.join(gt_dir, patient_id, f'Mask_{structure_name}.nii.gz')
            if not os.path.exists(structure_file):
                continue
            structure_nii = sitk.ReadImage(structure_file, sitk.sitkUInt8)
            structure = sitk.GetArrayFromImage(structure_nii)
            spacing = structure_nii.GetSpacing()
            mode = 'target' if 'PTV' in structure_name else 'OAR'
            pred_dvh = get_DVH_metrics(pred, structure, mode=mode, spacing=spacing)
            gt_dvh = get_DVH_metrics(gt, structure, mode=mode, spacing=spacing)
            for metric in gt_dvh:
                dvh_errs.append(abs(gt_dvh[metric] - pred_dvh[metric]))
        dvh_score = float(np.mean(dvh_errs)) if dvh_errs else float('nan')
        rows.append({
            'patient_id': patient_id,
            'dose_score': dose_score,
            'dvh_score': dvh_score,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_dir', required=True)
    parser.add_argument('--gt_dir', required=True)
    parser.add_argument('--denormalize', type=int, default=0)
    parser.add_argument('--worst_n', type=int, default=10)
    parser.add_argument('--out_txt', type=str, default='')
    args = parser.parse_args()

    rows = evaluate_per_patient(
        args.prediction_dir, args.gt_dir,
        denormalize=bool(args.denormalize),
    )
    dose_scores = [r['dose_score'] for r in rows]
    dvh_all = [r['dvh_score'] for r in rows if not np.isnan(r['dvh_score'])]

    lines = []
    lines.append('=' * 80)
    lines.append('OpenKBP Test Set Evaluation (per-patient)')
    lines.append('=' * 80)
    lines.append(f"Patients: {len(rows)}")
    lines.append(f"Dose Score : {np.mean(dose_scores):.4f} +/- {np.std(dose_scores):.4f} Gy")
    lines.append(f"DVH Score  : {np.mean(dvh_all):.4f} +/- {np.std(dvh_all):.4f} Gy")
    lines.append('')
    lines.append(f"Worst {args.worst_n} by Dose Score:")
    lines.append('-' * 80)
    lines.append(f"{'Rank':<6}{'Patient':<12}{'Dose Score':<14}{'DVH Score':<14}")
    lines.append('-' * 80)
    worst = sorted(rows, key=lambda r: r['dose_score'], reverse=True)[:args.worst_n]
    for i, r in enumerate(worst, 1):
        lines.append(f"{i:<6}{r['patient_id']:<12}{r['dose_score']:<14.4f}{r['dvh_score']:<14.4f}")
    lines.append('')
    lines.append(f"Worst {args.worst_n} by DVH Score:")
    lines.append('-' * 80)
    lines.append(f"{'Rank':<6}{'Patient':<12}{'Dose Score':<14}{'DVH Score':<14}")
    lines.append('-' * 80)
    worst_dvh = sorted(rows, key=lambda r: r['dvh_score'], reverse=True)[:args.worst_n]
    for i, r in enumerate(worst_dvh, 1):
        lines.append(f"{i:<6}{r['patient_id']:<12}{r['dose_score']:<14.4f}{r['dvh_score']:<14.4f}")
    lines.append('=' * 80)

    text = '\n'.join(lines)
    print(text)
    if args.out_txt:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_txt)), exist_ok=True)
        with open(args.out_txt, 'w') as f:
            f.write(text)


if __name__ == '__main__':
    main()
