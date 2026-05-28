#!/bin/bash
# v1.1 EMA best on OpenKBP test set (physical GPU 1) -> dose/DVH score + worst 10

set -e

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate dosedata

cd /home/zhaojianxiang/dosediff-zjx/MedSegDiff-master/scripts

CKPT_DIR="trained_models/v1_1_gated_xquery_vit/MedSegDiff_Flow_3D_OpenKBP_11ch_mc32_bs2_epoch600_continue600_from_e375"
MODEL_PATH="${CKPT_DIR}/ema_best_mae.pth"
TEST_DATA_DIR="/data0/zhaojianxiang/preprocessed_data/test-pats_preprocess"
OUTPUT_DIR="test_results/v1_1_ema_best_continue_ep275_steps20_gpu1"
EVAL_FILE="${OUTPUT_DIR}_eval.txt"
WORST_FILE="${OUTPUT_DIR}_worst10.txt"
PHYSICAL_GPU=1
STEPS=20

echo "=================================================================="
echo "v1.1 EMA best inference + evaluation on physical GPU ${PHYSICAL_GPU}"
echo "=================================================================="
echo "Model     : ${MODEL_PATH}"
echo "Output    : ${OUTPUT_DIR}"
echo "=================================================================="

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU} python dose_predict_3d.py \
    --model_name v1_1_gated_xquery_vit \
    --model_path "${MODEL_PATH}" \
    --data_dir "${TEST_DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --patch_size 128 128 128 \
    --batch_size 1 \
    --steps ${STEPS} \
    --gpu 0 \
    --model_channels 32

python ../evaluate_openKBP.py \
    --prediction_dir "$(realpath "${OUTPUT_DIR}")" \
    --gt_dir "${TEST_DATA_DIR}" \
    --denormalize 0 2>&1 | tee "${EVAL_FILE}"

export OUTPUT_DIR TEST_DATA_DIR WORST_FILE
python - <<'PYEOF'
import os, sys
sys.path.insert(0, '..')
import numpy as np
import SimpleITK as sitk
from evaluate_openKBP import get_3D_Dose_dif, get_DVH_metrics

pred_dir = os.environ['OUTPUT_DIR']
gt_dir = os.environ['TEST_DATA_DIR']
structures = [
    'Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid', 'Esophagus',
    'Larynx', 'Mandible', 'PTV70', 'PTV63', 'PTV56',
]
dose_scores, dvh_scores = {}, {}
for pid in sorted(os.listdir(pred_dir)):
    pdir = os.path.join(pred_dir, pid)
    if not os.path.isdir(pdir):
        continue
    pred_path = os.path.join(pdir, 'dose.nii.gz')
    if not os.path.exists(pred_path):
        continue
    pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
    gt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_dir, pid, 'dose.nii.gz')))
    mask = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(gt_dir, pid, 'Mask_possible_dose_mask.nii.gz'))
    )
    dose_scores[pid] = get_3D_Dose_dif(pred, gt, mask)
    dvh_errs = []
    for sn in structures:
        sf = os.path.join(gt_dir, pid, f'Mask_{sn}.nii.gz')
        if not os.path.exists(sf):
            continue
        snii = sitk.ReadImage(sf, sitk.sitkUInt8)
        st = sitk.GetArrayFromImage(snii)
        mode = 'target' if 'PTV' in sn else 'OAR'
        p_dvh = get_DVH_metrics(pred, st, mode=mode, spacing=snii.GetSpacing())
        g_dvh = get_DVH_metrics(gt, st, mode=mode, spacing=snii.GetSpacing())
        for m in g_dvh:
            dvh_errs.append(abs(g_dvh[m] - p_dvh[m]))
    dvh_scores[pid] = float(np.mean(dvh_errs)) if dvh_errs else 0.0

worst_dose = sorted(dose_scores.items(), key=lambda x: x[1], reverse=True)[:10]
worst_dvh = sorted(dvh_scores.items(), key=lambda x: x[1], reverse=True)[:10]
out = os.environ['WORST_FILE']
with open(out, 'w') as f:
    f.write('=== Dose worst 10 (Gy MAE) ===\n')
    for pid, s in worst_dose:
        f.write(f'{pid}\t{s:.4f}\n')
    f.write('=== DVH worst 10 (mean abs error Gy) ===\n')
    for pid, s in worst_dvh:
        f.write(f'{pid}\t{s:.4f}\n')
print('=== Dose worst 10 ===')
for pid, s in worst_dose:
    print(f'{pid}\t{s:.4f}')
print('=== DVH worst 10 ===')
for pid, s in worst_dvh:
    print(f'{pid}\t{s:.4f}')
PYEOF

echo "Done. Eval: ${EVAL_FILE}  Worst10: ${WORST_FILE}"
