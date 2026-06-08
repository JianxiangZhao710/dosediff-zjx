#!/bin/bash
# v1.5 EMA best on OpenKBP test set -> dose score + DVH score
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mednext

cd /root/dose-zjx/MedSegDiff-master/scripts

CKPT_DIR="../trained_models/v1_5_risk_penalty_adapter/MedSegDiff_Flow_3D_OpenKBP_v1_5_mc32_pc4_bs1_epoch600_local_20260530_014452_gpu0_p128_riskaware"
MODEL_PATH="${CKPT_DIR}/ema_best_mae.pth"
TEST_DATA_DIR="/root/autodl-tmp/test-pats_preprocess"
GT_DIR="${TEST_DATA_DIR}"
OUTPUT_DIR="test_results/v1_5_ema_best_ep150"
EVAL_FILE="${OUTPUT_DIR}_eval.txt"
STEPS=20
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="test_v1_5_ema_best_${TS}.log"

mkdir -p "${OUTPUT_DIR}"

echo "Model       : ${MODEL_PATH}"
echo "Test data   : ${TEST_DATA_DIR}"
echo "Output      : ${OUTPUT_DIR}"
echo "Steps       : ${STEPS}"
echo "Log         : ${LOG_FILE}"

{
    echo ">>> [1/2] Predict on test set..."
    python dose_predict_3d_v1_5.py \
        --model_path "${MODEL_PATH}" \
        --data_dir "${TEST_DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --patch_size 128 128 128 \
        --steps ${STEPS} \
        --gpu 0 \
        --model_channels 32 \
        --channel_mult 1 2 4 4 \
        --penalty_mode multi \
        --penalty_files S_target.nii.gz A_oar.nii.gz B_boundary.nii.gz dose_synthetic.nii.gz \
        --penalty_channel_names S_target A_oar B_boundary P_fused \
        --risk_base 0.2 \
        --risk_use_channels A_oar B_boundary P_fused

    echo ""
    echo ">>> [2/2] OpenKBP evaluation..."
    python ../evaluate_openKBP.py \
        --prediction_dir "$(realpath "${OUTPUT_DIR}")" \
        --gt_dir "${GT_DIR}" \
        --denormalize 0 2>&1 | tee "${EVAL_FILE}"

    echo ""
    echo ">>> Done. Evaluation: ${EVAL_FILE}"
} > "${LOG_FILE}" 2>&1

echo "Finished. See ${LOG_FILE} and ${EVAL_FILE}"
