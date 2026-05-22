#!/bin/bash
# v1.1 EMA best on OpenKBP test set -> dose score + DVH score

set -e

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate dosedata

cd /home/zhaojianxiang/dosediff-zjx/MedSegDiff-master/scripts

CKPT_DIR="trained_models/v1_1_gated_xquery_vit/MedSegDiff_Flow_3D_OpenKBP_11ch_mc32_bs2_epoch600_continue600_from_e375"
MODEL_PATH="${CKPT_DIR}/ema_best_mae.pth"
TEST_DATA_DIR="/data0/zhaojianxiang/preprocessed_data/test-pats_preprocess"
GT_DIR="${TEST_DATA_DIR}"

OUTPUT_DIR="test_results/v1_1_ema_best_continue_ep275_steps20"
EVAL_FILE="${OUTPUT_DIR}_eval.txt"
MODEL_CHANNELS=32
PATCH_D=128
PATCH_H=128
PATCH_W=128
STEPS=20
PHYSICAL_GPU=${PHYSICAL_GPU:-1}

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="test_v1_1_ema_best_${TS}.log"

mkdir -p "${OUTPUT_DIR}"

echo "=================================================================="
echo "v1.1 EMA best inference + OpenKBP evaluation"
echo "=================================================================="
echo "Model       : ${MODEL_PATH}"
echo "model_name  : v1_1_gated_xquery_vit"
echo "Val best    : epoch 275, MAE 2.8448 Gy (resume run)"
echo "Test data   : ${TEST_DATA_DIR}"
echo "Output      : ${OUTPUT_DIR}"
echo "Steps       : ${STEPS}"
echo "GPU (phys)  : ${PHYSICAL_GPU}"
echo "Log         : ${LOG_FILE}"
echo "=================================================================="

{
    echo ""
    echo ">>> [1/2] Predict on test set..."
    CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU} python dose_predict_3d.py \
        --model_name v1_1_gated_xquery_vit \
        --model_path "${MODEL_PATH}" \
        --data_dir "${TEST_DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --patch_size ${PATCH_D} ${PATCH_H} ${PATCH_W} \
        --batch_size 1 \
        --steps ${STEPS} \
        --gpu 0 \
        --model_channels ${MODEL_CHANNELS}

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
