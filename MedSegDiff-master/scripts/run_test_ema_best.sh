#!/bin/bash
# 用 EMA best-MAE 权重对 OpenKBP test set 做预测 + 评估 (dose MAE + DVH)
# 训练还在跑, 所以使用 ema_best_mae_epoch475_snapshot.pth (已固定的副本)
# 推理放在 GPU 2 (空闲约 8.5GB), 不影响 GPU 0/3 上的训练

set -e

# 1. 激活虚拟环境
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate dosedata

cd /home/zhaojianxiang/dosediff-zjx/MedSegDiff-master/scripts

# 2. 配置
CKPT_DIR="trained_models/MedSegDiff_Flow_3D_OpenKBP_11ch_mc32_bs2_epoch600"
MODEL_PATH="${CKPT_DIR}/ema_best_mae_epoch475_snapshot.pth"
TEST_DATA_DIR="/data0/zhaojianxiang/preprocessed_data/test-pats_preprocess"
GT_DIR="/data0/zhaojianxiang/preprocessed_data/test-pats_preprocess"

OUTPUT_DIR="test_results/openkbp_ema_best_epoch475_steps20"
MODEL_CHANNELS=32
PATCH_D=128
PATCH_H=128
PATCH_W=128
STEPS=20            # Flow Matching Euler 采样步数
BATCH_SIZE=1
PHYSICAL_GPU=2      # 物理 GPU 编号; --gpu 0 是 CUDA_VISIBLE_DEVICES 之后的逻辑编号

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="test_ema_best_epoch475_${TS}.log"
# 重要: evaluation 输出放在 OUTPUT_DIR 的同级目录, 避免 evaluator 把它当患者目录
EVAL_FILE="${OUTPUT_DIR%/}_eval.txt"

mkdir -p "${OUTPUT_DIR}"

echo "=================================================================="
echo "EMA best-MAE inference + OpenKBP evaluation"
echo "=================================================================="
echo "Model       : ${MODEL_PATH}"
echo "Test data   : ${TEST_DATA_DIR}"
echo "GT data     : ${GT_DIR}"
echo "Output      : ${OUTPUT_DIR}"
echo "Patch       : ${PATCH_D} x ${PATCH_H} x ${PATCH_W}"
echo "Model ch    : ${MODEL_CHANNELS}"
echo "Sample steps: ${STEPS}"
echo "GPU (phys)  : ${PHYSICAL_GPU}"
echo "Log         : ${LOG_FILE}"
echo "=================================================================="

# 3. 预测 + 评估串行执行
{
    echo ""
    echo ">>> [1/2] Predict on test set..."
    echo ""
    CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU} python dose_predict_3d.py \
        --model_path "${MODEL_PATH}" \
        --data_dir "${TEST_DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --patch_size ${PATCH_D} ${PATCH_H} ${PATCH_W} \
        --batch_size ${BATCH_SIZE} \
        --steps ${STEPS} \
        --gpu 0 \
        --model_channels ${MODEL_CHANNELS}

    echo ""
    echo ">>> [2/2] OpenKBP evaluation (dose MAE + DVH)..."
    echo ""
    python ../evaluate_openKBP.py \
        --prediction_dir "$(realpath "${OUTPUT_DIR}")" \
        --gt_dir "${GT_DIR}" \
        --denormalize 0 2>&1 | tee "${EVAL_FILE}"

    echo ""
    echo ">>> Done. Results saved to ${OUTPUT_DIR}/"
    echo ">>> Evaluation summary: ${EVAL_FILE}"
} > "${LOG_FILE}" 2>&1 &

PID=$!
echo $PID > test_ema_best.pid

echo ""
echo "Started in background, PID=${PID}"
echo "实用命令:"
echo "  tail -f ${LOG_FILE}                # 实时日志"
echo "  cat ${EVAL_FILE}                   # 评估结果(完成后)"
echo "  kill \$(cat test_ema_best.pid)      # 停止"
