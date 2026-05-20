#!/bin/bash
# 双卡训练脚本: GPU 0 + GPU 3 (两张 RTX 4080 16GB)
# OpenKBP 11-channel + Flow Matching, 整卷 128^3 输入
# 包含: EMA + 验证集 MAE + best checkpoint + 周期性 ckpt

set -e

# 1. 激活虚拟环境
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate dosedata

# 2. 切到脚本目录
cd /home/zhaojianxiang/dosediff-zjx/MedSegDiff-master/scripts

# 3. 通信端口（避免与已有任务冲突）
PORT=29508

# 4. 数据路径 (OpenKBP 预处理集)
TRAIN_DIR="/data0/zhaojianxiang/preprocessed_data/train-pats_preprocess/"
VAL_DIR="/data0/zhaojianxiang/preprocessed_data/validation-pats_preprocess/"

# 5. 训练超参
BS=1                   # per-GPU batch size
EPOCH=600
PATCH_D=128
PATCH_H=128
PATCH_W=128
MODEL_CHANNELS=32      # UNet 基础通道数 (16GB 显存下的安全值)
GRAD_ACCUM=6           # 2 GPU * bs1 * accum6 = 等效 batch 12
LR_MAX=1e-4
WEIGHT_DECAY=1e-4
WARMUP_RATIO=0.05
MIN_LR=1e-6
GRAD_CLIP=1.0

# 6. EMA & 验证 & 保存
EMA_DECAY=0.999        # 设 0 关闭 EMA
VAL_EVERY=25           # 每 N epoch 跑一次 val MAE
VAL_STEPS=10           # FM Euler 采样步数 (val 用, 10 步即可)
SAVE_EVERY=25          # 每 N epoch 保存一次普通 ckpt

# 7. 日志/PID
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="train_openkbp_11ch_gpu03_mc${MODEL_CHANNELS}_${TS}.log"
PID_FILE="run_gpu03.pid"

echo "=================================================================="
echo "Starting OpenKBP 11-channel training on GPU 0 + GPU 3 (DDP)"
echo "=================================================================="
echo "Data train  : ${TRAIN_DIR}"
echo "Data val    : ${VAL_DIR}"
echo "Patch       : ${PATCH_D} x ${PATCH_H} x ${PATCH_W} (full volume)"
echo "Per-GPU bs  : ${BS}"
echo "World size  : 2 (GPU 0 + GPU 3)"
echo "Grad accum  : ${GRAD_ACCUM}   -> effective batch = 2 * ${BS} * ${GRAD_ACCUM} = $((2 * BS * GRAD_ACCUM))"
echo "Model ch    : ${MODEL_CHANNELS}"
echo "Epoch       : ${EPOCH}"
echo "LR sched    : warmup_ratio=${WARMUP_RATIO}, lr_max=${LR_MAX}, min_lr=${MIN_LR}, wd=${WEIGHT_DECAY}"
echo "EMA decay   : ${EMA_DECAY}"
echo "Val every   : ${VAL_EVERY} (steps=${VAL_STEPS})"
echo "Save every  : ${SAVE_EVERY}"
echo "Log file    : ${LOG_FILE}"
echo "PID file    : ${PID_FILE}"
echo "=================================================================="

# 8. 启动: 只暴露 GPU 0 和 GPU 3
export CUDA_VISIBLE_DEVICES=0,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup torchrun \
    --nproc_per_node=2 \
    --master_port=${PORT} \
    dose_train_3d.py \
    --bs ${BS} \
    --epoch ${EPOCH} \
    --patch_size ${PATCH_D} ${PATCH_H} ${PATCH_W} \
    --model_channels ${MODEL_CHANNELS} \
    --grad_accum_steps ${GRAD_ACCUM} \
    --lr_max "${LR_MAX}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --min_lr "${MIN_LR}" \
    --grad_clip "${GRAD_CLIP}" \
    --data_root_train "${TRAIN_DIR}" \
    --data_root_val "${VAL_DIR}" \
    --ema_decay "${EMA_DECAY}" \
    --val_every ${VAL_EVERY} \
    --val_steps ${VAL_STEPS} \
    --save_every ${SAVE_EVERY} \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo $PID > "${PID_FILE}"

echo ""
echo "Training started with PID: ${PID}"
echo ""
echo "实用命令:"
echo "  tail -f ${LOG_FILE}"
echo "  watch -n 2 nvidia-smi -i 0,3"
echo "  kill \$(cat ${PID_FILE})"
