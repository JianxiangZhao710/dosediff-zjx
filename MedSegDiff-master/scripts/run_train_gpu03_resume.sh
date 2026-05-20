#!/bin/bash
# Resume training from previous best checkpoint
# Loads model_best_mae.pth (raw model) into the trainable model
# Loads ema_best_mae.pth (EMA shadow) into EMA tracker
# New training name has suffix to avoid overwriting previous run.

set -e

# 1. 激活虚拟环境
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate dosedata

cd /home/zhaojianxiang/dosediff-zjx/MedSegDiff-master/scripts

# 2. 通信端口（错开已有任务）
PORT=29509

# 3. 数据路径
TRAIN_DIR="/data0/zhaojianxiang/preprocessed_data/train-pats_preprocess/"
VAL_DIR="/data0/zhaojianxiang/preprocessed_data/validation-pats_preprocess/"

# 4. Resume 来源（已是当前训练的快照, 不会被新训练覆盖）
RESUME_DIR="trained_models/MedSegDiff_Flow_3D_OpenKBP_11ch_mc32_bs2_epoch600"
RESUME_FROM="${RESUME_DIR}/model_best_mae_epoch575_snapshot.pth"
RESUME_EMA="${RESUME_DIR}/ema_best_mae_epoch575_snapshot.pth"

# 5. 训练超参 (continuation)
BS=1
EPOCH=300              # 继续训练 300 epoch
PATCH_D=128
PATCH_H=128
PATCH_W=128
MODEL_CHANNELS=32
GRAD_ACCUM=6
# 继续训练用比原始 (1e-4) 更低的 peak LR, 避免把已收敛的权重打散
LR_MAX=3e-5
WEIGHT_DECAY=1e-4
WARMUP_RATIO=0.05
MIN_LR=1e-7
GRAD_CLIP=1.0

# 6. EMA & 验证 & 保存
EMA_DECAY=0.999
VAL_EVERY=20           # 数据集小, 300 epoch 拉密一点 val
VAL_STEPS=10
SAVE_EVERY=20

# 7. save_name 后缀, 形成独立保存目录
SUFFIX="continue${EPOCH}_from_e575"

# 8. 日志/PID
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="train_continue_e575_${TS}.log"
PID_FILE="run_gpu03_resume.pid"

echo "=================================================================="
echo "RESUME training from best (epoch 575, val MAE 2.8955 Gy)"
echo "=================================================================="
echo "Resume from : ${RESUME_FROM}"
echo "Resume EMA  : ${RESUME_EMA}"
echo "Train data  : ${TRAIN_DIR}"
echo "Val data    : ${VAL_DIR}"
echo "New save    : trained_models/MedSegDiff_Flow_3D_OpenKBP_11ch_mc${MODEL_CHANNELS}_bs2_epoch${EPOCH}_${SUFFIX}"
echo "Epochs      : ${EPOCH}"
echo "LR sched    : warmup_ratio=${WARMUP_RATIO}, lr_max=${LR_MAX} (lower than original 1e-4), min_lr=${MIN_LR}"
echo "EMA decay   : ${EMA_DECAY}"
echo "Val every   : ${VAL_EVERY} (steps=${VAL_STEPS})"
echo "Save every  : ${SAVE_EVERY}"
echo "GPU         : CUDA_VISIBLE_DEVICES=0,3 (DDP 2 GPU)"
echo "Log         : ${LOG_FILE}"
echo "PID         : ${PID_FILE}"
echo "=================================================================="

# 9. 启动
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
    --resume_from "${RESUME_FROM}" \
    --resume_ema "${RESUME_EMA}" \
    --save_name_suffix "${SUFFIX}" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo $PID > "${PID_FILE}"

echo ""
echo "Resume training started, PID=${PID}"
echo "实用命令:"
echo "  tail -f ${LOG_FILE}"
echo "  watch -n 2 nvidia-smi -i 0,3"
echo "  kill \$(cat ${PID_FILE})"
