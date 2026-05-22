#!/bin/bash
# Launch script for the v1.1 velocity-field network (UNetModel_GatedXQueryViT_3D).
#   - Gated encoder fusion (cond_proj + gate_proj, 1x1x1 Conv3D per encoder block;
#     gate_proj weights zero-init, bias = -2.0 -> initial gate ~ 0.12)
#   - X-query bottleneck ViT: Q = X, K/V = condition feature (from vit_cond_proj)
#   - Decoder skip connections use the gated h
#
# All other components (Flow Matching loss, EMA, LR schedule, etc.) are identical to v1.
# Checkpoints go to trained_models/v1_1_gated_xquery_vit/<save_name>/
#
# IMPORTANT: GPU 0 and GPU 3 are currently used by the v1 continuation training.
#            Before launching v1.1, either stop the v1 run (kill $(cat run_gpu03_resume.pid))
#            or change CUDA_VISIBLE_DEVICES to free GPUs.

set -e

# 1. Env
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate dosedata

cd /home/zhaojianxiang/dosediff-zjx/MedSegDiff-master/scripts

# 2. Port (avoid clash with existing trainings on 29508 / 29509)
PORT=29510

# 3. Data
TRAIN_DIR="/data0/zhaojianxiang/preprocessed_data/train-pats_preprocess/"
VAL_DIR="/data0/zhaojianxiang/preprocessed_data/validation-pats_preprocess/"

# 4. Training hyperparameters (same as v1 baseline so the comparison is apples-to-apples)
MODEL_NAME=v1_1_gated_xquery_vit
BS=1
EPOCH=600
PATCH_D=128
PATCH_H=128
PATCH_W=128
MODEL_CHANNELS=32
GRAD_ACCUM=6
LR_MAX=1e-4
WEIGHT_DECAY=1e-4
WARMUP_RATIO=0.05
MIN_LR=1e-6
GRAD_CLIP=1.0

# 5. EMA / val / save
EMA_DECAY=0.999
VAL_EVERY=25
VAL_STEPS=10
SAVE_EVERY=25

# 6. Logging
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="train_${MODEL_NAME}_${TS}.log"
PID_FILE="run_${MODEL_NAME}.pid"

echo "=================================================================="
echo "v1.1 training : ${MODEL_NAME}"
echo "=================================================================="
echo "Train data  : ${TRAIN_DIR}"
echo "Val   data  : ${VAL_DIR}"
echo "Patch       : ${PATCH_D} x ${PATCH_H} x ${PATCH_W}  (full volume)"
echo "Per-GPU bs  : ${BS}    Grad accum: ${GRAD_ACCUM}    eff. batch: $((2 * BS * GRAD_ACCUM))"
echo "Model ch    : ${MODEL_CHANNELS}"
echo "Epoch       : ${EPOCH}"
echo "LR sched    : warmup_ratio=${WARMUP_RATIO}, lr_max=${LR_MAX}, min_lr=${MIN_LR}, wd=${WEIGHT_DECAY}"
echo "EMA decay   : ${EMA_DECAY}"
echo "Val every   : ${VAL_EVERY} (steps=${VAL_STEPS})"
echo "Save every  : ${SAVE_EVERY}"
echo "Checkpoints : trained_models/${MODEL_NAME}/MedSegDiff_Flow_3D_OpenKBP_11ch_mc${MODEL_CHANNELS}_bs2_epoch${EPOCH}/"
echo "Log         : ${LOG_FILE}"
echo "PID         : ${PID_FILE}"
echo "=================================================================="

# 7. GPU & launch
#    Default to GPU 0 + GPU 3 like the v1 runs; change if those are busy.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,3}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup torchrun \
    --nproc_per_node=2 \
    --master_port=${PORT} \
    dose_train_3d.py \
    --model_name ${MODEL_NAME} \
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
echo "v1.1 training started, PID=${PID}"
echo "Tip: tail -f ${LOG_FILE}"
