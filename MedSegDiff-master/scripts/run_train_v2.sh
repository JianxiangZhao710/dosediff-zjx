#!/bin/bash
# Launch script for the v2 velocity-field network
# (UNetModel_ControlSwinFlow_3D, model_name=v2_control_swin_flow_unet).
#
# Structural changes vs v1.1:
#   - 3-stream encoder (X / CT / DIS)  ->  2-stream encoder
#       X-stream + Condition-stream (input = cat(ct, dis), 12 channels).
#   - Per-block gated residual control (cond_proj C->C, gate_proj 2C->C, bias=-2.0).
#   - Window cross-attention at 32^3 (ds=4) and 16^3 (ds=8); Q=X, K/V=cond.
#   - Decoder skip uses condition-controlled h_x.
#
# All Flow Matching training settings (loss, EMA, scheduler, sampler,
# patch size, batch, etc.) are IDENTICAL to v1.1 for an apples-to-apples
# comparison.

set -e

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate dosedata

cd /home/zhaojianxiang/dosediff-zjx/MedSegDiff-master/scripts

# Physical GPU id(s). Examples: PHYSICAL_GPU=2 (single)  PHYSICAL_GPU=0,3 (dual)
PHYSICAL_GPU=${PHYSICAL_GPU:-0,3}
PORT=${PORT:-29515}

TRAIN_DIR="/data0/zhaojianxiang/preprocessed_data/train-pats_preprocess/"
VAL_DIR="/data0/zhaojianxiang/preprocessed_data/validation-pats_preprocess/"

MODEL_NAME=v2_control_swin_flow_unet
BS=1
EPOCH=600
PATCH_D=128
PATCH_H=128
PATCH_W=128
MODEL_CHANNELS=32
if [[ "${PHYSICAL_GPU}" == *","* ]]; then
    NPROC=2
    GRAD_ACCUM=6
else
    NPROC=1
    GRAD_ACCUM=12
    PORT=$((29520 + PHYSICAL_GPU))
fi
LR_MAX=1e-4
WEIGHT_DECAY=1e-4
WARMUP_RATIO=0.05
MIN_LR=1e-6
GRAD_CLIP=1.0

EMA_DECAY=0.999
VAL_EVERY=25
VAL_STEPS=10
SAVE_EVERY=25

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="train_${MODEL_NAME}_${TS}.log"
PID_FILE="run_${MODEL_NAME}.pid"

echo "=================================================================="
echo "v2 training : ${MODEL_NAME}"
echo "=================================================================="
echo "Train data  : ${TRAIN_DIR}"
echo "Val   data  : ${VAL_DIR}"
echo "Patch       : ${PATCH_D} x ${PATCH_H} x ${PATCH_W}  (full volume)"
echo "GPU (phys)  : ${PHYSICAL_GPU}    nproc=${NPROC}"
echo "Per-GPU bs  : ${BS}    Grad accum: ${GRAD_ACCUM}    eff. batch: $((NPROC * BS * GRAD_ACCUM))"
echo "Model ch    : ${MODEL_CHANNELS}"
echo "Epoch       : ${EPOCH}"
echo "LR sched    : warmup_ratio=${WARMUP_RATIO}, lr_max=${LR_MAX}, min_lr=${MIN_LR}, wd=${WEIGHT_DECAY}"
echo "EMA decay   : ${EMA_DECAY}"
echo "Val every   : ${VAL_EVERY} (steps=${VAL_STEPS})"
echo "Save every  : ${SAVE_EVERY}"
echo "Checkpoints : trained_models/${MODEL_NAME}/MedSegDiff_Flow_3D_OpenKBP_11ch_mc${MODEL_CHANNELS}_bs${NPROC}_epoch${EPOCH}/"
echo "Log         : ${LOG_FILE}"
echo "PID         : ${PID_FILE}"
echo "=================================================================="

export CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup torchrun \
    --nproc_per_node=${NPROC} \
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
echo "v2 training started, PID=${PID}"
echo "Tip: tail -f ${LOG_FILE}"
