#!/bin/bash
# Resume v1.1 training from best checkpoint, train 600 more epochs.
# Loads model_best_mae.pth + ema_best_mae.pth from the interrupted run.

set -e

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate dosedata

cd /home/zhaojianxiang/dosediff-zjx/MedSegDiff-master/scripts

PORT=29511

TRAIN_DIR="/data0/zhaojianxiang/preprocessed_data/train-pats_preprocess/"
VAL_DIR="/data0/zhaojianxiang/preprocessed_data/validation-pats_preprocess/"

MODEL_NAME=v1_1_gated_xquery_vit
RESUME_DIR="trained_models/${MODEL_NAME}/MedSegDiff_Flow_3D_OpenKBP_11ch_mc32_bs2_epoch600"
RESUME_FROM="${RESUME_DIR}/model_best_mae.pth"
RESUME_EMA="${RESUME_DIR}/ema_best_mae.pth"

BS=1
EPOCH=600
PATCH_D=128
PATCH_H=128
PATCH_W=128
MODEL_CHANNELS=32
GRAD_ACCUM=6
# Lower peak LR for fine-tuning from a converged checkpoint (same idea as v1 resume)
LR_MAX=3e-5
WEIGHT_DECAY=1e-4
WARMUP_RATIO=0.05
MIN_LR=1e-7
GRAD_CLIP=1.0

EMA_DECAY=0.999
VAL_EVERY=25
VAL_STEPS=10
SAVE_EVERY=25

SUFFIX="continue600_from_e375"

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="train_${MODEL_NAME}_resume_${TS}.log"
PID_FILE="run_${MODEL_NAME}_resume.pid"

echo "=================================================================="
echo "v1.1 RESUME from best (epoch 375, val MAE 2.8711 Gy)"
echo "=================================================================="
echo "Resume from : ${RESUME_FROM}"
echo "Resume EMA  : ${RESUME_EMA}"
echo "New save    : trained_models/${MODEL_NAME}/MedSegDiff_Flow_3D_OpenKBP_11ch_mc${MODEL_CHANNELS}_bs2_epoch${EPOCH}_${SUFFIX}/"
echo "Epochs      : ${EPOCH}"
echo "LR sched    : warmup_ratio=${WARMUP_RATIO}, lr_max=${LR_MAX}, min_lr=${MIN_LR}"
echo "Log         : ${LOG_FILE}"
echo "=================================================================="

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
    --resume_from "${RESUME_FROM}" \
    --resume_ema "${RESUME_EMA}" \
    --save_name_suffix "${SUFFIX}" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo $PID > "${PID_FILE}"

echo ""
echo "v1.1 resume training started, PID=${PID}"
echo "Tip: tail -f ${LOG_FILE}"
