#!/bin/bash
# Resume v1.1 on the first available GPU among 0,1,2,3 (single-GPU).
# Waits until any card has enough free memory, then starts training.

set -e

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate dosedata

cd /home/zhaojianxiang/dosediff-zjx/MedSegDiff-master/scripts

TRAIN_DIR="/data0/zhaojianxiang/preprocessed_data/train-pats_preprocess/"
VAL_DIR="/data0/zhaojianxiang/preprocessed_data/validation-pats_preprocess/"

MODEL_NAME=v1_1_gated_xquery_vit
RESUME_DIR="trained_models/${MODEL_NAME}/MedSegDiff_Flow_3D_OpenKBP_11ch_mc32_bs2_epoch600_continue600_from_e375"
RESUME_FROM="${RESUME_DIR}/model_best_mae.pth"
RESUME_EMA="${RESUME_DIR}/ema_best_mae.pth"

BS=1
EPOCH=600
PATCH_D=128
PATCH_H=128
PATCH_W=128
MODEL_CHANNELS=32
GRAD_ACCUM=12
LR_MAX=3e-5
WEIGHT_DECAY=1e-4
WARMUP_RATIO=0.05
MIN_LR=1e-7
GRAD_CLIP=1.0

EMA_DECAY=0.999
VAL_EVERY=25
VAL_STEPS=10
SAVE_EVERY=25

# Used memory below this (MiB) => treat GPU as idle enough for v1.1 (~15GB peak)
GPU_FREE_MB=${GPU_FREE_MB:-2000}
GPU_POLL_SEC=${GPU_POLL_SEC:-60}

pick_free_gpu() {
    local best_gpu="" best_used=999999 used gpu
    for gpu in 0 1 2 3; do
        used=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        if [ -z "${used}" ]; then
            continue
        fi
        if [ "${used}" -lt "${GPU_FREE_MB}" ] && [ "${used}" -lt "${best_used}" ]; then
            best_used="${used}"
            best_gpu="${gpu}"
        fi
    done
    if [ -n "${best_gpu}" ]; then
        echo "${best_gpu}"
        return 0
    fi
    return 1
}

print_gpu_status() {
    echo "$(date '+%F %T') GPU memory (MiB):"
    for gpu in 0 1 2 3; do
        used=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        util=$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        echo "  GPU ${gpu}: used=${used:-?}  util=${util:-?}%"
    done
}

echo "=================================================================="
echo "v1.1 RESUME — auto-pick free GPU (0/1/2/3)"
echo "=================================================================="
echo "Resume from : ${RESUME_FROM}"
echo "Resume EMA  : ${RESUME_EMA}"
echo "Best so far : epoch 275, val MAE 2.8448 Gy (continue600_from_e375)"
echo "Free threshold: memory.used < ${GPU_FREE_MB} MiB"
echo "Poll interval : ${GPU_POLL_SEC}s"
echo "=================================================================="

PHYSICAL_GPU=""
while true; do
    if PHYSICAL_GPU=$(pick_free_gpu); then
        used=$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
        echo "Selected physical GPU ${PHYSICAL_GPU} (${used} MiB used)."
        break
    fi
    print_gpu_status
    echo "No free GPU yet, retry in ${GPU_POLL_SEC}s..."
    sleep "${GPU_POLL_SEC}"
done

SUFFIX="continue600_gpu${PHYSICAL_GPU}_from_ep275"
PORT=$((29520 + PHYSICAL_GPU))

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="train_${MODEL_NAME}_resume_gpu${PHYSICAL_GPU}_${TS}.log"
PID_FILE="run_${MODEL_NAME}_resume_auto.pid"

echo "New save    : trained_models/${MODEL_NAME}/MedSegDiff_Flow_3D_OpenKBP_11ch_mc${MODEL_CHANNELS}_bs1_epoch${EPOCH}_${SUFFIX}/"
echo "CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU}  master_port=${PORT}"
echo "Log         : ${LOG_FILE}"
echo "=================================================================="

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup torchrun \
    --nproc_per_node=1 \
    --master_port="${PORT}" \
    dose_train_3d.py \
    --model_name "${MODEL_NAME}" \
    --bs "${BS}" \
    --epoch "${EPOCH}" \
    --patch_size "${PATCH_D}" "${PATCH_H}" "${PATCH_W}" \
    --model_channels "${MODEL_CHANNELS}" \
    --grad_accum_steps "${GRAD_ACCUM}" \
    --lr_max "${LR_MAX}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --min_lr "${MIN_LR}" \
    --grad_clip "${GRAD_CLIP}" \
    --data_root_train "${TRAIN_DIR}" \
    --data_root_val "${VAL_DIR}" \
    --ema_decay "${EMA_DECAY}" \
    --val_every "${VAL_EVERY}" \
    --val_steps "${VAL_STEPS}" \
    --save_every "${SAVE_EVERY}" \
    --resume_from "${RESUME_FROM}" \
    --resume_ema "${RESUME_EMA}" \
    --save_name_suffix "${SUFFIX}" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"
echo ""
echo "v1.1 resume started on GPU ${PHYSICAL_GPU}, torchrun PID=${PID}"
echo "Tip: tail -f ${LOG_FILE}"
