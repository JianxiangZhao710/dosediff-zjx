#!/bin/bash
# Fresh v2 training on the first available GPU among 0, 1, 3 (single-GPU).
# Polls until any candidate card has low enough memory usage, then starts training.
# GPU 2 is intentionally excluded (typically reserved for other jobs).

set -e

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate dosedata

cd /home/zhaojianxiang/dosediff-zjx/MedSegDiff-master/scripts

TRAIN_DIR="/data0/zhaojianxiang/preprocessed_data/train-pats_preprocess/"
VAL_DIR="/data0/zhaojianxiang/preprocessed_data/validation-pats_preprocess/"

MODEL_NAME=v2_control_swin_flow_unet
GPU_CANDIDATES=(0 1 3)

BS=1
EPOCH=600
PATCH_D=128
PATCH_H=128
PATCH_W=128
MODEL_CHANNELS=32
GRAD_ACCUM=12
LR_MAX=1e-4
WEIGHT_DECAY=1e-4
WARMUP_RATIO=0.05
MIN_LR=1e-6
GRAD_CLIP=1.0

EMA_DECAY=0.999
VAL_EVERY=25
VAL_STEPS=10
SAVE_EVERY=25

# Treat GPU as idle when memory.used is below this (MiB). v2 single-GPU ~14GB peak.
GPU_FREE_MB=${GPU_FREE_MB:-2000}
GPU_POLL_SEC=${GPU_POLL_SEC:-60}

pick_free_gpu() {
    local best_gpu="" best_used=999999 used gpu
    for gpu in "${GPU_CANDIDATES[@]}"; do
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
    for gpu in "${GPU_CANDIDATES[@]}"; do
        used=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        util=$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        echo "  GPU ${gpu}: used=${used:-?}  util=${util:-?}%"
    done
}

echo "=================================================================="
echo "v2 FRESH training — auto-pick free GPU (0/1/3)"
echo "=================================================================="
echo "Model       : ${MODEL_NAME}"
echo "Train data  : ${TRAIN_DIR}"
echo "Free threshold: memory.used < ${GPU_FREE_MB} MiB"
echo "Poll interval : ${GPU_POLL_SEC}s"
echo "Epochs      : ${EPOCH}  lr_max=${LR_MAX}  eff.batch=${GRAD_ACCUM}"
echo "=================================================================="

PHYSICAL_GPU=""
while true; do
    if PHYSICAL_GPU=$(pick_free_gpu); then
        used=$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
        echo "Selected physical GPU ${PHYSICAL_GPU} (${used} MiB used)."
        break
    fi
    print_gpu_status
    echo "No free GPU among 0/1/3 yet, retry in ${GPU_POLL_SEC}s..."
    sleep "${GPU_POLL_SEC}"
done

SUFFIX="fresh_gpu${PHYSICAL_GPU}"
PORT=$((29520 + PHYSICAL_GPU))

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="train_${MODEL_NAME}_auto_gpu${PHYSICAL_GPU}_${TS}.log"
PID_FILE="run_${MODEL_NAME}_auto_gpu.pid"

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
    --save_name_suffix "${SUFFIX}" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"

echo ""
echo "v2 fresh training started on GPU ${PHYSICAL_GPU}, torchrun PID=${PID}"
echo "Tip: tail -f ${LOG_FILE}"
