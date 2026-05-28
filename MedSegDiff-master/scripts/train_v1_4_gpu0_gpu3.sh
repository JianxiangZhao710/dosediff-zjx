#!/bin/bash
# v1.4 on GPU0 + GPU3 (DDP x2), patch 128^3 full volume crop.
# 用法: bash scripts/train_v1_4_gpu0_gpu3.sh [额外参数...]

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

TRAIN_BS="${TRAIN_BS:-1}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="scripts/train_v1_4_gpu0_gpu3_p128_${TS}.log"

echo "[v1.4] GPU0+GPU3 DDP, patch=128x128x128, TRAIN_BS=${TRAIN_BS}, log=${LOG}"

nohup python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port 29507 \
    scripts/dose_train_3d_v1_4.py \
    --gpu 0 \
    --bs "${TRAIN_BS}" \
    --epoch 600 \
    --val_every 50 \
    --save_every 50 \
    --grad_accum_steps 4 \
    --warmup_ratio 0.05 \
    --lr_max 1e-4 \
    --min_lr 1e-6 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    --model_channels 32 \
    --channel_mult 1 2 4 4 \
    --patch_size 128 128 128 \
    --ema_decay 0.999 \
    --val_steps 10 \
    --penalty_mode single \
    --penalty_dropout_p 0.2 \
    --adapter_warmup_epochs 0 \
    --data_root_train /data0/zhaojianxiang/preprocessed_data/train-pats_preprocess/ \
    --data_root_val /data0/zhaojianxiang/preprocessed_data/validation-pats_preprocess/ \
    --save_name_suffix "${TS}_gpu0gpu3_p128_bs${TRAIN_BS}" \
    "$@" \
    > "${LOG}" 2>&1 &

echo "PID=$!  LOG=${LOG}"
