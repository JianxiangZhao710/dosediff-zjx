#!/bin/bash
# 从 epoch100 checkpoint 在 GPU0 单卡续训（128^3），学习率按已训练 epoch 对齐。
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CKPT_DIR="trained_models/v1_4_penalty_adapter/MedSegDiff_Flow_3D_OpenKBP_v1_4_mc32_pc1_bs2_epoch600_20260527_235413_gpu0gpu3_p128_bs1"
RESUME_MODEL="${CKPT_DIR}/model_epoch100.pth"
RESUME_EMA="${CKPT_DIR}/ema_epoch100.pth"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="scripts/train_v1_4_resume_gpu0_p128_${TS}.log"

echo "[v1.4 resume] GPU0, patch=128^3, from epoch100, save_dir=${CKPT_DIR}"
echo "log=${LOG}"

nohup python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port 29508 \
    scripts/dose_train_3d_v1_4.py \
    --gpu 0 \
    --bs 1 \
    --epoch 600 \
    --val_every 50 \
    --save_every 50 \
    --grad_accum_steps 8 \
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
    --save_dir "${CKPT_DIR}" \
    --resume_from "${RESUME_MODEL}" \
    --resume_ema "${RESUME_EMA}" \
    "$@" \
    > "${LOG}" 2>&1 &

echo "PID=$!  LOG=${LOG}"
