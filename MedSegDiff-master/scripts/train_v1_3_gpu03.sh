#!/bin/bash
# train_v1_3_gpu03.sh
# 双卡训练 v1.3：GPU 0 + GPU 3，DDP world_size=2，每 50 epoch 验证。
#
# 用法:
#   bash scripts/train_v1_3_gpu03.sh
#
# 可选：从 checkpoint 继续
#   RESUME_FROM=trained_models/.../model_epoch50.pth \
#   RESUME_EMA=trained_models/.../ema_epoch50.pth \
#   bash scripts/train_v1_3_gpu03.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# 使用 mednext 环境（与 v1.1/v1.2 训练一致）
PYTHON="${HOME}/.conda/envs/mednext/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: mednext python not found: $PYTHON" >&2
  exit 1
fi

SUFFIX="$(date +%Y%m%d_%H%M%S)"
EXTRA_ARGS=()
if [[ -n "${RESUME_FROM:-}" ]]; then
  EXTRA_ARGS+=(--resume_from "$RESUME_FROM")
fi
if [[ -n "${RESUME_EMA:-}" ]]; then
  EXTRA_ARGS+=(--resume_ema "$RESUME_EMA")
fi

"$PYTHON" -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29504 \
    scripts/dose_train_3d_v1_3.py \
    --bs 1 \
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
    --patch_size 64 128 128 \
    --ema_decay 0.999 \
    --val_steps 10 \
    --data_root_train /data0/zhaojianxiang/preprocessed_data/train-pats_preprocess/ \
    --data_root_val /data0/zhaojianxiang/preprocessed_data/validation-pats_preprocess/ \
    --save_name_suffix "${SUFFIX}_gpu03x2" \
    "${EXTRA_ARGS[@]}"
