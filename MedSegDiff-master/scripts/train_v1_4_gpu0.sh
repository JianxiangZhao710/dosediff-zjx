#!/bin/bash
# train_v1_4_gpu0.sh
# Launch v1.4 (ControlNet-lite Penalty Adapter) training on a single GPU,
# resuming from a v1.2 checkpoint (strict=False so adapter + zero-convs
# are trained from scratch).
#
# Usage:
#   bash scripts/train_v1_4_gpu0.sh \
#       --resume_from trained_models/v1_2_gated_xquery_vit/.../model_best_mae.pth \
#       --resume_ema  trained_models/v1_2_gated_xquery_vit/.../ema_best_mae.pth
#
# All flags after the script name are forwarded to dose_train_3d_v1_4.py.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port 29504 \
    scripts/dose_train_3d_v1_4.py \
    --gpu 0 \
    --bs 1 \
    --epoch 400 \
    --val_every 25 \
    --save_every 25 \
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
    --data_root_val   /data0/zhaojianxiang/preprocessed_data/validation-pats_preprocess/ \
    --save_name_suffix "$(date +%Y%m%d_%H%M%S)" \
    "$@"
