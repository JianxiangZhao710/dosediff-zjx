#!/bin/bash
# train_v1_3_gpu2.sh
# 在第四张卡 (GPU 3) 上从头训练 v1.3，每 50 epoch 验证一次。
#
# 用法:
#   bash scripts/train_v1_3_gpu2.sh
#
# 验证集结果保存在 trained_models/v1_3_gated_xquery_vit/.../

set -euo pipefail

# 固定使用第四张 GPU (cuda:3) — GPU 0-2 被其他训练占用
export CUDA_VISIBLE_DEVICES=3
# 减少显存碎片，提升大模型承载能力
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port 29503 \
    scripts/dose_train_3d_v1_3.py \
    --gpu 3 \
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
    --save_name_suffix "$(date +%Y%m%d_%H%M%S)"
