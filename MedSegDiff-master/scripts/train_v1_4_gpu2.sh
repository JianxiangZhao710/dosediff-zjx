#!/bin/bash
# train_v1_4_gpu2.sh
# 在第三张卡 (cuda:2) 上训练 v1.4 (ControlNet-lite Penalty Adapter)。
# 默认从 v1.2 checkpoint 以 strict=False 续训，penalty adapter + zero-convs
# 从零开始训练。
#
# 数据约定:
#   train-pats_preprocess/      训练集 -> --data_root_train
#   validation-pats_preprocess/ 测试集 -> --data_root_val
#       (在训练中作为 held-out monitor 跑 MAE，用于挑 best checkpoint)
#   penalty (单通道)            == dose_synthetic.nii.gz，
#       由 dataset 自动复用 SYN 流的同一体素同一归一化，
#       不需要额外的 penalty.nii.gz。
#
# 用法:
#   bash scripts/train_v1_4_gpu2.sh \
#       --resume_from trained_models/v1_2_gated_xquery_vit/.../model_best_mae.pth \
#       --resume_ema  trained_models/v1_2_gated_xquery_vit/.../ema_best_mae.pth
#
# 任何在脚本名之后的 CLI 参数都会原样转发给 dose_train_3d_v1_4.py。
# checkpoint 保存在 trained_models/v1_4_penalty_adapter/.../

set -euo pipefail

# 固定使用第三张 GPU (cuda:2)
export CUDA_VISIBLE_DEVICES=2
# 减少显存碎片，提升大模型承载能力
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# 默认 bs=1；可通过环境变量 TRAIN_BS 或命令行 --bs N 覆盖（勿与脚本内重复写 --bs）
TRAIN_BS="${TRAIN_BS:-1}"
EXTRA_ARGS=("$@")
if [[ ! " ${EXTRA_ARGS[*]} " =~ " --bs " ]]; then
    EXTRA_ARGS=(--bs "${TRAIN_BS}" "${EXTRA_ARGS[@]}")
fi

python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port 29505 \
    scripts/dose_train_3d_v1_4.py \
    --gpu 2 \
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
    "${EXTRA_ARGS[@]}"
