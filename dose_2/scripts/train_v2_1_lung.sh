#!/usr/bin/env bash
# v2.1 lung-cancer dose training (compiler: no P_fused shortcut + softplus pos/neg).
#
# Usage:
#   cd /root/dose-zjx/dose_2
#   bash scripts/train_v2_1_lung.sh
#   bash scripts/train_v2_1_lung.sh --epoch 400 --gpu 0
#   bash scripts/train_v2_1_lung.sh --resume_from trained_models/v2_0_lung/.../ema_best_mae.pth
set -euo pipefail

cd "$(dirname "$0")/.."  # -> dose_2/

DATA_ROOT="/root/autodl-tmp/lung_cancer_processed/processed"

python scripts/dose_train_3d_v2_1_lung.py \
    --data_root_train "${DATA_ROOT}/training" \
    --data_root_val   "${DATA_ROOT}/val" \
    --gpu 0 \
    --bs 1 \
    --epoch 600 \
    --grad_accum_steps 4 \
    --lr_max 1e-4 \
    --patch_size 128 128 128 \
    --model_channels 32 \
    --channel_mult 1 2 4 4 \
    --mode full_v2_1 \
    --penalty_files S_target.nii.gz A_oar.nii.gz B_boundary.nii.gz final_total.nii.gz \
    --penalty_channel_names S_target A_oar B_boundary P_fused \
    --risk_use_channels A_oar B_boundary P_fused \
    --val_every 25 \
    --val_steps 20 \
    --save_every 50 \
    "$@"
