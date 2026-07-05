#!/usr/bin/env bash
# v2.0 lung fine-tune: add an additive body-L1 term on top of the frozen
# structure-weighted MSE objective to lower the voxel-wise Dose Score
# while keeping the structure-region (DVH) fit intact.
#
# Starts from the current best checkpoint (NOT from scratch) and runs a
# short small-LR cosine restart. Saves to a NEW dir so the original run
# is untouched. Best model is saved on validation MAE.
#
# Usage:
#   cd /root/dose-zjx/dose_2
#   bash scripts/finetune_v2_0_lung_bodyL1.sh                 # defaults
#   bash scripts/finetune_v2_0_lung_bodyL1.sh --lambda_global_l1 0.3 --epoch 80
set -euo pipefail

cd "$(dirname "$0")/.."  # -> dose_2/

DATA_ROOT="/root/autodl-tmp/lung_cancer_processed/processed"
SRC_DIR="trained_models/v2_0_lung/lung_full_v2_0_ep400"
OUT_DIR="trained_models/v2_0_lung/lung_v2_0_ft_bodyL1"

# Prefer the best checkpoint; fall back to the latest epoch checkpoint.
RESUME_MODEL="${SRC_DIR}/model_best_mae.pth"
RESUME_EMA="${SRC_DIR}/ema_best_mae.pth"

python scripts/dose_train_3d_v2_0_lung.py \
    --data_root_train "${DATA_ROOT}/training" \
    --data_root_val   "${DATA_ROOT}/val" \
    --gpu 0 \
    --bs 1 \
    --epoch 100 \
    --grad_accum_steps 4 \
    --lr_max 2e-5 \
    --min_lr 1e-6 \
    --warmup_ratio 0.05 \
    --patch_size 128 128 128 \
    --model_channels 32 \
    --channel_mult 1 2 4 4 \
    --mode full_v2_0 \
    --penalty_files S_target.nii.gz A_oar.nii.gz B_boundary.nii.gz final_total.nii.gz \
    --penalty_channel_names S_target A_oar B_boundary P_fused \
    --risk_use_channels A_oar B_boundary P_fused \
    --lambda_global_l1 0.3 \
    --disable_penalty_dropout \
    --resume_from "${RESUME_MODEL}" \
    --resume_ema  "${RESUME_EMA}" \
    --resume_epoch 0 \
    --val_every 10 \
    --val_steps 20 \
    --save_every 20 \
    --num_workers 4 \
    --save_dir "${OUT_DIR}" \
    --save_name_suffix ft_bodyL1 \
    "$@"
