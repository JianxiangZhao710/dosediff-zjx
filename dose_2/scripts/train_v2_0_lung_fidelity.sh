#!/usr/bin/env bash
# From-scratch fidelity-first v2.0 lung training.
# Objective: make predictions match the GT label as closely as possible
# (voxel-wise Dose Score), using flattened spatial weights + clean-dose
# L1/L2 reconstruction + full-image anti-smoothing terms.
#
# Usage:
#   cd /root/dose-zjx/dose_2
#   bash scripts/train_v2_0_lung_fidelity.sh
#   bash scripts/train_v2_0_lung_fidelity.sh --epoch 600 --lambda_recon_l1 1.5
set -euo pipefail

cd "$(dirname "$0")/.."  # -> dose_2/

DATA_ROOT="/root/autodl-tmp/lung_cancer_processed/processed"

python scripts/dose_train_3d_v2_0_lung_fidelity.py \
    --data_root_train "${DATA_ROOT}/training" \
    --data_root_val   "${DATA_ROOT}/val" \
    --gpu 0 \
    --bs 1 \
    --epoch 400 \
    --grad_accum_steps 4 \
    --lr_max 1e-4 \
    --min_lr 1e-6 \
    --warmup_ratio 0.05 \
    --patch_size 128 128 128 \
    --model_channels 32 \
    --channel_mult 1 2 4 4 \
    --mode full_v2_0 \
    --penalty_files S_target.nii.gz A_oar.nii.gz B_boundary.nii.gz final_total.nii.gz \
    --penalty_channel_names S_target A_oar B_boundary P_fused \
    --risk_use_channels A_oar B_boundary P_fused \
    --ptv_weight 2.0 \
    --oar_weight 1.5 \
    --bg_weight 1.0 \
    --lambda_recon_l1 1.0 \
    --lambda_recon_l2 0.2 \
    --lambda_grad_loss 0.05 \
    --lambda_hf_loss 0.05 \
    --val_every 50 \
    --val_steps 20 \
    --save_every 50 \
    --num_workers 4 \
    --save_dir trained_models/v2_0_lung_fidelity/lung_fidelity_ep400 \
    --save_name_suffix fidelity \
    "$@"
