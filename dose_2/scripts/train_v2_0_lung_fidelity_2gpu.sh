#!/usr/bin/env bash
# 2-GPU DDP from-scratch fidelity-first v2.0 lung training.
#
# Usage:
#   cd /root/dose-zjx/dose_2
#   bash scripts/train_v2_0_lung_fidelity_2gpu.sh
#   bash scripts/train_v2_0_lung_fidelity_2gpu.sh --epoch 600
set -euo pipefail

cd "$(dirname "$0")/.."  # -> dose_2/

DATA_ROOT="/root/autodl-tmp/lung_cancer_processed/processed"
OUT_DIR="trained_models/v2_0_lung_fidelity/lung_fidelity_2gpu_ep400"
LOG="logs/train_lung_fidelity_2gpu.log"

mkdir -p logs

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mednext

export CUDA_VISIBLE_DEVICES=0,1

echo "========== FIDELITY 2-GPU TRAIN START $(date -Iseconds) ==========" | tee -a "${LOG}"

torchrun --standalone --nproc_per_node=2 scripts/dose_train_3d_v2_0_lung_fidelity_ddp.py \
    --data_root_train "${DATA_ROOT}/training" \
    --data_root_val   "${DATA_ROOT}/val" \
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
    --save_dir "${OUT_DIR}" \
    --save_name_suffix fidelity_2gpu \
    "$@" 2>&1 | tee -a "${LOG}"
