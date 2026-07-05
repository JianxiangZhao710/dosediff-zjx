#!/usr/bin/env bash
# 2-GPU DDP v2.1 lung short training (compiler: no P_fused shortcut + softplus).
#
# Usage:
#   cd /root/dose-zjx/dose_2
#   bash scripts/train_v2_1_lung_2gpu.sh
#   bash scripts/train_v2_1_lung_2gpu.sh --epoch 120
set -euo pipefail

cd "$(dirname "$0")/.."  # -> dose_2/

DATA_ROOT="/root/autodl-tmp/lung_cancer_processed/processed"
RESUME="trained_models/v2_0_lung/lung_full_v2_0_ep400/ema_best_mae.pth"
OUT_DIR="trained_models/v2_1_lung/lung_v2_1_short_2gpu_ep80"
LOG="logs/train_lung_v2_1_short_2gpu.log"

mkdir -p logs

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mednext

export CUDA_VISIBLE_DEVICES=0,1

echo "========== v2.1 2-GPU SHORT TRAIN START $(date -Iseconds) ==========" | tee -a "${LOG}"

torchrun --standalone --nproc_per_node=2 scripts/dose_train_3d_v2_1_lung_ddp.py \
    --data_root_train "${DATA_ROOT}/training" \
    --data_root_val   "${DATA_ROOT}/val" \
    --bs 1 \
    --epoch 80 \
    --grad_accum_steps 4 \
    --lr_max 5e-5 \
    --min_lr 1e-6 \
    --warmup_ratio 0.05 \
    --patch_size 128 128 128 \
    --model_channels 32 \
    --channel_mult 1 2 4 4 \
    --mode full_v2_1 \
    --penalty_files S_target.nii.gz A_oar.nii.gz B_boundary.nii.gz final_total.nii.gz \
    --penalty_channel_names S_target A_oar B_boundary P_fused \
    --risk_use_channels A_oar B_boundary P_fused \
    --resume_from "${RESUME}" \
    --val_every 10 \
    --val_steps 20 \
    --save_every 20 \
    --num_workers 4 \
    --save_dir "${OUT_DIR}" \
    --save_name_suffix short_2gpu \
    "$@" 2>&1 | tee -a "${LOG}"
