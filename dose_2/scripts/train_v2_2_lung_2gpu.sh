#!/usr/bin/env bash
# 2-GPU DDP v2.2 lung training (basis gating + bounded boundary).
# Scheme 3: warm-start v2.0 backbone, reset compiler to random init.
#
# Usage:
#   cd /root/dose-zjx/dose_2
#   bash scripts/train_v2_2_lung_2gpu.sh
set -euo pipefail

cd "$(dirname "$0")/.."  # -> dose_2/

DATA_ROOT="/root/autodl-tmp/lung_cancer_processed/processed"
RESUME="trained_models/v2_0_lung/lung_full_v2_0_ep400/ema_best_mae.pth"
OUT_DIR="trained_models/v2_2_lung/lung_v2_2_gate_reset_2gpu_ep80"
LOG="logs/train_lung_v2_2_2gpu.log"

mkdir -p logs

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mednext

export CUDA_VISIBLE_DEVICES=0,1

echo "========== v2.2 2-GPU TRAIN (reset_compiler) START $(date -Iseconds) ==========" | tee -a "${LOG}"

torchrun --standalone --nproc_per_node=2 scripts/dose_train_3d_v2_2_lung_ddp.py \
    --data_root_train "${DATA_ROOT}/training" \
    --data_root_val   "${DATA_ROOT}/val" \
    --bs 1 \
    --epoch 80 \
    --grad_accum_steps 4 \
    --lr_max 2e-5 \
    --min_lr 1e-6 \
    --warmup_ratio 0.05 \
    --patch_size 128 128 128 \
    --model_channels 32 \
    --channel_mult 1 2 4 4 \
    --mode full_v2_2 \
    --compiler_gate_mode gate \
    --freeze_backbone_epochs 20 \
    --resume_from "${RESUME}" \
    --reset_compiler \
    --val_every 10 \
    --val_steps 20 \
    --save_every 20 \
    --num_workers 4 \
    --save_dir "${OUT_DIR}" \
    --save_name_suffix gate_reset_2gpu \
    "$@" 2>&1 | tee -a "${LOG}"
