#!/usr/bin/env bash
# Single-GPU v2.2 lung training (compiler: basis gating + bounded boundary).
#
# Usage:
#   cd /root/dose-zjx/dose_2
#   bash scripts/train_v2_2_lung.sh
set -euo pipefail

cd "$(dirname "$0")/.."  # -> dose_2/

DATA_ROOT="/root/autodl-tmp/lung_cancer_processed/processed"
RESUME="trained_models/v2_0_lung/lung_full_v2_0_ep400/ema_best_mae.pth"
OUT_DIR="trained_models/v2_2_lung/lung_v2_2_gate_reset_ep80"
LOG="logs/train_lung_v2_2.log"

mkdir -p logs

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mednext

export CUDA_VISIBLE_DEVICES=0

echo "========== v2.2 TRAIN START $(date -Iseconds) ==========" | tee -a "${LOG}"

python scripts/dose_train_3d_v2_2_lung.py \
    --data_root_train "${DATA_ROOT}/training" \
    --data_root_val   "${DATA_ROOT}/val" \
    --gpu 0 \
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
    --penalty_files S_target.nii.gz A_oar.nii.gz B_boundary.nii.gz final_total.nii.gz \
    --penalty_channel_names S_target A_oar B_boundary P_fused \
    --risk_use_channels A_oar B_boundary P_fused \
    --resume_from "${RESUME}" \
    --reset_compiler \
    --val_every 10 \
    --val_steps 20 \
    --save_every 20 \
    --num_workers 4 \
    --save_dir "${OUT_DIR}" \
    --save_name_suffix gate_reset_ep80 \
    "$@" 2>&1 | tee -a "${LOG}"
