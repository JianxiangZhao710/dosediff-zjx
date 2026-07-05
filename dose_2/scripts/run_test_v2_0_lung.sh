#!/usr/bin/env bash
# v2.0 lung-cancer dose inference (sliding window, native RD geometry).
#
# Usage:
#   cd /root/dose-zjx/dose_2
#   bash scripts/run_test_v2_0_lung.sh trained_models/v2_0_lung/<run>/ema_best_mae.pth
#   bash scripts/run_test_v2_0_lung.sh <ckpt> --steps 50 --gpu 0
set -euo pipefail

cd "$(dirname "$0")/.."  # -> dose_2/

CKPT="${1:?Usage: run_test_v2_0_lung.sh <checkpoint.pth> [extra args]}"
shift || true

DATA_ROOT="/root/autodl-tmp/lung_cancer_processed/processed"
OUT_DIR="predictions/v2_0_lung_test"

python scripts/dose_predict_3d_v2_0_lung.py \
    --model_path "${CKPT}" \
    --data_dir   "${DATA_ROOT}/test" \
    --output_dir "${OUT_DIR}" \
    --patch_size 128 128 128 \
    --model_channels 32 \
    --channel_mult 1 2 4 4 \
    --mode full_v2_0 \
    --penalty_files S_target.nii.gz A_oar.nii.gz B_boundary.nii.gz final_total.nii.gz \
    --penalty_channel_names S_target A_oar B_boundary P_fused \
    --risk_use_channels A_oar B_boundary P_fused \
    --steps 30 \
    --gpu 0 \
    "$@"
