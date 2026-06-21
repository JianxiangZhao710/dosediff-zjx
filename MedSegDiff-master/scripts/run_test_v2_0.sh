#!/bin/bash
# v2.0 Dynamic Constraint Router 推理脚本 (mednext 环境)
#
# 用法:
#   bash scripts/run_test_v2_0.sh <MODEL_PATH> [extra args...]
# 例:
#   bash scripts/run_test_v2_0.sh \
#       trained_models/v2_0_dynamic_router/<run>/ema_best_mae.pth --save_aux
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mednext

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODEL_PATH="${1:?need MODEL_PATH as first arg}"
shift || true

DATA_DIR="${DATA_DIR:-/root/autodl-tmp/test-pats_preprocess/}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-test_results/v2_0_${TS}}"
LOG="scripts/test_v2_0_${TS}.log"

echo "[v2.0 test] model=${MODEL_PATH}"
echo "[v2.0 test] data=${DATA_DIR} out=${OUT_DIR}"
echo "log=${LOG}"

python scripts/dose_predict_3d_v2_0.py \
    --model_path "${MODEL_PATH}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUT_DIR}" \
    --patch_size 128 128 128 \
    --steps 30 \
    --gpu 0 \
    --model_channels 32 \
    --channel_mult 1 2 4 4 \
    --penalty_mode multi \
    --penalty_files S_target.nii.gz A_oar.nii.gz B_boundary.nii.gz dose_synthetic.nii.gz \
    --penalty_channel_names S_target A_oar B_boundary P_fused \
    --risk_base 0.2 \
    --risk_use_channels A_oar B_boundary P_fused \
    --mode full_v2_0 \
    --compiler_out_channels 4 \
    --compiler_num_res_blocks 2 \
    --relation_dim 128 \
    --router_delta_scale 0.5 \
    "$@" \
    2>&1 | tee "${LOG}"
