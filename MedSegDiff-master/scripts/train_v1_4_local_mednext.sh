#!/bin/bash
# v1.4 本地训练脚本 (mednext 环境, AutoDL 数据路径)
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mednext

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_TRAIN="/root/autodl-tmp/train-pats_preprocess/"
DATA_VAL="/root/autodl-tmp/validation-pats_preprocess/"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="scripts/train_v1_4_local_mednext_${TS}.log"

echo "[v1.4 local] GPU=${CUDA_VISIBLE_DEVICES}, data=${DATA_TRAIN}"
echo "log=${LOG}"

python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port 29504 \
    scripts/dose_train_3d_v1_4.py \
    --gpu 0 \
    --bs 1 \
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
    --penalty_mode multi \
    --penalty_files S_target.nii.gz A_oar.nii.gz B_boundary.nii.gz \
    --penalty_dropout_p 0.2 \
    --adapter_warmup_epochs 0 \
    --data_root_train "${DATA_TRAIN}" \
    --data_root_val "${DATA_VAL}" \
    --save_name_suffix "local_${TS}_gpu0_p128_multi3" \
    "$@" \
    2>&1 | tee "${LOG}"
