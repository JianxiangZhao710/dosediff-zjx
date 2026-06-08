#!/bin/bash
# v1.5 Risk-aware ControlNet-lite Penalty Adapter 本地训练脚本 (mednext 环境, AutoDL 数据路径)
#
# penalty 通道顺序 [S_target, A_oar, B_boundary, P_fused]，对应文件:
#   S_target.nii.gz / A_oar.nii.gz / B_boundary.nii.gz / dose_synthetic.nii.gz
# risk mask 默认使用 A_oar / abs(B_boundary) / abs(P_fused) 构造 soft mask。
#
# 可选: 从 v1.4 checkpoint 续训 (zero-conv 保持 0 初始化, 初始 forward 等价 v1.4):
#   bash scripts/train_v1_5_local_mednext.sh \
#       --resume_from trained_models/v1_4_penalty_adapter/.../model_best_mae.pth \
#       --resume_ema  trained_models/v1_4_penalty_adapter/.../ema_best_mae.pth
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
LOG="scripts/train_v1_5_local_mednext_${TS}.log"

echo "[v1.5 local] GPU=${CUDA_VISIBLE_DEVICES}, data=${DATA_TRAIN}"
echo "log=${LOG}"

python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port 29506 \
    scripts/dose_train_3d_v1_5.py \
    --gpu 0 \
    --bs 1 \
    --epoch 600 \
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
    --penalty_files S_target.nii.gz A_oar.nii.gz B_boundary.nii.gz dose_synthetic.nii.gz \
    --penalty_channel_names S_target A_oar B_boundary P_fused \
    --penalty_dropout_p 0.2 \
    --adapter_warmup_epochs 0 \
    --risk_base 0.2 \
    --risk_use_channels A_oar B_boundary P_fused \
    --data_root_train "${DATA_TRAIN}" \
    --data_root_val "${DATA_VAL}" \
    --save_name_suffix "local_${TS}_gpu0_p128_riskaware" \
    "$@" \
    2>&1 | tee "${LOG}"
