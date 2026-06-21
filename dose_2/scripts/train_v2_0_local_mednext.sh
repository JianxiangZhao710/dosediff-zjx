#!/bin/bash
# v2.0 Dynamic Relation-to-Field Constraint Router 本地训练脚本
# (mednext 环境, AutoDL 数据路径)
#
# 数据流:
#   penalty_basis [S_target, A_oar, B_boundary, P_fused]
#     -> ConstraintFieldCompiler3D (+ RelationEncoder/FiLM) -> compiled field
#     -> PenaltyAdapter3D -> ZeroConv(0-init)
#     -> DynamicConstraintRouter3D (逐层/逐体素/随 timestep) -> X-stream 注入
#
# penalty 通道顺序 [S_target, A_oar, B_boundary, P_fused]，对应文件:
#   S_target.nii.gz / A_oar.nii.gz / B_boundary.nii.gz / dose_synthetic.nii.gz
#   (dose_synthetic 作为 P_fused 的 coarse / synthetic prior, 非 GT dose)
#
# ZeroConv 保持 0 初始化, Router 初始 delta=0 -> 初始行为等价 v1.5/v1.2。
# 可从 v1.5 checkpoint 续训 (strict=False, 主干/adapter/zero-conv 复用)。
# v1.5 权重在 MedSegDiff-master 下，不在 dose_2/trained_models/：
#   PRETRAINED_ROOT="../MedSegDiff-master/trained_models"
#   bash scripts/train_v2_0_local_mednext.sh \
#       --resume_from "${PRETRAINED_ROOT}/v1_5_risk_penalty_adapter/<run>/model_best_mae.pth" \
#       --resume_ema  "${PRETRAINED_ROOT}/v1_5_risk_penalty_adapter/<run>/ema_best_mae.pth"
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
LOG="scripts/train_v2_0_local_mednext_${TS}.log"

echo "[v2.0 local] GPU=${CUDA_VISIBLE_DEVICES}, data=${DATA_TRAIN}"
echo "log=${LOG}"

python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port 29507 \
    scripts/dose_train_3d_v2_0.py \
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
    --mode full_v2_0 \
    --compiler_out_channels 4 \
    --compiler_num_res_blocks 2 \
    --relation_dim 128 \
    --router_delta_scale 0.5 \
    --lambda_grad_loss 0.0 \
    --lambda_hf_loss 0.0 \
    --router_reg_weight 0.0 \
    --router_smooth_weight 0.0 \
    --data_root_train "${DATA_TRAIN}" \
    --data_root_val "${DATA_VAL}" \
    --save_name_suffix "local_${TS}_gpu0_p128_dynrouter" \
    "$@" \
    2>&1 | tee "${LOG}"
