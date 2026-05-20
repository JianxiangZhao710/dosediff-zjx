#!/bin/bash
# 单卡训练脚本 (GPU 3, 即第 4 张显卡)
# 数据: OpenKBP 预处理集 (/data0/zhaojianxiang/preprocessed_data/train-pats_preprocess)
# 条件: CT + 11 个 Mask (PTV70/63/56 + 7 OARs + possible_dose_mask)
# 模型保存路径: trained_models/MedSegDiff_Flow_3D_OpenKBP_11ch_bs1_epoch${EPOCH}/

set -e

# 1. 激活虚拟环境
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate dosedata

# 2. 切到脚本目录
cd /home/zhaojianxiang/dosediff-zjx/MedSegDiff-master/scripts

# 3. DDP 端口（与其它训练任务错开即可）
PORT=29507

# 4. 训练超参
BS=1                  # per-GPU batch size
EPOCH=600
# 整卷训练: OpenKBP 数据已统一为 128^3, 设为整卷尺寸时加载器跳过随机裁剪
PATCH_D=128
PATCH_H=128
PATCH_W=128
GRAD_ACCUM=12         # 单卡 bs=1 时，accum=12 等效 batch=12
WARMUP_RATIO=0.05
MIN_LR=1e-6

# 5. 日志文件
LOG_FILE="train_openkbp_11ch_gpu3_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="run_gpu3.pid"

echo "=================================================================="
echo "Starting OpenKBP 11-channel training on GPU 3"
echo "=================================================================="
echo "Data        : /data0/zhaojianxiang/preprocessed_data/train-pats_preprocess"
echo "Patch size  : ${PATCH_D} ${PATCH_H} ${PATCH_W}"
echo "Per-GPU bs  : ${BS}    grad_accum_steps: ${GRAD_ACCUM}    (effective bs ${GRAD_ACCUM})"
echo "Epoch       : ${EPOCH}"
echo "LR schedule : warmup_ratio=${WARMUP_RATIO}, min_lr=${MIN_LR}"
echo "Log file    : ${LOG_FILE}"
echo "PID file    : ${PID_FILE}"
echo "=================================================================="

# 6. 后台启动: 只使用 GPU 3
CUDA_VISIBLE_DEVICES=3 nohup torchrun \
    --nproc_per_node=1 \
    --master_port=${PORT} \
    dose_train_3d.py \
    --bs ${BS} \
    --epoch ${EPOCH} \
    --patch_size ${PATCH_D} ${PATCH_H} ${PATCH_W} \
    --grad_accum_steps ${GRAD_ACCUM} \
    --warmup_ratio "${WARMUP_RATIO}" \
    --min_lr "${MIN_LR}" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo $PID > "${PID_FILE}"

echo "Training started with PID: ${PID}"
echo ""
echo "Useful commands:"
echo "  tail -f ${LOG_FILE}                 # 实时查看日志"
echo "  nvidia-smi -i 3                     # 查看 GPU 3 使用情况"
echo "  kill \$(cat ${PID_FILE})             # 停止训练"
