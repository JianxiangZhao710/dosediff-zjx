#!/bin/bash

# 1. 激活虚拟环境
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate dosedata

# 2. 设置目录
cd /home/zhaojianxiang/dosediff-zjx/MedSegDiff-master/scripts

# 3. 设置端口 (随机一个端口以防冲突)
PORT=29506

# 4. 训练超参数
WARMUP_RATIO=0.05
MIN_LR=1e-6

# 5. 定义日志文件
LOG_FILE="train_3d_5ch_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training in background..."
echo "Logs will be written to: $LOG_FILE"
echo "Process ID will be saved to: run.pid"
echo "LR schedule: warmup_ratio=$WARMUP_RATIO, min_lr=$MIN_LR"

# 6. 后台运行
CUDA_VISIBLE_DEVICES=0,2,3 nohup torchrun \
    --nproc_per_node=3 \
    --master_port=$PORT \
    dose_train_3d.py \
    --bs 1 \
    --epoch 600 \
    --patch_size 64 128 128 \
    --warmup_ratio "$WARMUP_RATIO" \
    --min_lr "$MIN_LR" \
    > "$LOG_FILE" 2>&1 &

# 7. 保存 PID
PID=$!
echo $PID > run.pid

echo "Training started with PID: $PID"
echo "To view logs, run: tail -f $LOG_FILE"
