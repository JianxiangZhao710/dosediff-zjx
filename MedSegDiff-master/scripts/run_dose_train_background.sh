#!/bin/bash
# 后台运行 dose 3D 训练脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs"
LOG_FILE="${LOG_DIR}/dose_train_3d_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${LOG_DIR}/dose_train_3d.pid"

mkdir -p "$LOG_DIR"

source /home/zhaojianxiang/dosediff-zjx/dose/bin/activate
cd "$SCRIPT_DIR"

echo "启动训练，日志: $LOG_FILE"
echo "查看实时日志: tail -f $LOG_FILE"
echo "停止训练: kill \$(cat $PID_FILE)"

nohup torchrun --nproc_per_node=1 --master_port=29505 dose_train_3d.py --bs 1 --epoch 600 \
    >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "PID: $(cat $PID_FILE)"
