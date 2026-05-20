#!/bin/bash
# 在 GPU 1 上跑随机 10 例预测，不占用训练常用的 0/2/3
set -e
cd "$(dirname "$0")"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate dosedata

OUT_LOG="/data0/zhaojianxiang/dosedata/dose_test/predict_gpu1.log"
mkdir -p /data0/zhaojianxiang/dosedata/dose_test

nohup python predict_random_samples.py --gpu 1 >> "$OUT_LOG" 2>&1 &
echo $! > /data0/zhaojianxiang/dosedata/dose_test/predict_gpu1.pid
echo "预测已在后台启动，PID=$!"
echo "日志: $OUT_LOG"
echo "查看: tail -f $OUT_LOG"
