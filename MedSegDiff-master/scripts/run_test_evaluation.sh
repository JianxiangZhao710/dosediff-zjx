#!/bin/bash
# 使用测试集进行预测和评估的完整流程脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 配置参数
MODEL_PATH="trained_models/MedSegDiff_Flow_3D_OpenKBP_11ch_bs1_epoch600/model_epoch200.pth"
TEST_DATA_DIR="/data0/zhaojianxiang/preprocessed_data/test-pats_preprocess"
GT_DIR="/data0/zhaojianxiang/preprocessed_data/test-pats_preprocess"
OUTPUT_DIR="results/openkbp_11ch_epoch200"
STEPS=50
BATCH_SIZE=1
GPU=0

echo "=================================================================================="
echo "测试集预测和评估流程"
echo "=================================================================================="
echo "模型路径: $MODEL_PATH"
echo "测试数据: $TEST_DATA_DIR"
echo "Ground Truth: $GT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "=================================================================================="

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型文件不存在: $MODEL_PATH"
    exit 1
fi

# 检查数据目录是否存在
if [ ! -d "$TEST_DATA_DIR" ]; then
    echo "❌ 错误: 测试数据目录不存在: $TEST_DATA_DIR"
    exit 1
fi

if [ ! -d "$GT_DIR" ]; then
    echo "❌ 错误: Ground Truth目录不存在: $GT_DIR"
    exit 1
fi

# 激活虚拟环境
source /home/zhaojianxiang/dosediff-zjx/dose/bin/activate

# 步骤1: 预测
echo ""
echo "=================================================================================="
echo "步骤 1: 开始预测..."
echo "=================================================================================="

python dose_predict_3d.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$TEST_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --patch_size 32 128 128 \
    --batch_size $BATCH_SIZE \
    --steps $STEPS \
    --gpu $GPU

if [ $? -ne 0 ]; then
    echo "❌ 预测失败！"
    exit 1
fi

echo ""
echo "✓ 预测完成！"

# 步骤2: 评估
echo ""
echo "=================================================================================="
echo "步骤 2: 开始评估..."
echo "=================================================================================="

cd ..
python evaluate_openKBP.py \
    --prediction_dir "$SCRIPT_DIR/$OUTPUT_DIR" \
    --gt_dir "$GT_DIR" \
    --denormalize 0

if [ $? -ne 0 ]; then
    echo "❌ 评估失败！"
    exit 1
fi

echo ""
echo "=================================================================================="
echo "✓ 所有步骤完成！"
echo "=================================================================================="
echo "预测结果保存在: $OUTPUT_DIR"
echo "评估结果已显示在上方"
