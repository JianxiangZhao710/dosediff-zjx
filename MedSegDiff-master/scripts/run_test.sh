#!/bin/bash
# 快速测试脚本：使用测试集预测和评估 model_epoch200.pth

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="trained_models/MedSegDiff_Flow_3D_OpenKBP_11ch_bs1_epoch600/model_epoch200.pth"
TEST_DATA_DIR="/data0/zhaojianxiang/preprocessed_data/test-pats_preprocess"
GT_DIR="/data0/zhaojianxiang/preprocessed_data/test-pats_preprocess"
OUTPUT_DIR="test_results/openkbp_11ch_epoch200"

echo "=========================================="
echo "测试模型: model_epoch200.pth"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "测试数据: $TEST_DATA_DIR"
echo "Ground Truth: $GT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

# 激活虚拟环境
source /home/zhaojianxiang/dosediff-zjx/dose/bin/activate

# 运行测试脚本
# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

python test_model.py \
    --model_path "$MODEL_PATH" \
    --test_data_dir "$TEST_DATA_DIR" \
    --gt_dir "$GT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --patch_size 32 128 128 \
    --batch_size 1 \
    --steps 20 \
    --gpu 1

echo ""
echo "测试完成！结果保存在: $OUTPUT_DIR"
