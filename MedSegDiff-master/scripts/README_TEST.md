# 模型测试说明

## 快速开始

### 方法1: 使用快速脚本（推荐）

```bash
cd /home/zhaojianxiang/dosediff-zjx/MedSegDiff-master/scripts
./run_test.sh
```

### 方法2: 使用Python脚本

```bash
cd /home/zhaojianxiang/dosediff-zjx/MedSegDiff-master/scripts
source /home/zhaojianxiang/dosediff-zjx/dose/bin/activate

python test_model.py \
    --model_path trained_models/MedSegDiff_Flow_3D_bs1_epoch600/model_epoch200.pth \
    --test_data_dir /data0/zhaojianxiang/MedSegDiff_Data_3D/test \
    --gt_dir /data0/zhaojianxiang/preprocessed_data/test-pats \
    --output_dir test_results/model_epoch200 \
    --patch_size 32 128 128 \
    --batch_size 1 \
    --steps 50 \
    --gpu 0
```

## 参数说明

- `--model_path`: 模型权重路径（默认: `model_epoch200.pth`）
- `--test_data_dir`: 测试数据目录（用于预测输入）
- `--gt_dir`: Ground Truth数据目录（用于评估）
- `--output_dir`: 预测结果保存目录
- `--patch_size`: Patch大小 (D, H, W)
- `--batch_size`: Batch大小
- `--steps`: Flow Matching采样步数
- `--gpu`: GPU编号

## 可选参数

- `--skip_predict`: 跳过预测步骤，直接进行评估（如果预测结果已存在）
- `--skip_evaluate`: 跳过评估步骤，只进行预测

## 单独运行预测

```bash
python dose_predict_3d.py \
    --model_path trained_models/MedSegDiff_Flow_3D_bs1_epoch600/model_epoch200.pth \
    --data_dir /data0/zhaojianxiang/MedSegDiff_Data_3D/test \
    --output_dir test_results/model_epoch200 \
    --patch_size 32 128 128 \
    --batch_size 1 \
    --steps 50 \
    --gpu 0
```

## 单独运行评估

```bash
python ../evaluate_openKBP.py \
    --prediction_dir test_results/model_epoch200 \
    --gt_dir /data0/zhaojianxiang/preprocessed_data/test-pats \
    --denormalize 0
```

注意：预测脚本已经反归一化，所以评估时使用 `--denormalize 0`

## 数据检查

在运行测试前，可以检查数据完整性：

```bash
python check_data.py --check_all
python check_data.py --test_data_dir /data0/zhaojianxiang/MedSegDiff_Data_3D/test
python check_data.py --gt_dir /data0/zhaojianxiang/preprocessed_data/test-pats
```

## 输出结果

- 预测结果保存在: `test_results/model_epoch200/`
- 每个患者一个子目录，包含 `dose.nii.gz`
- 评估结果会打印到终端，并保存到 `test_results/model_epoch200/evaluation_results.txt`

## 评估指标

评估脚本会计算：
1. **3D剂量分布的平均绝对误差 (MAE)**: 平均值和标准差（单位：Gy）
2. **各结构DVH指标的绝对误差**: 
   - 靶区（PTV）: D1, D95, D99
   - 危及器官（OAR）: D_0.1_cc, mean

## 注意事项

1. 确保虚拟环境已激活
2. 确保有足够的GPU内存
3. 预测过程可能需要较长时间（取决于数据量和GPU性能）
4. 预测结果已经反归一化到Gy单位（0-80 Gy）
