#!/usr/bin/env python
"""
完整的测试脚本：预测 + 评估
使用测试集对训练好的模型进行预测和评估
"""
import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='测试模型：预测 + 评估')
    parser.add_argument('--model_path', type=str,
                        default='trained_models/MedSegDiff_Flow_3D_OpenKBP_11ch_bs1_epoch600/model_epoch200.pth',
                        help='模型权重路径')
    parser.add_argument('--test_data_dir', type=str,
                        default='/data0/zhaojianxiang/preprocessed_data/test-pats_preprocess',
                        help='测试数据目录（包含 ct.nii.gz 与 Mask_*.nii.gz，用于预测输入）')
    parser.add_argument('--gt_dir', type=str,
                        default='/data0/zhaojianxiang/preprocessed_data/test-pats_preprocess',
                        help='真实数据目录（包含 dose.nii.gz 与 Mask_*.nii.gz，用于评估）')
    parser.add_argument('--output_dir', type=str,
                        default='test_results/model_epoch200',
                        help='预测结果保存目录')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 128, 128],
                        help='Patch大小 (D, H, W)，默认: 64 128 128')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch大小，默认: 1')
    parser.add_argument('--steps', type=int, default=20, help='Flow Matching采样步数，默认: 20（推荐10-30步）')
    parser.add_argument('--gpu', type=int, default=1, help='使用的GPU编号，默认: 1')
    parser.add_argument('--skip_predict', action='store_true',
                        help='跳过预测步骤，直接进行评估（如果预测结果已存在）')
    parser.add_argument('--skip_evaluate', action='store_true',
                        help='跳过评估步骤，只进行预测')
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("=" * 80)
    print("模型测试脚本")
    print("=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"测试数据目录: {args.test_data_dir}")
    print(f"Ground Truth目录: {args.gt_dir}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 80)
    
    # 步骤1: 预测
    if not args.skip_predict:
        print("\n" + "=" * 80)
        print("步骤 1/2: 进行预测...")
        print("=" * 80)
        
        predict_cmd = [
            'python', 'dose_predict_3d.py',
            '--model_path', args.model_path,
            '--data_dir', args.test_data_dir,
            '--output_dir', args.output_dir,
            '--patch_size'] + [str(x) for x in args.patch_size] + [
            '--batch_size', str(args.batch_size),
            '--steps', str(args.steps),
            '--gpu', str(args.gpu)
        ]
        
        print(f"执行命令: {' '.join(predict_cmd)}")
        result = subprocess.run(predict_cmd, cwd=script_dir)
        
        if result.returncode != 0:
            print(f"\n❌ 预测失败，退出码: {result.returncode}")
            return 1
        
        print("\n✓ 预测完成")
    else:
        print("\n跳过预测步骤（使用已有预测结果）")
    
    # 步骤2: 评估
    if not args.skip_evaluate:
        print("\n" + "=" * 80)
        print("步骤 2/2: 进行评估...")
        print("=" * 80)
        
        # 检查预测结果是否存在
        if not os.path.exists(args.output_dir):
            print(f"❌ 预测结果目录不存在: {args.output_dir}")
            print("请先运行预测步骤（不要使用 --skip_predict）")
            return 1
        
        # 检查ground truth目录是否存在
        if not os.path.exists(args.gt_dir):
            print(f"❌ Ground Truth目录不存在: {args.gt_dir}")
            return 1
        
        evaluate_cmd = [
            'python', '../evaluate_openKBP.py',
            '--prediction_dir', os.path.abspath(args.output_dir),
            '--gt_dir', args.gt_dir,
            '--denormalize', '0'  # 预测脚本已经反归一化，所以设为0
        ]
        
        print(f"执行命令: {' '.join(evaluate_cmd)}")
        result = subprocess.run(evaluate_cmd, cwd=script_dir)
        
        if result.returncode != 0:
            print(f"\n❌ 评估失败，退出码: {result.returncode}")
            return 1
        
        print("\n✓ 评估完成")
        
        # 保存评估结果到文件
        result_file = os.path.join(args.output_dir, 'evaluation_results.txt')
        print(f"\n评估结果已保存到: {result_file}")
    else:
        print("\n跳过评估步骤")
    
    print("\n" + "=" * 80)
    print("✓ 测试完成！")
    print("=" * 80)
    return 0

if __name__ == '__main__':
    sys.exit(main())
