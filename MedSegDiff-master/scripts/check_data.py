#!/usr/bin/env python
"""
检查预测和评估数据的完整性和格式
"""
import os
import sys
import argparse
import nibabel as nib
import numpy as np

def check_prediction_data(prediction_dir):
    """检查预测数据目录"""
    print("=" * 80)
    print("检查预测数据目录:", prediction_dir)
    print("=" * 80)
    
    if not os.path.exists(prediction_dir):
        print(f"❌ 预测数据目录不存在: {prediction_dir}")
        return False
    
    patient_list = sorted([p for p in os.listdir(prediction_dir) 
                          if os.path.isdir(os.path.join(prediction_dir, p))])
    
    if len(patient_list) == 0:
        print(f"❌ 预测数据目录中没有找到患者数据")
        return False
    
    print(f"✓ 找到 {len(patient_list)} 个患者")
    
    missing_files = []
    invalid_files = []
    
    for patient_id in patient_list:
        patient_dir = os.path.join(prediction_dir, patient_id)
        dose_file = os.path.join(patient_dir, 'dose.nii.gz')
        
        if not os.path.exists(dose_file):
            missing_files.append(f"{patient_id}/dose.nii.gz")
        else:
            try:
                nii = nib.load(dose_file)
                data = nii.get_fdata()
                print(f"  {patient_id}: dose.nii.gz - shape: {data.shape}, "
                      f"min: {data.min():.2f}, max: {data.max():.2f}, mean: {data.mean():.2f}")
            except Exception as e:
                invalid_files.append(f"{patient_id}/dose.nii.gz: {str(e)}")
    
    if missing_files:
        print(f"\n❌ 缺失文件 ({len(missing_files)}):")
        for f in missing_files[:10]:  # 只显示前10个
            print(f"  - {f}")
        if len(missing_files) > 10:
            print(f"  ... 还有 {len(missing_files) - 10} 个文件缺失")
    
    if invalid_files:
        print(f"\n❌ 无效文件 ({len(invalid_files)}):")
        for f in invalid_files:
            print(f"  - {f}")
    
    if not missing_files and not invalid_files:
        print("\n✓ 所有预测数据文件完整且有效")
        return True
    else:
        return False

def check_ground_truth_data(gt_dir):
    """检查真实数据目录"""
    print("\n" + "=" * 80)
    print("检查真实数据目录:", gt_dir)
    print("=" * 80)
    
    if not os.path.exists(gt_dir):
        print(f"❌ 真实数据目录不存在: {gt_dir}")
        return False
    
    patient_list = sorted([p for p in os.listdir(gt_dir) 
                          if os.path.isdir(os.path.join(gt_dir, p))])
    
    if len(patient_list) == 0:
        print(f"❌ 真实数据目录中没有找到患者数据")
        return False
    
    print(f"✓ 找到 {len(patient_list)} 个患者")
    
    # 必需的文件
    required_files = [
        'dose.nii.gz',
        'possible_dose_mask.nii.gz'
    ]
    
    # 可选的结构文件
    optional_structures = [
        'Brainstem.nii.gz',
        'SpinalCord.nii.gz',
        'RightParotid.nii.gz',
        'LeftParotid.nii.gz',
        'Esophagus.nii.gz',
        'Larynx.nii.gz',
        'Mandible.nii.gz',
        'PTV70.nii.gz',
        'PTV63.nii.gz',
        'PTV56.nii.gz'
    ]
    
    missing_required = []
    missing_optional = {s: [] for s in optional_structures}
    invalid_files = []
    
    for patient_id in patient_list:
        patient_dir = os.path.join(gt_dir, patient_id)
        
        # 检查必需文件
        for req_file in required_files:
            file_path = os.path.join(patient_dir, req_file)
            if not os.path.exists(file_path):
                missing_required.append(f"{patient_id}/{req_file}")
            else:
                try:
                    nii = nib.load(file_path)
                    data = nii.get_fdata()
                    print(f"  {patient_id}: {req_file} - shape: {data.shape}")
                except Exception as e:
                    invalid_files.append(f"{patient_id}/{req_file}: {str(e)}")
        
        # 检查可选结构文件
        for struct_file in optional_structures:
            file_path = os.path.join(patient_dir, struct_file)
            if not os.path.exists(file_path):
                missing_optional[struct_file].append(patient_id)
    
    # 报告结果
    if missing_required:
        print(f"\n❌ 缺失必需文件 ({len(missing_required)}):")
        for f in missing_required[:10]:
            print(f"  - {f}")
        if len(missing_required) > 10:
            print(f"  ... 还有 {len(missing_required) - 10} 个文件缺失")
    else:
        print("\n✓ 所有必需文件存在")
    
    # 统计可选文件
    struct_stats = {}
    for struct_file, missing_list in missing_optional.items():
        available = len(patient_list) - len(missing_list)
        struct_stats[struct_file] = (available, len(missing_list))
    
    print("\n可选结构文件统计:")
    for struct_file, (available, missing) in sorted(struct_stats.items()):
        status = "✓" if missing == 0 else "⚠"
        print(f"  {status} {struct_file}: {available}/{len(patient_list)} 患者有该结构")
    
    if invalid_files:
        print(f"\n❌ 无效文件 ({len(invalid_files)}):")
        for f in invalid_files:
            print(f"  - {f}")
    
    return len(missing_required) == 0 and len(invalid_files) == 0

def check_test_data_for_prediction(data_dir):
    """检查用于预测的测试数据"""
    print("\n" + "=" * 80)
    print("检查预测输入数据目录:", data_dir)
    print("=" * 80)
    
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return False
    
    patient_list = sorted([p for p in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, p))])
    
    if len(patient_list) == 0:
        print(f"❌ 数据目录中没有找到患者数据")
        return False
    
    print(f"✓ 找到 {len(patient_list)} 个患者")
    
    # 必需的文件
    required_files = ['ct.nii.gz']
    
    # PSDM文件
    psdm_files = [
        'PSDM_PTV70.nii.gz',
        'PSDM_PTV63.nii.gz',
        'PSDM_PTV56.nii.gz',
        'PSDM_Brainstem.nii.gz',
        'PSDM_SpinalCord.nii.gz',
        'PSDM_RightParotid.nii.gz',
        'PSDM_LeftParotid.nii.gz',
        'PSDM_Esophagus.nii.gz',
        'PSDM_Larynx.nii.gz',
        'PSDM_Mandible.nii.gz',
        'PSDM_possible_dose_mask.nii.gz'
    ]
    
    missing_required = []
    missing_psdm = {f: [] for f in psdm_files}
    invalid_files = []
    
    for patient_id in patient_list[:5]:  # 只检查前5个患者作为示例
        patient_dir = os.path.join(data_dir, patient_id)
        
        # 检查必需文件
        for req_file in required_files:
            file_path = os.path.join(patient_dir, req_file)
            if not os.path.exists(file_path):
                missing_required.append(f"{patient_id}/{req_file}")
            else:
                try:
                    nii = nib.load(file_path)
                    data = nii.get_fdata()
                    print(f"  {patient_id}: {req_file} - shape: {data.shape}")
                except Exception as e:
                    invalid_files.append(f"{patient_id}/{req_file}: {str(e)}")
        
        # 检查PSDM文件
        for psdm_file in psdm_files:
            file_path = os.path.join(patient_dir, psdm_file)
            if not os.path.exists(file_path):
                missing_psdm[psdm_file].append(patient_id)
    
    if missing_required:
        print(f"\n❌ 缺失必需文件:")
        for f in missing_required:
            print(f"  - {f}")
    else:
        print("\n✓ CT文件存在")
    
    print("\nPSDM文件统计 (前5个患者):")
    for psdm_file, missing_list in missing_psdm.items():
        available = 5 - len(missing_list)
        status = "✓" if len(missing_list) == 0 else "⚠"
        print(f"  {status} {psdm_file}: {available}/5 患者有该文件")
    
    return len(missing_required) == 0

def main():
    parser = argparse.ArgumentParser(description='检查预测和评估数据的完整性')
    parser.add_argument('--prediction_dir', type=str, default=None,
                        help='预测结果目录（用于评估）')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='真实数据目录（用于评估）')
    parser.add_argument('--test_data_dir', type=str, default=None,
                        help='测试数据目录（用于预测）')
    parser.add_argument('--check_all', action='store_true',
                        help='检查所有常见的数据目录')
    
    args = parser.parse_args()
    
    all_ok = True
    
    if args.check_all:
        # 检查常见的数据目录
        common_dirs = {
            'test_data': '/data0/zhaojianxiang/MedSegDiff_Data_3D/test',
            'validation_data': '/data0/zhaojianxiang/MedSegDiff_Data_3D/validation',
            'train_data': '/data0/zhaojianxiang/MedSegDiff_Data_3D/train'
        }
        
        for name, path in common_dirs.items():
            if os.path.exists(path):
                print(f"\n检查 {name}: {path}")
                check_test_data_for_prediction(path)
    
    if args.prediction_dir:
        ok = check_prediction_data(args.prediction_dir)
        all_ok = all_ok and ok
    
    if args.gt_dir:
        ok = check_ground_truth_data(args.gt_dir)
        all_ok = all_ok and ok
    
    if args.test_data_dir:
        ok = check_test_data_for_prediction(args.test_data_dir)
        all_ok = all_ok and ok
    
    print("\n" + "=" * 80)
    if all_ok:
        print("✓ 所有检查通过！")
    else:
        print("⚠ 发现一些问题，请检查上述输出")
    print("=" * 80)

if __name__ == '__main__':
    main()
