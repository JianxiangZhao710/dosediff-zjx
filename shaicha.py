import SimpleITK as sitk
import numpy as np
from pathlib import Path
import sys

def check_alignment_anomaly(patient_dir: Path) -> str:
    """
    检查单个患者的 CT 和重采样后的 Dose 是否发生空间偏移
    """
    ct_path = patient_dir / "ct.nii.gz"
    dose_path = patient_dir / "dose_resampled.nii.gz"
    
    if not ct_path.exists() or not dose_path.exists():
        return "跳过：缺少 ct.nii.gz 或 dose_resampled.nii.gz"

    try:
        # 读取图像为 Numpy 数组
        ct = sitk.ReadImage(str(ct_path))
        dose = sitk.ReadImage(str(dose_path))
        
        ct_arr = sitk.GetArrayFromImage(ct)
        dose_arr = sitk.GetArrayFromImage(dose)
    except Exception as e:
        return f"读取报错: {e}"

    # ==========================================
    # 异常检查 1：剂量全空 (完全脱靶)
    # ==========================================
    max_dose = np.max(dose_arr)
    if max_dose < 1e-3: # 最大剂量接近0
        return "❌ 严重错误：重采样后剂量全空（物理空间完全无交集，极度偏移）"

    # ==========================================
    # 异常检查 2：空气剂量检测 (局部偏移/错位)
    # ==========================================
    # 选取最大剂量的 30% 作为“高剂量区”阈值（排除低剂量散射）
    high_dose_threshold = max_dose * 0.3
    high_dose_mask = dose_arr > high_dose_threshold
    
    if np.sum(high_dose_mask) == 0:
        return "❌ 异常：未找到有效的高剂量区"

    # 提取高剂量区对应的 CT 体素值
    ct_hu_in_high_dose = ct_arr[high_dose_mask]
    
    # 统计有多少高剂量体素落在了空气中 (HU < -700 通常认为是空气/体外)
    air_voxels = np.sum(ct_hu_in_high_dose < -700)
    total_high_dose_voxels = len(ct_hu_in_high_dose)
    
    air_ratio = air_voxels / total_high_dose_voxels

    # 如果超过 15% 的高剂量打在空气里，大概率是偏移了 (可根据你的数据集微调该阈值)
    if air_ratio > 0.15:
        return f"⚠️ 偏移警告：有 {air_ratio:.1%} 的高剂量分布在体外空气中 (HU < -700)"

    return "✅ 正常"

def run_dataset_inspection(output_root: str):
    """
    遍历整个输出目录，输出质检报告
    """
    out_path = Path(output_root)
    if not out_path.exists():
        print("输出目录不存在！")
        return
        
    patients = sorted([p for p in out_path.iterdir() if p.is_dir()])
    
    print(f"开始质检，共发现 {len(patients)} 个样本...\n")
    print("-" * 60)
    
    error_count = 0
    for patient_dir in patients:
        status = check_alignment_anomaly(patient_dir)
        
        # 只打印有问题的样本，保持控制台整洁
        if "❌" in status or "⚠️" in status:
            print(f"患者: {patient_dir.name}")
            print(f"状态: {status}")
            print("-" * 60)
            error_count += 1
            
    print(f"\n质检完成！共发现 {error_count} 个异常样本。")

if __name__ == "__main__":
    # 指向你刚才跑出 .nii.gz 结果的输出总目录
    # 例如：/data1/home/zhaojianxiang/Dosedata/dose_mask
    if len(sys.argv) >= 2:
        output_dir = sys.argv[1]
    else:
        output_dir = "/data1/home/zhaojianxiang/Dosedata/dose_mask" 
        
    run_dataset_inspection(output_dir)