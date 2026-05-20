import os
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def analyze_z_distribution(data_dir):
    data_dir = Path(data_dir)
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    z_sizes = []
    
    for subdir in tqdm(subdirs, desc="Analyzing Z dimensions"):
        # Just check one file per sample to get Z size, ct.nii.gz is best
        ct_file = subdir / "ct.nii.gz"
        if not ct_file.exists():
            # Fallback to any nii.gz if ct doesn't exist
            files = list(subdir.glob("*.nii.gz"))
            if not files:
                continue
            ct_file = files[0]
            
        try:
            img = nib.load(ct_file)
            shape = img.shape
            if len(shape) >= 3:
                z_sizes.append(shape[2])
        except Exception:
            pass
            
    # Calculate statistics
    z_sizes = np.array(z_sizes)
    
    print("\n" + "="*40)
    print("Z轴层数分布统计")
    print("="*40)
    print(f"总样本数: {len(z_sizes)}")
    print(f"最小 Z 值: {np.min(z_sizes)}")
    print(f"最大 Z 值: {np.max(z_sizes)}")
    print(f"平均 Z 值: {np.mean(z_sizes):.1f}")
    print(f"中位数 Z 值: {np.median(z_sizes):.1f}")
    
    print("\n区间统计:")
    ranges = [
        (0, 64, "Z <= 64"),
        (65, 96, "64 < Z <= 96"),
        (97, 128, "96 < Z <= 128"),
        (129, 160, "128 < Z <= 160"),
        (161, 200, "160 < Z <= 200"),
        (201, 300, "200 < Z <= 300"),
        (301, float('inf'), "Z > 300")
    ]
    
    for r_min, r_max, label in ranges:
        count = np.sum((z_sizes >= r_min) & (z_sizes <= r_max))
        percentage = count / len(z_sizes) * 100
        print(f"{label:<15}: {count:>4} 个样本 ({percentage:>5.1f}%)")
        
    print("\n关键阈值切分:")
    below_128 = np.sum(z_sizes <= 128)
    above_128 = np.sum(z_sizes > 128)
    print(f"Z <= 128: {below_128} 个样本 ({below_128/len(z_sizes)*100:.1f}%)")
    print(f"Z >  128: {above_128} 个样本 ({above_128/len(z_sizes)*100:.1f}%)")
    
if __name__ == "__main__":
    analyze_z_distribution("/data0/zhaojianxiang/dosedata/test_sample/")
