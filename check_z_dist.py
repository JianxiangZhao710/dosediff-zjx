import os
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np

base_dir = "/data0/zhaojianxiang/dosedata/test_sample/"
sample_names = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

z_dims = []
for sample in tqdm(sample_names, desc="读取图像信息"):
    ct_path = os.path.join(base_dir, sample, "ct.nii.gz")
    if not os.path.exists(ct_path):
        continue
    
    # 仅读取图像头信息（元数据），速度极快，不加载整张图
    reader = sitk.ImageFileReader()
    reader.SetFileName(ct_path)
    reader.ReadImageInformation()
    size = reader.GetSize() # SimpleITK 的 size 格式为 (X, Y, Z)
    z_dims.append(size[2])

if len(z_dims) == 0:
    print("未找到有效的 ct.nii.gz 文件")
else:
    z_dims = np.array(z_dims)
    print("\n========= Z轴维度分布统计 =========")
    print(f"成功读取样本数: {len(z_dims)}")
    print(f"Z轴最小值: {z_dims.min()}")
    print(f"Z轴最大值: {z_dims.max()}")
    print(f"Z轴平均值: {z_dims.mean():.2f}")
    print(f"Z轴中位数: {np.median(z_dims)}")
    print("-" * 35)
    print(f"Z <= 128 的样本数: {(z_dims <= 128).sum()}  (占比 {((z_dims <= 128).sum() / len(z_dims) * 100):.1f}%)")
    print(f"Z > 128  的样本数: {(z_dims > 128).sum()}  (占比 {((z_dims > 128).sum() / len(z_dims) * 100):.1f}%)")
    print("-" * 35)
    
    # 详细区间分布
    bins = [0, 64, 96, 128, 160, 192, 224, 256, 300, 400, 1000]
    hist, bin_edges = np.histogram(z_dims, bins=bins)
    print("详细区间分布:")
    for i in range(len(hist)):
        if hist[i] > 0:
            print(f"  {bin_edges[i]:>3} < Z <= {bin_edges[i+1]:>3}: {hist[i]} 个")
