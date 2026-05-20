import os
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

def check_shapes(data_dir):
    data_dir = Path(data_dir)
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    mismatches = []
    
    for subdir in tqdm(subdirs, desc="Checking directories"):
        for nifti_file in subdir.glob("*.nii.gz"):
            try:
                img = nib.load(nifti_file)
                shape = img.shape
                # Usually shape is (512, 512, D) or something. Let's see if all are 512x512x512
                if shape != (512, 512, 512) and shape != (512, 512, 512, 1):
                    mismatches.append(f"{nifti_file}: {shape}")
            except Exception as e:
                mismatches.append(f"{nifti_file}: Error loading - {e}")
                
    if not mismatches:
        print("所有文件的维度都是 512x512x512。")
    else:
        print(f"发现 {len(mismatches)} 个维度不匹配的文件，前20个如下：")
        for m in mismatches[:20]:
            print(m)

if __name__ == "__main__":
    check_shapes("/data0/zhaojianxiang/dosedata/test_sample/")
