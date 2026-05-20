import os
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

def find_max_z(data_dir):
    data_dir = Path(data_dir)
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    max_z = 0
    max_z_file = ""
    
    for subdir in tqdm(subdirs, desc="Checking directories"):
        # We can just check one file per subdir since they usually match within a sample
        # but let's check all just to be safe.
        for nifti_file in subdir.glob("*.nii.gz"):
            try:
                img = nib.load(nifti_file)
                shape = img.shape
                if len(shape) >= 3:
                    z = shape[2]
                    if z > max_z:
                        max_z = z
                        max_z_file = str(nifti_file)
            except Exception as e:
                pass
                
    print(f"\n最大 Z 轴切片数量为: {max_z}")
    print(f"对应的文件是: {max_z_file}")

if __name__ == "__main__":
    find_max_z("/data0/zhaojianxiang/dosedata/test_sample/")
