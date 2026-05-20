import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

# 配置路径
source_root = '/data0/zhaojianxiang/preprocessed_data'
target_root = '/data0/zhaojianxiang/preprocessed_data/3D_NPZ'

# 定义通道列表 (CT + 11 PSDM)
# 注意：CT 单独处理，这里列出 11 个 PSDM 的关键词
psdm_list = [
    'PTV70', 'PTV63', 'PTV56', 
    'Brainstem', 'SpinalCord', 
    'RightParotid', 'LeftParotid', 
    'Esophagus', 'Larynx', 
    'Mandible', 'possible_dose_mask'
]

phases = ['train', 'validation', 'test']

def normalize_ct(img):
    img = np.clip(img, 0, 2500)
    img = img / 1250.0 - 1.0
    return img

def normalize_dose(img):
    img = np.clip(img, 0, 80)
    img = img / 40.0 - 1.0
    return img

def process_patient(patient_dir, save_path):
    # 1. 读取 CT
    ct_path = os.path.join(patient_dir, 'ct.nii.gz')
    if not os.path.exists(ct_path):
        print(f"Missing CT for {patient_dir}")
        return
    
    ct_nii = nib.load(ct_path)
    ct_data = ct_nii.get_fdata().astype(np.float32)
    ct_data = normalize_ct(ct_data) # (128, 128, 128)
    
    # 2. 读取 Dose
    dose_path = os.path.join(patient_dir, 'dose.nii.gz')
    if not os.path.exists(dose_path):
        print(f"Missing Dose for {patient_dir}")
        return
    dose_data = nib.load(dose_path).get_fdata().astype(np.float32)
    dose_data = normalize_dose(dose_data) # (128, 128, 128)
    
    # 3. 读取 11 个 PSDM
    psdm_stack = []
    for name in psdm_list:
        psdm_path = os.path.join(patient_dir, f'PSDM_{name}.nii.gz')
        if os.path.exists(psdm_path):
            data = nib.load(psdm_path).get_fdata().astype(np.float32)
            # PSDM 已经在 PSDM_OpenKBP.py 中除以 100 了，数值范围通常在 -1 到 1 之间，这里保持原样即可
        else:
            # 如果缺失（虽然之前的脚本应该生成了全0文件），给个全0
            print(f"Warning: Missing {name} in {patient_dir}, using zeros.")
            data = np.zeros_like(ct_data)
        psdm_stack.append(data)
    
    # stack 形状: (11, 128, 128, 128) -> 转置为 (11, D, H, W) 
    # 注意 nibabel 读进来通常是 (H, W, D) 或者 (x, y, z)。
    # 之前的 csv2nii.py 做了 transpose: mask = np.transpose(mask, (2, 0, 1))[::-1, :, :]
    # 所以现在的 nii.gz 应该是 (Z, Y, X) 或者类似。
    # 我们统一把所有数据读进来，不做额外的 transpose，只要 CT, Dose, PSDM 空间一致即可。
    # 为了方便 PyTorch (C, D, H, W)，我们将数据堆叠在第一个维度。
    
    # CT: (D, H, W) -> (1, D, H, W)
    input_data = [ct_data[np.newaxis, ...]]
    
    # PSDM: list of (D, H, W) -> (11, D, H, W)
    for p in psdm_stack:
        input_data.append(p[np.newaxis, ...])
        
    # 合并输入: (12, D, H, W)
    input_tensor = np.concatenate(input_data, axis=0)
    
    # Dose: (1, D, H, W)
    dose_tensor = dose_data[np.newaxis, ...]
    
    # 保存压缩的 npz
    np.savez_compressed(save_path, input=input_tensor, gt=dose_tensor)

def main():
    for phase in phases:
        input_phase_dir = os.path.join(source_root, f'{phase}-pats_preprocess')
        output_phase_dir = os.path.join(target_root, phase)
        os.makedirs(output_phase_dir, exist_ok=True)
        
        patients = [p for p in os.listdir(input_phase_dir) if os.path.isdir(os.path.join(input_phase_dir, p))]
        print(f"Processing {phase}: {len(patients)} patients")
        
        for pat in tqdm(patients):
            pat_dir = os.path.join(input_phase_dir, pat)
            save_path = os.path.join(output_phase_dir, f'{pat}.npz')
            process_patient(pat_dir, save_path)

if __name__ == '__main__':
    main()
