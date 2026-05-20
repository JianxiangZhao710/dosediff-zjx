import os
import shutil
import numpy as np

# 配置路径
source_root = "/data0/zhaojianxiang/preprocessed_data"
target_root = "/data0/zhaojianxiang/MedSegDiff_Data_3D"

# 定义需要移动的文件和重命名规则
# 键是源文件名（支持部分匹配），值是目标文件名
files_to_copy = {
    'ct.nii.gz': 'ct.nii.gz',
    'dose.nii.gz': 'dose.nii.gz',
    'PSDM_PTV70.nii.gz': 'PSDM_PTV70.nii.gz',
    'PSDM_PTV63.nii.gz': 'PSDM_PTV63.nii.gz',
    'PSDM_PTV56.nii.gz': 'PSDM_PTV56.nii.gz',
    'PSDM_Brainstem.nii.gz': 'PSDM_Brainstem.nii.gz',
    'PSDM_SpinalCord.nii.gz': 'PSDM_SpinalCord.nii.gz',
    'PSDM_RightParotid.nii.gz': 'PSDM_RightParotid.nii.gz',
    'PSDM_LeftParotid.nii.gz': 'PSDM_LeftParotid.nii.gz',
    'PSDM_Esophagus.nii.gz': 'PSDM_Esophagus.nii.gz',
    'PSDM_Larynx.nii.gz': 'PSDM_Larynx.nii.gz',
    'PSDM_Mandible.nii.gz': 'PSDM_Mandible.nii.gz',
    'PSDM_possible_dose_mask.nii.gz': 'PSDM_possible_dose_mask.nii.gz'
}

phases = {
    'train': 'train-pats_preprocess',
    'validation': 'validation-pats_preprocess',
    'test': 'test-pats_preprocess'
}

def organize_data():
    if not os.path.exists(target_root):
        os.makedirs(target_root)
        print(f"Created target root: {target_root}")

    for phase_name, source_folder_name in phases.items():
        source_phase_path = os.path.join(source_root, source_folder_name)
        target_phase_path = os.path.join(target_root, phase_name)
        
        if not os.path.exists(source_phase_path):
            print(f"Warning: Source path not found: {source_phase_path}")
            continue
            
        print(f"Processing {phase_name}...")
        
        # 遍历每个病人
        patients = os.listdir(source_phase_path)
        for i, patient_id in enumerate(patients):
            if i % 10 == 0:
                print(f"  Processing patient {i}/{len(patients)}: {patient_id}")
                
            source_patient_path = os.path.join(source_phase_path, patient_id)
            target_patient_path = os.path.join(target_phase_path, patient_id)
            
            if not os.path.isdir(source_patient_path):
                continue
                
            os.makedirs(target_patient_path, exist_ok=True)
            
            # 复制指定文件
            for src_name, dst_name in files_to_copy.items():
                src_file = os.path.join(source_patient_path, src_name)
                dst_file = os.path.join(target_patient_path, dst_name)
                
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dst_file)
                else:
                    print(f"    Warning: Missing {src_name} for {patient_id}")

    print("Data organization completed!")

if __name__ == "__main__":
    organize_data()
