import os
import shutil

source_root = "/data0/zhaojianxiang/preprocessed_data"
target_root = "/data0/zhaojianxiang/MedSegDiff_Data_3D"

phases = {
    'train': 'train-pats_preprocess',
    'validation': 'validation-pats_preprocess',
    'test': 'test-pats_preprocess'
}

def copy_masks():
    print("Starting Mask Copy...")
    for phase_name, source_folder_name in phases.items():
        source_phase_path = os.path.join(source_root, source_folder_name)
        target_phase_path = os.path.join(target_root, phase_name)
        
        if not os.path.exists(source_phase_path):
            continue
            
        patients = os.listdir(source_phase_path)
        print(f"Checking {phase_name} ({len(patients)} patients)...")
        
        count = 0
        for patient_id in patients:
            source_patient_path = os.path.join(source_phase_path, patient_id)
            target_patient_path = os.path.join(target_phase_path, patient_id)
            
            if not os.path.isdir(source_patient_path):
                continue
                
            # 查找所有 Mask_ 开头的文件
            files = os.listdir(source_patient_path)
            mask_files = [f for f in files if f.startswith("Mask_") and f.endswith(".nii.gz")]
            
            for m_file in mask_files:
                src = os.path.join(source_patient_path, m_file)
                dst = os.path.join(target_patient_path, m_file)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    count += 1
        
        print(f"  Copied {count} missing mask files for {phase_name}.")

if __name__ == "__main__":
    copy_masks()
