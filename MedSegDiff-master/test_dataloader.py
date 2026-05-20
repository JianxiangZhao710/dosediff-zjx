import os
import sys
import torch
import numpy as np
import nibabel as nib

def test_single_sample(patient_dir, patch_size=(64, 128, 128)):
    print(f"Testing data loading for patient: {patient_dir}")
    
    def load_nii(path):
        if os.path.exists(path):
            return nib.load(path).get_fdata().astype(np.float32)
        else:
            print(f"File not found: {path}")
            return None

    def normalize_ct(img):
        img = np.clip(img, 0, 2500)
        img = img / 1250.0 - 1.0
        return img

    def normalize_dose(img):
        img = np.clip(img, 0, 80)
        img = img / 40.0 - 1.0
        return img

    # 1. Load CT
    ct_path = os.path.join(patient_dir, 'ct.nii.gz')
    ct_data = load_nii(ct_path)
    if ct_data is None: return
    print(f"Original CT shape (H, W, D): {ct_data.shape}")
    ct_data = ct_data.transpose(2, 0, 1) # (D, H, W)
    
    # 2. Load Dose
    dose_path = os.path.join(patient_dir, 'dose_resampled.nii.gz')
    dose_data = load_nii(dose_path)
    if dose_data is None: return
    print(f"Original Dose shape (H, W, D): {dose_data.shape}")
    dose_data = dose_data.transpose(2, 0, 1)

    # 3. Load Radio Biology Map
    radio_path = os.path.join(patient_dir, 'radio_biology_map.nii.gz')
    radio_map = load_nii(radio_path)
    if radio_map is None: return
    print(f"Original Radio Map shape (H, W, D): {radio_map.shape}")
    radio_map = radio_map.transpose(2, 0, 1)
    radio_map = np.round(radio_map).astype(np.int64)
    
    # Print unique values in the radio map
    unique_vals = np.unique(radio_map)
    print(f"Unique values in Radio Map: {unique_vals}")
    
    # 4. Construct Condition Channels (5 Channels One-Hot)
    num_classes = 5
    # Handle unexpected values by clipping to [0, 4] just in case
    radio_map_clipped = np.clip(radio_map, 0, num_classes - 1)
    dis_stack = np.eye(num_classes)[radio_map_clipped].transpose(3, 0, 1, 2).astype(np.float32)

    # Normalize CT & Dose
    ct_data = normalize_ct(ct_data)
    dose_data = normalize_dose(dose_data)
    
    print(f"Transposed & Normalized CT shape (D, H, W): {ct_data.shape}, min: {ct_data.min():.2f}, max: {ct_data.max():.2f}")
    print(f"Transposed & Normalized Dose shape (D, H, W): {dose_data.shape}, min: {dose_data.min():.2f}, max: {dose_data.max():.2f}")
    print(f"One-Hot Radio Map shape (5, D, H, W): {dis_stack.shape}")

    # 5. Random Crop
    current_depth = ct_data.shape[0]
    target_depth = patch_size[0]
    
    if current_depth > target_depth:
        start_z = np.random.randint(0, current_depth - target_depth)
        end_z = start_z + target_depth
        
        ct_crop = ct_data[start_z:end_z, :, :]
        dose_crop = dose_data[start_z:end_z, :, :]
        dis_crop = dis_stack[:, start_z:end_z, :, :]
        print(f"Cropped depth from {current_depth} to {target_depth} (z: {start_z} to {end_z})")
    else:
        pad_z = target_depth - current_depth
        ct_crop = np.pad(ct_data, ((0, pad_z), (0,0), (0,0)), 'constant')
        dose_crop = np.pad(dose_data, ((0, pad_z), (0,0), (0,0)), 'constant')
        dis_crop = np.pad(dis_stack, ((0,0), (0, pad_z), (0,0), (0,0)), 'constant')
        print(f"Padded depth from {current_depth} to {target_depth} (pad_z: {pad_z})")

    # Add Channel Dim to CT and Dose -> (1, D, H, W)
    ct_crop = ct_crop[np.newaxis, ...]
    dose_crop = dose_crop[np.newaxis, ...]
    
    tensor_ct = torch.from_numpy(ct_crop)
    tensor_dis = torch.from_numpy(dis_crop)
    tensor_dose = torch.from_numpy(dose_crop)

    print("\n--- Final Output Tensors ---")
    print(f"CT Tensor shape: {tensor_ct.shape} (Expected: 1, {patch_size[0]}, {patch_size[1]}, {patch_size[2]})")
    print(f"Condition Tensor shape: {tensor_dis.shape} (Expected: 5, {patch_size[0]}, {patch_size[1]}, {patch_size[2]})")
    print(f"Dose Tensor shape: {tensor_dose.shape} (Expected: 1, {patch_size[0]}, {patch_size[1]}, {patch_size[2]})")
    print("Success!")

if __name__ == "__main__":
    test_dir = "/data0/zhaojianxiang/dosedata/test_128/thoracic_processed_2448628/"
    test_single_sample(test_dir)
