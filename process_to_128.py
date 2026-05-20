import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm
from pathlib import Path
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ignore warnings from nibabel about headers
warnings.filterwarnings("ignore")

def process_sample(args):
    subdir, output_dir = args
    try:
        ct_path = subdir / "ct.nii.gz"
        dose_path = subdir / "dose_resampled.nii.gz"
        mask_path = subdir / "radio_biology_map.nii.gz"
        
        if not (ct_path.exists() and dose_path.exists() and mask_path.exists()):
            return "Missing files"
            
        # 1. Load mask to determine Z size and center
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        Z_orig = mask_data.shape[2]
        
        # Filter samples with Z > 160
        if Z_orig > 160:
            return "Z > 160, skipped"
            
        target_Z = 128
        target_XY = 128
        
        # Calculate Z crop/pad parameters
        pad_front, pad_back = 0, 0
        start_z, end_z = 0, Z_orig
        
        if Z_orig < target_Z:
            pad_total = target_Z - Z_orig
            pad_front = pad_total // 2
            pad_back = pad_total - pad_front
        elif Z_orig > target_Z:
            # Find tumor center. Usually PTV is class 1.
            # If no 1, use any > 0
            nz_indices = np.where(mask_data == 1)[2]
            if len(nz_indices) == 0:
                nz_indices = np.where(mask_data > 0)[2]
            
            if len(nz_indices) > 0:
                center_z = int(np.median(nz_indices))
            else:
                center_z = Z_orig // 2
                
            start_z = max(0, center_z - target_Z // 2)
            end_z = start_z + target_Z
            # Handle boundary
            if end_z > Z_orig:
                end_z = Z_orig
                start_z = max(0, end_z - target_Z)
                
        def process_volume(data, affine, is_mask, is_ct):
            # Scale factors for XY (Dynamic calculation, safe for any resolution)
            zoom_factors = [target_XY / data.shape[0], target_XY / data.shape[1], 1.0]
            
            # Use order=0 for mask (nearest), order=1 for continuous
            order = 0 if is_mask else 1
            
            # Resize XY only
            resized = zoom(data, zoom_factors, order=order, mode='nearest')
            
            # Crop or Pad along Z axis (axis=2)
            if Z_orig < target_Z:
                pad_val = -1000 if is_ct else 0
                final_data = np.pad(
                    resized, 
                    pad_width=((0,0), (0,0), (pad_front, pad_back)), 
                    mode='constant', 
                    constant_values=pad_val
                )
            elif Z_orig > target_Z:
                final_data = resized[:, :, start_z:end_z]
            else:
                final_data = resized
                
            # Adjust Affine Matrix
            new_affine = affine.copy()
            
            # 【核心修复点】: 动态获取真实尺寸，计算各自的物理缩放比例，废除硬编码 512.0
            xy_scale_x = data.shape[0] / target_XY
            xy_scale_y = data.shape[1] / target_XY
            
            new_affine[0, 0] *= xy_scale_x
            new_affine[1, 1] *= xy_scale_y
            
            # Z origin shift
            if Z_orig < target_Z:
                new_affine[2, 3] += (-pad_front) * affine[2, 2]
            elif Z_orig > target_Z:
                new_affine[2, 3] += start_z * affine[2, 2]
                
            if is_mask:
                final_data = np.round(final_data).astype(np.int16)
            else:
                final_data = final_data.astype(np.float32)
                
            return final_data, new_affine

        out_subdir = output_dir / subdir.name
        out_subdir.mkdir(parents=True, exist_ok=True)
        
        # Process Mask
        m_data, m_aff = process_volume(mask_data, mask_img.affine, is_mask=True, is_ct=False)
        nib.save(nib.Nifti1Image(m_data, m_aff, mask_img.header), out_subdir / "radio_biology_map.nii.gz")
        
        # Process CT
        ct_img = nib.load(ct_path)
        c_data, c_aff = process_volume(ct_img.get_fdata(), ct_img.affine, is_mask=False, is_ct=True)
        nib.save(nib.Nifti1Image(c_data, c_aff, ct_img.header), out_subdir / "ct.nii.gz")
        
        # Process Dose
        dose_img = nib.load(dose_path)
        d_data, d_aff = process_volume(dose_img.get_fdata(), dose_img.affine, is_mask=False, is_ct=False)
        nib.save(nib.Nifti1Image(d_data, d_aff, dose_img.header), out_subdir / "dose_resampled.nii.gz")
        
        return "Success"
        
    except Exception as e:
        return f"Error in {subdir.name}: {str(e)}"

def main():
    input_dir = Path("/data0/zhaojianxiang/dosedata/test_sample/")
    output_dir = Path("/data0/zhaojianxiang/dosedata/test_128/")
    
    # Create target directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    # Prepare arguments for multiprocessing
    tasks = [(d, output_dir) for d in subdirs]
    
    results = {
        "Success": 0, 
        "Z > 160, skipped": 0, 
        "Missing files": 0, 
        "Errors": 0
    }
    
    print(f"Total directories found: {len(tasks)}")
    print("Processing using multi-processing...")
    
    # Use ProcessPoolExecutor to parallelize and speed up
    with ProcessPoolExecutor(max_workers=os.cpu_count() // 2 or 1) as executor:
        futures = [executor.submit(process_sample, arg) for arg in tasks]
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Converting samples"):
            res = future.result()
            if res == "Success":
                results["Success"] += 1
            elif res == "Z > 160, skipped":
                results["Z > 160, skipped"] += 1
            elif res == "Missing files":
                results["Missing files"] += 1
            else:
                results["Errors"] += 1
                # print(res) # Only print if you want to see detailed errors
                
    print("\n" + "="*40)
    print("Processing Complete Summary")
    print("="*40)
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()