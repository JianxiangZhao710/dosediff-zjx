import os
import glob
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

def resize_nifti(file_path, is_mask=False):
    try:
        # Load NIfTI
        img = nib.load(file_path)
        data = img.get_fdata()
        affine = img.affine
        header = img.header
        
        # Check current shape
        current_shape = data.shape
        if current_shape == (512, 512, 512) or current_shape == (512, 512, 512, 1):
            return None # Already correct size
            
        # Convert to torch tensor for resizing
        # Torch expects shape: [Batch, Channel, Depth, Height, Width] -> [1, 1, Z, Y, X]
        # Our data is typically [X, Y, Z]
        data_tensor = torch.from_numpy(data).float()
        
        # Handle 4D data (if it has a single channel dimension)
        if len(data.shape) == 4 and data.shape[3] == 1:
            data_tensor = data_tensor.squeeze(3)
            
        # Add Batch and Channel dimensions: [1, 1, X, Y, Z]
        data_tensor = data_tensor.unsqueeze(0).unsqueeze(0)
        
        # Setup interpolation mode
        # Masks (radio_biology_map) MUST use nearest neighbor to preserve discrete classes
        # CT and Dose use trilinear for smooth interpolation
        mode = 'nearest' if is_mask else 'trilinear'
        align_corners = None if is_mask else False
        
        # Resize to 512x512x512
        # F.interpolate expects size format [Depth, Height, Width] corresponding to the last 3 dims
        resized_tensor = F.interpolate(
            data_tensor, 
            size=(512, 512, 512), 
            mode=mode, 
            align_corners=align_corners
        )
        
        # Convert back to numpy: [X, Y, Z]
        resized_data = resized_tensor.squeeze(0).squeeze(0).numpy()
        
        # If original was 4D, make it 4D again
        if len(current_shape) == 4:
            resized_data = np.expand_dims(resized_data, axis=-1)
            
        # VERY IMPORTANT: Adjust the affine matrix z-spacing
        # If we change the number of Z slices from Z_old to 512, 
        # the physical spacing in Z axis must be scaled by (Z_old / 512)
        new_affine = affine.copy()
        z_scale_factor = current_shape[2] / 512.0
        
        # The affine matrix stores scaling in the diagonal elements (0,0), (1,1), (2,2)
        # Assuming Z is the 3rd dimension (index 2)
        new_affine[2, 2] = affine[2, 2] * z_scale_factor
        
        # Create new NIfTI image
        # Using original datatype for masks (usually int) or float32 for continuous
        if is_mask:
            resized_data = np.round(resized_data).astype(data.dtype)
        else:
            resized_data = resized_data.astype(np.float32)
            
        new_img = nib.Nifti1Image(resized_data, new_affine, header)
        
        return new_img
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# Do not actually run this on the full dataset during thought process, just writing out the logic
