import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset


OPENKBP_MASK_NAMES = [
    'Mask_PTV70',
    'Mask_PTV63',
    'Mask_PTV56',
    'Mask_Brainstem',
    'Mask_SpinalCord',
    'Mask_RightParotid',
    'Mask_LeftParotid',
    'Mask_Esophagus',
    'Mask_Larynx',
    'Mask_Mandible',
    'Mask_possible_dose_mask',
]


class Dataset_PSDM_3D_Train(Dataset):
    """OpenKBP 3D 数据集：输入 CT + 11 个二值 mask 作为条件，目标为 dose。

    每个病人目录应包含：
        - ct.nii.gz
        - dose.nii.gz
        - Mask_<organ>.nii.gz  (共 11 个，缺失自动用全零代替)
    """

    def __init__(self, data_root, patch_size=(64, 128, 128), mask_names=None):
        self.data_root = data_root
        self.patch_size = patch_size
        self.mask_names = mask_names if mask_names is not None else OPENKBP_MASK_NAMES
        self.num_mask_channels = len(self.mask_names)
        self.patient_list = sorted([
            p for p in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, p))
        ])

        print(
            f"Dataset initialized. Found {len(self.patient_list)} patients in {data_root}. "
            f"Using {self.num_mask_channels} mask channels."
        )

    def __len__(self):
        return len(self.patient_list)

    def normalize_ct(self, img):
        img = np.clip(img, 0, 2500)
        img = img / 1250.0 - 1.0
        return img

    def normalize_dose(self, img):
        img = np.clip(img, 0, 80)
        img = img / 40.0 - 1.0
        return img

    def load_nii(self, path):
        if os.path.exists(path):
            return nib.load(path).get_fdata().astype(np.float32)
        return None

    def __getitem__(self, index):
        patient_id = self.patient_list[index]
        patient_dir = os.path.join(self.data_root, patient_id)

        ct_nii = nib.load(os.path.join(patient_dir, 'ct.nii.gz'))
        ct_data = ct_nii.get_fdata().astype(np.float32).transpose(2, 0, 1)  # (D, H, W)

        dose_data = nib.load(os.path.join(patient_dir, 'dose.nii.gz')).get_fdata().astype(np.float32)
        dose_data = dose_data.transpose(2, 0, 1)

        empty_shape = ct_data.shape  # (D, H, W)

        mask_channels = []
        for name in self.mask_names:
            m = self.load_nii(os.path.join(patient_dir, '{}.nii.gz'.format(name)))
            if m is None:
                m = np.zeros(empty_shape, dtype=np.float32)
            else:
                m = m.transpose(2, 0, 1)
                if m.shape != empty_shape:
                    m = np.zeros(empty_shape, dtype=np.float32)
            m = (m > 0).astype(np.float32)
            mask_channels.append(m)
        dis_stack = np.stack(mask_channels, axis=0)  # (11, D, H, W)

        ct_data = self.normalize_ct(ct_data)
        dose_data = self.normalize_dose(dose_data)

        current_depth = ct_data.shape[0]
        target_depth = self.patch_size[0]

        if current_depth > target_depth:
            start_z = np.random.randint(0, current_depth - target_depth)
            end_z = start_z + target_depth

            ct_crop = ct_data[start_z:end_z, :, :]
            dose_crop = dose_data[start_z:end_z, :, :]
            dis_crop = dis_stack[:, start_z:end_z, :, :]
        elif current_depth < target_depth:
            pad_z = target_depth - current_depth
            ct_crop = np.pad(ct_data, ((0, pad_z), (0, 0), (0, 0)), 'constant', constant_values=-1.0)
            dose_crop = np.pad(dose_data, ((0, pad_z), (0, 0), (0, 0)), 'constant', constant_values=-1.0)
            dis_crop = np.pad(dis_stack, ((0, 0), (0, pad_z), (0, 0), (0, 0)), 'constant', constant_values=0)
        else:
            ct_crop = ct_data
            dose_crop = dose_data
            dis_crop = dis_stack

        ct_crop = ct_crop[np.newaxis, ...]
        dose_crop = dose_crop[np.newaxis, ...]

        return (
            torch.from_numpy(ct_crop),       # (1,  D, H, W)
            torch.from_numpy(dis_crop),      # (11, D, H, W)
            torch.from_numpy(dose_crop),     # (1,  D, H, W)
        )
