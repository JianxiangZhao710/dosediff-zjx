import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
from collections import defaultdict

class Dataset_PSDM_train_3D(Dataset):
    def __init__(self, data_root, patch_size=(64, 128, 128)):
        self.data_root = data_root
        self.patch_size = patch_size
        
        # 1. 获取所有切片文件
        # 使用 os.path.join 确保路径正确
        ct_path = os.path.join(data_root, 'ct', '*.npy')
        ct_files = glob.glob(ct_path)
        
        if len(ct_files) == 0:
            print(f"Warning: No files found in {ct_path}")
        
        # 2. 按患者ID分组
        self.patient_dict = defaultdict(list)
        for f in ct_files:
            # 文件名格式: pt_{id}_{slice}.npy
            filename = os.path.basename(f)
            parts = filename.split('_')
            # 假设格式严格为 pt_ID_SLICE.npy
            if len(parts) >= 3:
                pt_id = parts[1]
                try:
                    slice_id = int(parts[2].split('.')[0])
                    self.patient_dict[pt_id].append((slice_id, filename))
                except ValueError:
                    pass
            
        # 3. 对每个患者的切片按 slice_id 排序
        self.patient_ids = list(self.patient_dict.keys())
        for pid in self.patient_ids:
            self.patient_dict[pid].sort(key=lambda x: x[0])
            
        print(f"Dataset initialized. Found {len(self.patient_ids)} patients.")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        slices = self.patient_dict[pid]
        
        ct_list = []
        dose_list = []
        dis_list = [] 
        
        mask_folders = [
            'Mask_PTV70', 'Mask_PTV63', 'Mask_PTV56',
            'Mask_Brainstem', 'Mask_Esophagus', 'Mask_Larynx', 
            'Mask_LeftParotid', 'Mask_Mandible', 'Mask_possible_dose_mask',
            'Mask_RightParotid', 'Mask_SpinalCord'
        ]
        
        psdm_folders = [
            'PSDM_Brainstem', 'PSDM_Esophagus', 'PSDM_Larynx',
            'PSDM_LeftParotid', 'PSDM_Mandible', 'PSDM_possible_dose_mask',
            'PSDM_PTV56', 'PSDM_PTV63', 'PSDM_PTV70',
            'PSDM_RightParotid', 'PSDM_SpinalCord'
        ]
        
        # 加载所有切片
        for _, filename in slices:
            # CT
            ct = np.load(os.path.join(self.data_root, 'ct', filename))
            ct_list.append(ct)
            
            # Dose
            dose = np.load(os.path.join(self.data_root, 'dose', filename))
            dose_list.append(dose)
            
            # Masks & PSDM
            masks = {}
            # 批量加载 masks 以减少 IO 寻找时间 (虽然还是循环，但逻辑清晰)
            for folder in mask_folders:
                masks[folder] = np.load(os.path.join(self.data_root, folder, filename))
            
            # PTVs_mask 组合
            ptvs_mask = (70.0/70.) * masks['Mask_PTV70'] + \
                        (63.0/70.) * masks['Mask_PTV63'] + \
                        (56.0/70.) * masks['Mask_PTV56']
            
            slice_dis = [ptvs_mask]
            
            # Ordered Masks
            ordered_masks = [
                'Mask_Brainstem', 'Mask_Esophagus', 'Mask_Larynx', 
                'Mask_LeftParotid', 'Mask_Mandible', 'Mask_possible_dose_mask',
                'Mask_RightParotid', 'Mask_SpinalCord'
            ]
            for folder in ordered_masks:
                slice_dis.append(masks[folder])
                
            # Ordered PSDM
            for folder in psdm_folders:
                psdm = np.load(os.path.join(self.data_root, folder, filename))
                slice_dis.append(psdm)
                
            dis_list.append(np.stack(slice_dis, axis=0)) # (C_dis, H, W)

        # Stack into Volume
        ct_vol = np.stack(ct_list, axis=0) # (D, H, W)
        dose_vol = np.stack(dose_list, axis=0) # (D, H, W)
        dis_vol = np.stack(dis_list, axis=0) # (D, C_dis, H, W)
        
        # Add Channel Dim for CT/Dose and Transpose Dis
        ct_vol = ct_vol[np.newaxis, ...] # (1, D, H, W)
        dose_vol = dose_vol[np.newaxis, ...] # (1, D, H, W)
        dis_vol = dis_vol.transpose(1, 0, 2, 3) # (C_dis, D, H, W)
        
        # Random Crop
        ct_crop, dis_crop, dose_crop = self.random_crop(ct_vol, dis_vol, dose_vol)
        
        return torch.from_numpy(ct_crop).float(), torch.from_numpy(dis_crop).float(), torch.from_numpy(dose_crop).float()

    def random_crop(self, ct, dis, dose):
        _, D, H, W = ct.shape
        pD, pH, pW = self.patch_size
        
        # 简单的 padding 逻辑 (如果尺寸不够)
        if D < pD or H < pH or W < pW:
            # 这里简单做 zero padding，实际可能需要更复杂的处理
            pad_d = max(0, pD - D)
            pad_h = max(0, pH - H)
            pad_w = max(0, pW - W)
            ct = np.pad(ct, ((0,0), (0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            dis = np.pad(dis, ((0,0), (0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            dose = np.pad(dose, ((0,0), (0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            _, D, H, W = ct.shape
            
        d_start = np.random.randint(0, D - pD + 1)
        h_start = np.random.randint(0, H - pH + 1)
        w_start = np.random.randint(0, W - pW + 1)
        
        ct_crop = ct[:, d_start:d_start+pD, h_start:h_start+pH, w_start:w_start+pW]
        dis_crop = dis[:, d_start:d_start+pD, h_start:h_start+pH, w_start:w_start+pW]
        dose_crop = dose[:, d_start:d_start+pD, h_start:h_start+pH, w_start:w_start+pW]
        
        return ct_crop, dis_crop, dose_crop
