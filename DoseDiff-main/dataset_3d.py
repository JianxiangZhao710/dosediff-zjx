import os
import numpy as np
import torch
from torch.utils.data import Dataset
import re

class Dataset_PSDM_3D_Train(Dataset):
    def __init__(self, data_root, patch_size=(32, 128, 128)):
        self.data_root = data_root
        self.patch_size = patch_size # (Depth, Height, Width)
        
        # 获取所有 CT 文件
        ct_dir = os.path.join(data_root, 'ct')
        files = os.listdir(ct_dir)
        
        # 按病人分组
        self.patient_dict = {}
        pattern = re.compile(r'pt_(\d+)_(\d+).npy')
        
        for f in files:
            match = pattern.match(f)
            if match:
                pid = match.group(1)
                sid = int(match.group(2))
                if pid not in self.patient_dict:
                    self.patient_dict[pid] = []
                self.patient_dict[pid].append((sid, f))
        
        # 排序并保存每个病人的切片列表
        self.patient_ids = list(self.patient_dict.keys())
        for pid in self.patient_ids:
            # 按 slice id 排序
            self.patient_dict[pid].sort(key=lambda x: x[0])
            # 只保留文件名
            self.patient_dict[pid] = [x[1] for x in self.patient_dict[pid]]

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        file_names = self.patient_dict[pid]
        
        # 随机选择一个起始深度
        depth = len(file_names)
        req_d, req_h, req_w = self.patch_size
        
        # 处理深度不足的情况（简单填充）
        if depth < req_d:
            start_d = 0
            selected_files = file_names
            # 循环填充直到满足 req_d
            while len(selected_files) < req_d:
                selected_files += file_names
            selected_files = selected_files[:req_d]
        else:
            start_d = np.random.randint(0, depth - req_d + 1)
            selected_files = file_names[start_d : start_d + req_d]
        
        ct_list = []
        dose_list = []
        dis_list = []
        
        for f in selected_files:
            # 加载文件
            ct = np.load(os.path.join(self.data_root, 'ct', f))
            dose = np.load(os.path.join(self.data_root, 'dose', f))
            
            # 加载 Masks
            # 简化：假设 dataset.py 中的逻辑正确，我们直接加载这些文件
            # 实际上 dataset.py 是一个个 load 的，这里为了性能应该优化，但先保持一致
            
            # Mask names from dataset.py
            loaded_masks = {}
            mask_files = [
                'Mask_PTV70', 'Mask_PTV63', 'Mask_PTV56',
                'Mask_Brainstem', 'Mask_Esophagus', 'Mask_Larynx', 'Mask_LeftParotid', 'Mask_Mandible',
                'Mask_possible_dose_mask', 'Mask_RightParotid', 'Mask_SpinalCord',
                'PSDM_Brainstem', 'PSDM_Esophagus', 'PSDM_Larynx', 'PSDM_LeftParotid', 'PSDM_Mandible',
                'PSDM_possible_dose_mask', 'PSDM_PTV56', 'PSDM_PTV63', 'PSDM_PTV70',
                'PSDM_RightParotid', 'PSDM_SpinalCord'
            ]
            
            for m_name in mask_files:
                loaded_masks[m_name] = np.load(os.path.join(self.data_root, m_name, f))
            
            # 计算 PTVs_mask
            PTVs_mask = (70.0 / 70. * loaded_masks['Mask_PTV70'] + 
                         63.0 / 70. * loaded_masks['Mask_PTV63'] + 
                         56.0 / 70. * loaded_masks['Mask_PTV56'])
            
            # 组装 dis (20 channels)
            dis_slice_list = [PTVs_mask]
            
            # 8 Masks
            dis_slice_list.extend([loaded_masks[k] for k in [
                'Mask_Brainstem', 'Mask_Esophagus', 'Mask_Larynx', 'Mask_LeftParotid', 'Mask_Mandible',
                'Mask_possible_dose_mask', 'Mask_RightParotid', 'Mask_SpinalCord'
            ]])
            
            # 11 PSDMs
            dis_slice_list.extend([loaded_masks[k] for k in [
                'PSDM_Brainstem', 'PSDM_Esophagus', 'PSDM_Larynx', 'PSDM_LeftParotid', 'PSDM_Mandible',
                'PSDM_possible_dose_mask', 'PSDM_PTV56', 'PSDM_PTV63', 'PSDM_PTV70',
                'PSDM_RightParotid', 'PSDM_SpinalCord'
            ]])
            
            dis_slice_stack = np.stack(dis_slice_list, axis=0) # (C, H, W)
            dis_list.append(dis_slice_stack)
            
            ct_list.append(ct)
            dose_list.append(dose)

        # 转换为 Volume
        ct_vol = np.stack(ct_list, axis=0)       # (D, H, W)
        dose_vol = np.stack(dose_list, axis=0)   # (D, H, W)
        dis_vol = np.stack(dis_list, axis=1)     # (D, C, H, W) -> stack on dim 1 (depth) is wrong for stack list of (C,H,W)
                                                 # dis_list is list of (C, H, W). stack(axis=0) -> (D, C, H, W)
        dis_vol = np.stack(dis_list, axis=0)
        
        # 调整维度顺序
        # CT/Dose: (D, H, W) -> (1, D, H, W)
        ct_vol = ct_vol[np.newaxis, ...]
        dose_vol = dose_vol[np.newaxis, ...]
        
        # Dis: (D, C, H, W) -> (C, D, H, W)
        dis_vol = dis_vol.transpose(1, 0, 2, 3)
        
        return torch.from_numpy(ct_vol).float(), torch.from_numpy(dis_vol).float(), torch.from_numpy(dose_vol).float()
