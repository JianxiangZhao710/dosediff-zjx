import torch
import torch.nn as nn
import numpy as np

class FlowMatching(nn.Module):
    def __init__(self, net, sigma_min=1e-4):
        super().__init__()
        self.net = net
        self.sigma_min = sigma_min

    def get_loss(self, x_1, ct, dis):
        """
        x_1: Ground Truth Dose (B, C, D, H, W)
        ct: Condition CT
        dis: Condition Dis
        """
        b = x_1.shape[0]
        device = x_1.device
        
        # 1. 采样时间步 t ~ Uniform(0, 1)
        t = torch.rand((b,), device=device).type_as(x_1)
        
        # 2. 采样初始噪声 x_0 ~ N(0, I)
        x_0 = torch.randn_like(x_1)
        
        # 3. 构造中间态 x_t (Optimal Transport path)
        # x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
        # 广播 t 到 (B, 1, 1, 1, 1)
        t_expand = t.view(b, 1, 1, 1, 1)
        x_t = (1 - (1 - self.sigma_min) * t_expand) * x_0 + t_expand * x_1
        
        # 4. 计算目标向量场 v_target
        # target = x_1 - (1 - sigma_min) * x_0
        target_v = x_1 - (1 - self.sigma_min) * x_0
        
        # 5. 模型预测
        # 模型输入: x_t, t, ct, dis
        # Time step scaled to [0, 1000] for standard sinusoidal embeddings
        pred_v = self.net(x_t, t * 1000.0, ct, dis)
        
        # 6. Spatially weighted loss with strict weighted-average normalization.
        # OpenKBP 11-channel convention (binary masks, may overlap):
        #   dis[:, 0:3]   -> PTVs            (PTV70, PTV63, PTV56)
        #   dis[:, 3:10]  -> OARs            (Brainstem, SpinalCord, RPa, LPa,
        #                                     Esophagus, Larynx, Mandible)
        #   dis[:, 10:11] -> possible_dose_mask (body envelope, unused here)
        bg_weight = 0.5
        oar_weight = 5.0
        ptv_weight = 10.0

        is_ptv = torch.clamp(torch.sum(dis[:, 0:3, ...], dim=1, keepdim=True), 0.0, 1.0)
        is_oar = torch.clamp(torch.sum(dis[:, 3:10, ...], dim=1, keepdim=True), 0.0, 1.0)
        # Priority: PTV > OAR > background (so a voxel can't be double-counted)
        oar_only = is_oar * (1.0 - is_ptv)
        bg_only = (1.0 - is_oar) * (1.0 - is_ptv)
        spatial_weight = (
            is_ptv * ptv_weight
            + oar_only * oar_weight
            + bg_only * bg_weight
        )

        squared_error = (pred_v - target_v) ** 2
        weighted_squared_error = squared_error * spatial_weight
        loss = torch.sum(weighted_squared_error) / (torch.sum(spatial_weight) + 1e-8)
        return loss

    @torch.no_grad()
    def sample(self, ct, dis, steps=50):
        """
        Euler method sampling
        """
        b = ct.shape[0]
        device = ct.device
        
        # x_0 ~ N(0, I)
        # 假设 ct 是 (B, 1, D, H, W)，output dose 也是 (B, 1, D, H, W)
        # 这里的形状需要与 dataset 中的 dose shape 一致
        x = torch.randn(b, 1, ct.shape[2], ct.shape[3], ct.shape[4], device=device)
        
        dt = 1.0 / steps
        
        for i in range(steps):
            t_val = i / steps
            t = torch.ones(b, device=device) * t_val
            
            # 预测速度 v
            # Time step scaled to [0, 1000] for standard sinusoidal embeddings
            v = self.net(x, t * 1000.0, ct, dis)
            
            # Euler update
            x = x + v * dt
            
        return x
