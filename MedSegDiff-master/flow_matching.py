import torch
import torch.nn as nn
import numpy as np

class FlowMatching(nn.Module):
    def __init__(self, net, sigma_min=1e-4):
        super().__init__()
        self.net = net
        self.sigma_min = sigma_min

    def get_loss(self, x_1, ct, syn, dis):
        """
        x_1: Ground Truth Dose (B, C, D, H, W)
        ct: Condition CT
        syn: Condition Synthetic Dose
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
        # 模型输入: x_t, t, ct, syn, dis
        # Time step scaled to [0, 1000] for standard sinusoidal embeddings
        pred_v = self.net(x_t, t * 1000.0, ct, syn, dis)
        
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
    def sample(self, ct, syn, dis, steps=50):
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
            v = self.net(x, t * 1000.0, ct, syn, dis)
            
            # Euler update
            x = x + v * dt
            
        return x


# ----------------------------- v1.3: CT + SYN only (no DIS) -----------------------------
class FlowMatchingV13(nn.Module):
    """FlowMatching wrapper for v1.3 — 3-condition (CT / SYN / t), no DIS."""

    def __init__(self, net, sigma_min=1e-4):
        super().__init__()
        self.net = net
        self.sigma_min = sigma_min

    def get_loss(self, x_1, ct, syn):
        """
        x_1: Ground Truth Dose (B, C, D, H, W)
        ct : Condition CT
        syn: Condition Synthetic Dose
        """
        b = x_1.shape[0]
        device = x_1.device

        t = torch.rand((b,), device=device).type_as(x_1)
        x_0 = torch.randn_like(x_1)

        t_expand = t.view(b, 1, 1, 1, 1)
        x_t = (1 - (1 - self.sigma_min) * t_expand) * x_0 + t_expand * x_1
        target_v = x_1 - (1 - self.sigma_min) * x_0

        pred_v = self.net(x_t, t * 1000.0, ct, syn)

        loss = torch.nn.functional.mse_loss(pred_v, target_v)
        return loss

    @torch.no_grad()
    def sample(self, ct, syn, steps=50):
        """Euler sampling for v1.3."""
        b = ct.shape[0]
        device = ct.device

        x = torch.randn(b, 1, ct.shape[2], ct.shape[3], ct.shape[4], device=device)
        dt = 1.0 / steps

        for i in range(steps):
            t_val = i / steps
            t = torch.ones(b, device=device) * t_val
            v = self.net(x, t * 1000.0, ct, syn)
            x = x + v * dt

        return x


# ----------------------------- v1.4: CT + SYN + DIS + Penalty -----------------------------
class FlowMatchingV14(nn.Module):
    """FlowMatching wrapper for v1.4 — v1.2 4-condition (CT/SYN/DIS) plus
    a 3D penalty field consumed by the ControlNet-lite penalty adapter.

    The flow-matching training objective and the spatially-weighted MSE
    loss are identical to the v1 ``FlowMatching`` wrapper. The only
    interface change is one extra ``penalty`` argument.
    """

    def __init__(self, net, sigma_min=1e-4):
        super().__init__()
        self.net = net
        self.sigma_min = sigma_min

    def get_loss(self, x_1, ct, syn, dis, penalty):
        """
        x_1     : Ground Truth Dose (B, 1, D, H, W)
        ct      : (B, 1,  D, H, W)
        syn     : (B, 1,  D, H, W)
        dis     : (B, 11, D, H, W)
        penalty : (B, P,  D, H, W)  P = penalty_channels
        """
        b = x_1.shape[0]
        device = x_1.device

        t = torch.rand((b,), device=device).type_as(x_1)
        x_0 = torch.randn_like(x_1)

        t_expand = t.view(b, 1, 1, 1, 1)
        x_t = (1 - (1 - self.sigma_min) * t_expand) * x_0 + t_expand * x_1
        target_v = x_1 - (1 - self.sigma_min) * x_0

        pred_v = self.net(x_t, t * 1000.0, ct, syn, dis, penalty)

        # Same spatial weighting scheme as v1 FlowMatching:
        # PTV (10) > OAR (5) > background (0.5)
        bg_weight = 0.5
        oar_weight = 5.0
        ptv_weight = 10.0

        is_ptv = torch.clamp(torch.sum(dis[:, 0:3, ...], dim=1, keepdim=True), 0.0, 1.0)
        is_oar = torch.clamp(torch.sum(dis[:, 3:10, ...], dim=1, keepdim=True), 0.0, 1.0)
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
    def sample(self, ct, syn, dis, penalty, steps=50):
        """Euler sampling for v1.4."""
        b = ct.shape[0]
        device = ct.device

        x = torch.randn(b, 1, ct.shape[2], ct.shape[3], ct.shape[4], device=device)
        dt = 1.0 / steps

        for i in range(steps):
            t_val = i / steps
            t = torch.ones(b, device=device) * t_val
            v = self.net(x, t * 1000.0, ct, syn, dis, penalty)
            x = x + v * dt

        return x
