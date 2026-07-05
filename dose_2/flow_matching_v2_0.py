"""
v2.0 — Flow-Matching wrapper for the Dynamic Relation-to-Field
Constraint Router model.

``FlowMatchingV20`` extends the v1.4/v1.5 wrapper (``FlowMatchingV14``)
with three additions, all *optional* and disabled-by-default-safe:

  1. Router regularisation
       * mean-injection penalty (discourage opening the router
         everywhere -> avoids degenerating into global injection),
       * smoothness / total-variation penalty (avoid noisy router maps).
     These read the scalars the v2.0 UNet computes during its forward
     pass (``_router_mean_reg`` / ``_router_tv_reg``).

  2. Constraint-aware morphology loss (optional, small weight)
       * risk-region gradient-consistency loss,
       * high-frequency structure-preservation loss.
     Both operate on an estimated *clean dose* recovered from the
     predicted velocity, and are weighted toward high-risk regions via
     the base risk map ``M_risk_base`` exposed by the UNet.

  3. Component bookkeeping (``self.last_components``) for logging.

The core Flow-Matching objective and the spatially-weighted MSE are
identical to v1.4/v1.5; the Euler sampler is inherited unchanged (the
v2.0 forward signature ``(x_t, t, ct, syn, dis, penalty)`` is the same).

Predicted clean-dose estimate
-----------------------------
For the OT path ``x_t = (1-(1-σ)t) x_0 + t x_1`` with target velocity
``v = x_1 - (1-σ) x_0``, one can show ``x_t + (1-t) v = σ x_0 + x_1``.
With ``σ = σ_min ≈ 1e-4`` this gives a stable estimate

    x1_hat ≈ x_t + (1 - t) * v_pred

which we use for the morphology losses. It is exact in the limit
σ_min -> 0 and well-behaved across t (at t->1 it collapses to x_t≈x_1).
"""
import torch
import torch.nn.functional as F

from flow_matching import FlowMatchingV14


def _spatial_gradients(x):
    """Forward-difference spatial gradients of (B,1,D,H,W).

    Returns three tensors cropped to a common shape so they can be
    compared / weighted elementwise.
    """
    gx = x[:, :, 1:, 1:, 1:] - x[:, :, :-1, 1:, 1:]
    gy = x[:, :, 1:, 1:, 1:] - x[:, :, 1:, :-1, 1:]
    gz = x[:, :, 1:, 1:, 1:] - x[:, :, 1:, 1:, :-1]
    return gx, gy, gz


def _avg_lowpass(x, k=5):
    """Low-pass via average pooling (stride 1) -> high-freq = x - lowpass."""
    pad = k // 2
    return F.avg_pool3d(x, kernel_size=k, stride=1, padding=pad)


class FlowMatchingV20(FlowMatchingV14):
    """Flow-Matching wrapper for v2.0 (dynamic constraint router)."""

    def __init__(
        self,
        net,
        sigma_min=1e-4,
        lambda_grad_loss=0.0,
        lambda_hf_loss=0.0,
        router_reg_weight=0.0,
        router_smooth_weight=0.0,
        hf_kernel=5,
        morphology_risk_weighted=True,
        lambda_global_l1=0.0,
        global_l1_use_body_mask=True,
        body_ct_threshold=-0.999,
    ):
        super().__init__(net, sigma_min)
        self.lambda_grad_loss = float(lambda_grad_loss)
        self.lambda_hf_loss = float(lambda_hf_loss)
        self.router_reg_weight = float(router_reg_weight)
        self.router_smooth_weight = float(router_smooth_weight)
        self.hf_kernel = int(hf_kernel)
        self.morphology_risk_weighted = bool(morphology_risk_weighted)
        # Additive global (body) L1 on the estimated clean dose. Targets the
        # many low-weight background / falloff voxels that the structure-
        # weighted MSE under-fits, improving voxel-wise Dose Score while
        # leaving the structure-region (DVH) fit of the base loss intact.
        self.lambda_global_l1 = float(lambda_global_l1)
        self.global_l1_use_body_mask = bool(global_l1_use_body_mask)
        self.body_ct_threshold = float(body_ct_threshold)
        self.last_components = {}

    def _raw_net(self):
        return self.net.module if hasattr(self.net, "module") else self.net

    def _weighted_mse(self, pred_v, target_v, dis):
        bg_weight, oar_weight, ptv_weight = 0.5, 5.0, 10.0
        is_ptv = torch.clamp(torch.sum(dis[:, 0:3, ...], dim=1, keepdim=True), 0.0, 1.0)
        is_oar = torch.clamp(torch.sum(dis[:, 3:10, ...], dim=1, keepdim=True), 0.0, 1.0)
        oar_only = is_oar * (1.0 - is_ptv)
        bg_only = (1.0 - is_oar) * (1.0 - is_ptv)
        spatial_weight = is_ptv * ptv_weight + oar_only * oar_weight + bg_only * bg_weight
        sq = (pred_v - target_v) ** 2
        return torch.sum(sq * spatial_weight) / (torch.sum(spatial_weight) + 1e-8)

    def _morphology_loss(self, pred_clean, gt, risk_map):
        """Risk-region gradient + high-frequency structure-preservation."""
        grad_loss = torch.zeros((), device=pred_clean.device)
        hf_loss = torch.zeros((), device=pred_clean.device)

        if self.lambda_grad_loss > 0.0:
            pgx, pgy, pgz = _spatial_gradients(pred_clean)
            ggx, ggy, ggz = _spatial_gradients(gt)
            diff = (pgx - ggx).abs() + (pgy - ggy).abs() + (pgz - ggz).abs()
            if self.morphology_risk_weighted and risk_map is not None:
                w = risk_map[:, :, 1:, 1:, 1:]
                grad_loss = (diff * w).sum() / (w.sum() + 1e-8)
            else:
                grad_loss = diff.mean()

        if self.lambda_hf_loss > 0.0:
            hf_pred = pred_clean - _avg_lowpass(pred_clean, self.hf_kernel)
            hf_gt = gt - _avg_lowpass(gt, self.hf_kernel)
            diff = (hf_pred - hf_gt).abs()
            if self.morphology_risk_weighted and risk_map is not None:
                hf_loss = (diff * risk_map).sum() / (risk_map.sum() + 1e-8)
            else:
                hf_loss = diff.mean()

        return grad_loss, hf_loss

    def get_loss(self, x_1, ct, syn, dis, penalty):
        b = x_1.shape[0]
        device = x_1.device

        t = torch.rand((b,), device=device).type_as(x_1)
        x_0 = torch.randn_like(x_1)
        t_expand = t.view(b, 1, 1, 1, 1)
        x_t = (1 - (1 - self.sigma_min) * t_expand) * x_0 + t_expand * x_1
        target_v = x_1 - (1 - self.sigma_min) * x_0

        pred_v = self.net(x_t, t * 1000.0, ct, syn, dis, penalty)

        base_loss = self._weighted_mse(pred_v, target_v, dis)
        total = base_loss

        raw = self._raw_net()
        comps = {"base": base_loss.detach()}

        # ---- Router regularisation ----
        if self.router_reg_weight > 0.0 and hasattr(raw, "_router_mean_reg"):
            reg = raw._router_mean_reg.to(device).float()
            total = total + self.router_reg_weight * reg
            comps["router_mean"] = reg.detach()
        if self.router_smooth_weight > 0.0 and hasattr(raw, "_router_tv_reg"):
            tv = raw._router_tv_reg.to(device).float()
            total = total + self.router_smooth_weight * tv
            comps["router_tv"] = tv.detach()

        # ---- Estimated clean dose (shared by global-L1 + morphology) ----
        need_clean = (
            self.lambda_global_l1 > 0.0
            or self.lambda_grad_loss > 0.0
            or self.lambda_hf_loss > 0.0
        )
        pred_clean = x_t + (1.0 - t_expand) * pred_v if need_clean else None

        # ---- Additive global (body) L1: improves voxel-wise Dose Score ----
        if self.lambda_global_l1 > 0.0:
            if self.global_l1_use_body_mask:
                body = (ct > self.body_ct_threshold).to(pred_clean.dtype)
            else:
                body = torch.ones_like(x_1)
            l1 = (pred_clean - x_1).abs()
            g_l1 = (l1 * body).sum() / (body.sum() + 1e-8)
            total = total + self.lambda_global_l1 * g_l1
            comps["global_l1"] = g_l1.detach()

        # ---- Optional morphology loss (estimated clean dose) ----
        if self.lambda_grad_loss > 0.0 or self.lambda_hf_loss > 0.0:
            risk_map = getattr(raw, "_last_m_risk", None)
            if risk_map is not None:
                risk_map = risk_map.to(device).float()
            grad_loss, hf_loss = self._morphology_loss(pred_clean, x_1, risk_map)
            if self.lambda_grad_loss > 0.0:
                total = total + self.lambda_grad_loss * grad_loss
                comps["grad"] = grad_loss.detach()
            if self.lambda_hf_loss > 0.0:
                total = total + self.lambda_hf_loss * hf_loss
                comps["hf"] = hf_loss.detach()

        comps["total"] = total.detach()
        self.last_components = comps
        return total
