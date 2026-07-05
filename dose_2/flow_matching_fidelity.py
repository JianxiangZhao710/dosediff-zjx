"""
Fidelity-first Flow-Matching wrapper (v2.0 backbone).

Goal: make the generated dose match the *ground-truth label* as closely as
possible at the voxel level (Dose Score / MAE), rather than biasing the fit
toward clinical structures (the structure-weighted DVH-oriented objective in
``FlowMatchingV20``).

This file is **standalone** and does NOT modify any existing module. It
subclasses ``FlowMatchingV20`` and overrides the loss with a fidelity design:

    L_total = L_base    (velocity MSE, with *flattened* spatial weights)
            + λ_l1 · L1(pred_clean, x_1, body)      # main fidelity term
            + λ_l2 · L2(pred_clean, x_1, body)      # stabilise / penalise outliers
            + λ_grad · L_grad(full image)           # anti over-smoothing
            + λ_hf   · L_hf(full image)             # anti over-smoothing
            + (optional router regularisers)

Key differences vs. FlowMatchingV20:
  * Spatial weights for the base velocity MSE are configurable and default to
    a near-uniform ``ptv=2 / oar=1.5 / bg=1`` (vs. the DVH-oriented 10/5/0.5).
  * A direct clean-dose reconstruction loss (L1 + small L2) is the dominant
    term, computed over the CT-derived body mask (or whole volume).
  * Morphology grad/HF losses default to **full-image** (not risk-weighted),
    so sharpness is preserved everywhere, not just in high-risk regions.

The clean-dose estimate uses the OT identity ``x1_hat ≈ x_t + (1-t)·v_pred``
(exact as σ_min → 0), identical to FlowMatchingV20.
"""
import torch

from flow_matching_v2_0 import FlowMatchingV20


class FlowMatchingFidelity(FlowMatchingV20):
    """Fidelity-first loss: prioritise voxel-wise match to the label."""

    def __init__(
        self,
        net,
        sigma_min=1e-4,
        # --- flattened base velocity-MSE spatial weights ---
        ptv_weight=2.0,
        oar_weight=1.5,
        bg_weight=1.0,
        # --- clean-dose reconstruction (main fidelity term) ---
        lambda_recon_l1=1.0,
        lambda_recon_l2=0.2,
        recon_use_body_mask=True,
        body_ct_threshold=-0.999,
        # --- anti over-smoothing (full image by default) ---
        lambda_grad_loss=0.05,
        lambda_hf_loss=0.05,
        hf_kernel=5,
        morphology_risk_weighted=False,
        # --- optional router regularisers (kept off by default) ---
        router_reg_weight=0.0,
        router_smooth_weight=0.0,
    ):
        super().__init__(
            net,
            sigma_min=sigma_min,
            lambda_grad_loss=lambda_grad_loss,
            lambda_hf_loss=lambda_hf_loss,
            router_reg_weight=router_reg_weight,
            router_smooth_weight=router_smooth_weight,
            hf_kernel=hf_kernel,
            morphology_risk_weighted=morphology_risk_weighted,
            lambda_global_l1=0.0,  # recon handled explicitly below
        )
        self.ptv_weight = float(ptv_weight)
        self.oar_weight = float(oar_weight)
        self.bg_weight = float(bg_weight)
        self.lambda_recon_l1 = float(lambda_recon_l1)
        self.lambda_recon_l2 = float(lambda_recon_l2)
        self.recon_use_body_mask = bool(recon_use_body_mask)
        self.body_ct_threshold = float(body_ct_threshold)

    def _weighted_mse(self, pred_v, target_v, dis):
        """Velocity MSE with configurable (flattened) spatial weights."""
        is_ptv = torch.clamp(torch.sum(dis[:, 0:3, ...], dim=1, keepdim=True), 0.0, 1.0)
        is_oar = torch.clamp(torch.sum(dis[:, 3:10, ...], dim=1, keepdim=True), 0.0, 1.0)
        oar_only = is_oar * (1.0 - is_ptv)
        bg_only = (1.0 - is_oar) * (1.0 - is_ptv)
        spatial_weight = (
            is_ptv * self.ptv_weight
            + oar_only * self.oar_weight
            + bg_only * self.bg_weight
        )
        sq = (pred_v - target_v) ** 2
        return torch.sum(sq * spatial_weight) / (torch.sum(spatial_weight) + 1e-8)

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
        comps = {"base": base_loss.detach()}
        raw = self._raw_net()

        # ---- Estimated clean dose (OT identity) ----
        pred_clean = x_t + (1.0 - t_expand) * pred_v

        # ---- Body mask for reconstruction ----
        if self.recon_use_body_mask:
            body = (ct > self.body_ct_threshold).to(pred_clean.dtype)
        else:
            body = torch.ones_like(x_1)
        denom = body.sum() + 1e-8

        # ---- Main fidelity terms: L1 (+ small L2) on clean dose ----
        if self.lambda_recon_l1 > 0.0:
            l1 = ((pred_clean - x_1).abs() * body).sum() / denom
            total = total + self.lambda_recon_l1 * l1
            comps["recon_l1"] = l1.detach()
        if self.lambda_recon_l2 > 0.0:
            l2 = (((pred_clean - x_1) ** 2) * body).sum() / denom
            total = total + self.lambda_recon_l2 * l2
            comps["recon_l2"] = l2.detach()

        # ---- Anti over-smoothing: grad / high-frequency (full image) ----
        if self.lambda_grad_loss > 0.0 or self.lambda_hf_loss > 0.0:
            risk_map = None
            if self.morphology_risk_weighted:
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

        # ---- Optional router regularisation ----
        if self.router_reg_weight > 0.0 and hasattr(raw, "_router_mean_reg"):
            reg = raw._router_mean_reg.to(device).float()
            total = total + self.router_reg_weight * reg
            comps["router_mean"] = reg.detach()
        if self.router_smooth_weight > 0.0 and hasattr(raw, "_router_tv_reg"):
            tv = raw._router_tv_reg.to(device).float()
            total = total + self.router_smooth_weight * tv
            comps["router_tv"] = tv.detach()

        comps["total"] = total.detach()
        self.last_components = comps
        return total
