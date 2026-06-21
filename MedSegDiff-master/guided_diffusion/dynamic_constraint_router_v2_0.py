"""
v2.0 — Dynamic Constraint Router.

This module implements the **second stage** of the v2.0 upgrade: a
time- and risk-aware router that replaces v1.5's *fixed* ``W_risk`` as
the final per-voxel injection strength for the ControlNet-lite penalty
residual.

v1.5 injection:
    h_x = h_x + W_risk * ZeroConv(penalty_feat)            # W_risk fixed

v2.0 injection:
    h_x = h_x + router_weight_l * ZeroConv(compiled_feat)  # learned, dynamic

Each injection level ``l`` owns one ``DynamicConstraintRouter3D``. The
router decides *where*, *when* and *how strongly* to use the constraint
field, conditioned on:

    * the current main-network feature ``h_x``                (where / state)
    * the compiled penalty-adapter feature at this level      (what)
    * the base risk map ``M_risk_base`` (resized to this level) (prior)
    * the Flow-Matching timestep embedding ``t_emb``          (when)

Time dependence lets the model use a different constraint policy across
the generative trajectory: high-noise steps lean on the global
structural prior, while later steps focus on boundaries and high-risk
regions.

Stable initialisation
---------------------
The router does **not** output the weight directly. It outputs a bounded
*delta* on top of the v1.5 floor weight:

    W_floor      = risk_base + (1 - risk_base) * M_risk_base
    delta        = tanh(conv(...)) * delta_scale
    router_weight = clamp(W_floor + delta, risk_base, 1)

The final 1x1x1 conv is **zero-initialised**, so at initialisation
``delta == 0`` and ``router_weight == W_floor`` — i.e. the router starts
out *exactly* reproducing v1.5's risk-map behaviour and only learns a
correction from there. Combined with the zero-init ZeroConv (whose
output is 0 at init), the whole v2.0 model is functionally identical to
v1.5 / v1.2 at initialisation.
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import conv_nd


class DynamicConstraintRouter3D(nn.Module):
    """Per-level dynamic injection-weight predictor.

    Args:
        feat_channels:   channels of the main-network feature ``h_x``.
        adapter_channels:channels of the compiled penalty-adapter feature
                         injected at this level.
        time_embed_dim:  dimensionality of the timestep embedding fed in
                         (the main UNet's ``time_embed`` output width).
        hidden:          router conv hidden width (default derived).
        risk_base:       floor weight for low-risk regions.
        delta_scale:     max magnitude of the learned correction.
        kernel:          spatial kernel for the first router conv.
        time_dependent:  enable timestep conditioning.
        time_proj_ch:    channels the timestep embedding is projected to.
        dims:            spatial dims (3).
    """

    def __init__(
        self,
        feat_channels,
        adapter_channels,
        time_embed_dim,
        hidden=None,
        risk_base=0.2,
        delta_scale=0.5,
        kernel=3,
        time_dependent=True,
        time_proj_ch=16,
        dims=3,
    ):
        super().__init__()
        self.risk_base = float(risk_base)
        self.delta_scale = float(delta_scale)
        self.time_dependent = bool(time_dependent)
        self.time_proj_ch = int(time_proj_ch) if time_dependent else 0

        if hidden is None:
            hidden = max(16, feat_channels // 2)

        if self.time_dependent:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, self.time_proj_ch),
            )
        else:
            self.time_mlp = None

        in_ch = feat_channels + adapter_channels + 1 + self.time_proj_ch
        pad = kernel // 2
        self.conv1 = conv_nd(dims, in_ch, hidden, kernel, padding=pad)
        self.act = nn.SiLU()
        # Final 1x1x1 conv: zero-initialised so initial delta == 0 and
        # router_weight == v1.5 floor weight (W_risk).
        self.conv2 = conv_nd(dims, hidden, 1, 1)
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, h_x, adapter_feat, risk_map, t_emb=None):
        """Predict the per-voxel injection weight at this level.

        Args:
            h_x:          (B, feat_channels, d, h, w).
            adapter_feat: (B, adapter_channels, d, h, w) — same spatial
                          size as ``h_x``.
            risk_map:     (B, 1, d, h, w) base risk map M_risk_base in
                          [0, 1], already resized to this level.
            t_emb:        (B, time_embed_dim) timestep embedding.

        Returns:
            (B, 1, d, h, w) injection weight in [risk_base, 1].
        """
        parts = [h_x, adapter_feat, risk_map]
        if self.time_dependent and t_emb is not None:
            tp = self.time_mlp(t_emb)  # (B, time_proj_ch)
            tp = tp.view(tp.shape[0], self.time_proj_ch, 1, 1, 1)
            tp = tp.expand(-1, -1, *h_x.shape[2:])
            parts.append(tp)

        z = th.cat(parts, dim=1)
        z = self.act(self.conv1(z))
        delta = th.tanh(self.conv2(z)) * self.delta_scale

        w_floor = self.risk_base + (1.0 - self.risk_base) * risk_map
        weight = (w_floor + delta).clamp(self.risk_base, 1.0)
        return weight


def router_total_variation(w):
    """Mean spatial total-variation of a (B,1,D,H,W) weight map."""
    dx = (w[:, :, 1:, :, :] - w[:, :, :-1, :, :]).abs().mean()
    dy = (w[:, :, :, 1:, :] - w[:, :, :, :-1, :]).abs().mean()
    dz = (w[:, :, :, :, 1:] - w[:, :, :, :, :-1]).abs().mean()
    return dx + dy + dz
