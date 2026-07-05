"""
v2.2 — Constraint Field Compiler (basis gating, no additive cancellation).

Changes vs v2.1 (and v2.0 shortcut issue):

  * **Region gating** instead of additive residual before activation:
        v2.1:  C_pos = softplus(f_pos + S_target)   → f can cancel S in PTV
        v2.2:  C_pos = softplus(f_pos) * S_target    → active only where S>0

  * **Boundary** bounded and band-local:
        v2.1:  C_boundary = f_bnd + B_boundary
        v2.2:  C_boundary = tanh(f_bnd) * (|B_boundary| + eps)

  * **C_fused** still has **no** direct P_fused bypass (same as v2.1).
    Optional ablation ``gate_prior`` adds a tiny learnable coarse prior:
        C_fused += softplus(log_eta) * P_fused

Gate modes (``gate_mode``):
    ``gate``       : default scheme A (pure multiplicative gating)
    ``gate1p``     : scheme B — multiply by (1 + basis) for denser gradients
    ``gate_prior`` : scheme C — weak learnable P_fused term on C_fused only
"""
import torch as th
import torch.nn.functional as F

from .constraint_field_compiler_v2_0 import (
    ConstraintFieldCompiler3D,
    DEFAULT_BASIS_CHANNEL_NAMES,
    DEFAULT_COMPILED_CHANNEL_NAMES,
    RelationEncoder,
    _FiLMResBlock,
)

__all__ = (
    "ConstraintFieldCompiler3D_v22",
    "DEFAULT_BASIS_CHANNEL_NAMES",
    "DEFAULT_COMPILED_CHANNEL_NAMES",
    "RelationEncoder",
    "_FiLMResBlock",
    "V22_GATE_MODES",
)

V22_GATE_MODES = ("gate", "gate1p", "gate_prior")


class ConstraintFieldCompiler3D_v22(ConstraintFieldCompiler3D):
    """v2.2 compiler: basis-gated pos/neg + bounded boundary (architecture only)."""

    def __init__(
        self,
        *args,
        gate_mode="gate",
        boundary_eps=1e-3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if gate_mode not in V22_GATE_MODES:
            raise ValueError(f"gate_mode must be one of {V22_GATE_MODES}, got {gate_mode!r}")
        self.gate_mode = gate_mode
        self.boundary_eps = float(boundary_eps)
        # scheme C: init softplus(log_eta) ≈ 0.007
        self.log_fused_prior_scale = th.nn.Parameter(th.tensor(-5.0))

    def _pos_gate(self, f_pos, s):
        act = F.softplus(f_pos)
        if self.gate_mode == "gate1p":
            return act * (1.0 + s)
        return act * s

    def _neg_gate(self, f_neg, a):
        act = F.softplus(f_neg)
        if self.gate_mode == "gate1p":
            return act * (1.0 + a)
        return act * a

    def forward(self, penalty, dis=None, cond=None, return_weights=False):
        rel = None
        if self.use_relation and dis is not None:
            rel = self.relation_encoder(dis, penalty)

        x = penalty
        if self.use_cond_context and cond is not None:
            x = th.cat([penalty, cond], dim=1)

        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h, rel)

        field = self.field_head(h)
        f_pos = field[:, 0:1, ...]
        f_neg = field[:, 1:2, ...]
        f_bnd = field[:, 2:3, ...]

        w = F.softplus(self.weight_head(h))
        alpha = w[:, 0:1, ...]
        beta = w[:, 1:2, ...]
        gamma = w[:, 2:3, ...]

        s = self._basis(penalty, "S_target").clamp_min(0.0)
        a = self._basis(penalty, "A_oar").clamp_min(0.0)
        b = self._basis(penalty, "B_boundary")

        c_pos = self._pos_gate(f_pos, s)
        c_neg = self._neg_gate(f_neg, a)
        c_boundary = th.tanh(f_bnd) * (b.abs() + self.boundary_eps)
        c_fused = alpha * c_pos - beta * c_neg + gamma * c_boundary

        if self.gate_mode == "gate_prior":
            p = self._basis(penalty, "P_fused").clamp_min(0.0)
            eta = F.softplus(self.log_fused_prior_scale)
            c_fused = c_fused + eta * p

        outs = [c_pos, c_neg, c_boundary, c_fused]
        if self.n_extra > 0:
            outs.append(field[:, 3:3 + self.n_extra, ...])
        compiled = th.cat(outs, dim=1)

        if return_weights:
            extra = {"alpha": alpha, "beta": beta, "gamma": gamma}
            if self.gate_mode == "gate_prior":
                extra["fused_prior_eta"] = eta.expand_as(alpha)
            return compiled, extra
        return compiled
