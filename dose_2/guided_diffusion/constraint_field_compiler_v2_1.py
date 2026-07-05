"""
v2.1 — Constraint Field Compiler (architecture-only fix for v2.0 channel collapse).

Changes vs ``ConstraintFieldCompiler3D`` (v2.0), **no new loss terms**:

  A) Break the P_fused shortcut in ``C_fused``
       v2.0:  C_fused = α·C_pos − β·C_neg + γ·C_boundary + P_fused
       v2.1:  C_fused = α·C_pos − β·C_neg + γ·C_boundary
     ``P_fused`` remains in the penalty *basis* (stem input) but is no longer
     added directly to the fused output, forcing the three compiled channels
     to carry coarse-prior information through the learned fusion weights.

  B) Softplus instead of ReLU for ``C_pos`` / ``C_neg``
       v2.0:  C_pos = ReLU(f_pos + S_target)   → f can cancel S → channel dies
       v2.1:  C_pos = softplus(f_pos + S_target) → always > 0, no hard zeroing

``C_boundary`` is unchanged: ``f_bnd + B_boundary`` (signed boundary field).

All other modules (RelationEncoder, FiLM blocks, weight_head) are inherited
unchanged from v2.0.
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
    "ConstraintFieldCompiler3D_v21",
    "DEFAULT_BASIS_CHANNEL_NAMES",
    "DEFAULT_COMPILED_CHANNEL_NAMES",
    "RelationEncoder",
    "_FiLMResBlock",
)


class ConstraintFieldCompiler3D_v21(ConstraintFieldCompiler3D):
    """v2.1 compiler: no P_fused bypass + softplus pos/neg (architecture only)."""

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

        # B: softplus — channels cannot be hard-killed by negative corrections
        c_pos = F.softplus(f_pos + s)
        c_neg = F.softplus(f_neg + a)
        c_boundary = f_bnd + b
        # A: no direct P_fused bypass in C_fused
        c_fused = alpha * c_pos - beta * c_neg + gamma * c_boundary

        outs = [c_pos, c_neg, c_boundary, c_fused]
        if self.n_extra > 0:
            outs.append(field[:, 3:3 + self.n_extra, ...])
        compiled = th.cat(outs, dim=1)

        if return_weights:
            return compiled, {"alpha": alpha, "beta": beta, "gamma": gamma}
        return compiled
