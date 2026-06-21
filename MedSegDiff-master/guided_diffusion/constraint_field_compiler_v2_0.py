"""
v2.0 — Learnable Relation-to-Field Constraint Compiler.

This module implements the **first stage** of the v2.0
"Dynamic Relation-to-Field Constraint Router" upgrade:

    penalty_basis  ->  RelationEncoder (global relation embedding)
                   ->  ConstraintFieldCompiler3D (FiLM-modulated 3D conv)
                   ->  compiled_constraint_field   [C_pos, C_neg, C_boundary, C_fused]

Motivation
----------
In v1.5 the multi-channel penalty basis ``[S_target, A_oar, B_boundary,
P_fused]`` was treated as a *fixed* hand-crafted constraint and injected
through a fixed risk weight ``W_risk``. The fused channel followed a
fixed manual formula ``P = alpha*S - beta*A + gamma*B``.

In v2.0 the four channels are no longer the final constraint — they are
**interpretable basis fields**. A lightweight learnable compiler refines
them into a model-friendly continuous constraint representation whose
fusion weights ``alpha(x), beta(x), gamma(x)`` are *spatially varying*
and *learned*, instead of fixed scalars.

Outputs (``compiler_out_channels`` defaults to 4):

    C_pos       : positive / source field (around the target)
    C_neg       : negative / avoidance field (around OARs)
    C_boundary  : signed boundary / transition field
    C_fused     : spatially-adaptive fusion
                  C_fused = alpha*C_pos - beta*C_neg + gamma*C_boundary
                            + P_fused_coarse_prior

Design constraints
------------------
  * Only Conv3D / lightweight ResBlock are used (no heavy attention),
    matching the v1.4 ``PenaltyAdapter3D`` philosophy.
  * **No label leakage**: the compiler only consumes condition tensors
    (penalty basis, DIS masks, optionally CT/SYN). Ground-truth dose is
    never used to build any prior.
  * The compiled field keeps the *same spatial resolution* as the input
    penalty so it can be fed straight into the existing v1.4/v1.5
    ``PenaltyAdapter3D`` multi-scale encoder.

The relation embedding is derived from current tensors only (DIS mask
volumes + penalty-channel statistics). When an explicit Target-OAR
relation graph becomes available later, ``RelationEncoder`` can be
swapped for a real graph encoder without touching the compiler API.
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import conv_nd, normalization


# Canonical basis channel ordering shared with the v1.5 dataset / UNet.
DEFAULT_BASIS_CHANNEL_NAMES = ("S_target", "A_oar", "B_boundary", "P_fused")
DEFAULT_COMPILED_CHANNEL_NAMES = ("C_pos", "C_neg", "C_boundary", "C_fused")


class RelationEncoder(nn.Module):
    """Sample-level relation embedding from DIS masks + penalty basis.

    First-version (no graph neural network): we derive a compact set of
    global statistics that summarise the target/OAR layout and the
    penalty-channel intensities, then map them through a small MLP to a
    ``relation_dim`` embedding used to FiLM-modulate the compiler.

    Statistics (per sample):
        * per-DIS-channel volume fraction (mean occupancy)            -> dis_channels
        * per-penalty-channel [mean, abs-mean, abs-max, std]          -> 4 * penalty_channels
        * PTV volume fraction, OAR volume fraction, PTV-OAR overlap    -> 3

    The PTV/OAR split assumes the OpenKBP DIS convention
    (``dis[:, 0:3]`` = PTVs, ``dis[:, 3:10]`` = OARs); when the channel
    count differs the split degrades gracefully.
    """

    def __init__(
        self,
        dis_channels=11,
        penalty_channels=4,
        relation_dim=128,
        hidden_dim=128,
        ptv_slice=(0, 3),
        oar_slice=(3, 10),
    ):
        super().__init__()
        self.dis_channels = dis_channels
        self.penalty_channels = penalty_channels
        self.relation_dim = relation_dim
        self.ptv_slice = ptv_slice
        self.oar_slice = oar_slice

        self.input_dim = dis_channels + 4 * penalty_channels + 3
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, relation_dim),
            nn.SiLU(),
        )

    @th.no_grad()
    def _safe_slice(self, dis, lo, hi):
        c = dis.shape[1]
        lo = max(0, min(lo, c))
        hi = max(lo, min(hi, c))
        if hi <= lo:
            return None
        return dis[:, lo:hi, ...]

    def extract_stats(self, dis, penalty):
        """Return a (B, input_dim) statistics tensor (differentiable)."""
        b = penalty.shape[0]
        spatial = (2, 3, 4)

        dis_vol = dis.mean(dim=spatial)  # (B, dis_channels)

        pen_mean = penalty.mean(dim=spatial)
        pen_absmean = penalty.abs().mean(dim=spatial)
        pen_absmax = penalty.abs().amax(dim=spatial)
        pen_std = penalty.var(dim=spatial, unbiased=False).clamp_min(0.0).sqrt()
        pen_stats = th.cat([pen_mean, pen_absmean, pen_absmax, pen_std], dim=1)

        ptv = self._safe_slice(dis, *self.ptv_slice)
        oar = self._safe_slice(dis, *self.oar_slice)
        if ptv is not None:
            ptv_union = ptv.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        else:
            ptv_union = th.zeros(b, 1, *dis.shape[2:], device=dis.device, dtype=dis.dtype)
        if oar is not None:
            oar_union = oar.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        else:
            oar_union = th.zeros_like(ptv_union)

        ptv_frac = ptv_union.mean(dim=spatial)
        oar_frac = oar_union.mean(dim=spatial)
        overlap = (ptv_union * oar_union).mean(dim=spatial)  # near-contact / conflict proxy

        stats = th.cat([dis_vol, pen_stats, ptv_frac, oar_frac, overlap], dim=1)
        return stats

    def forward(self, dis, penalty):
        stats = self.extract_stats(dis, penalty)
        return self.mlp(stats)  # (B, relation_dim)


class _FiLMResBlock(nn.Module):
    """Lightweight 3D residual block with optional FiLM modulation.

    Channel-preserving. When ``relation_dim > 0`` the block accepts a
    relation embedding and applies feature-wise linear modulation
    (scale & shift) to the intermediate features.
    """

    def __init__(self, channels, relation_dim=0, dims=3, dropout=0.0):
        super().__init__()
        self.norm1 = normalization(channels)
        self.conv1 = conv_nd(dims, channels, channels, 3, padding=1)
        self.norm2 = normalization(channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = conv_nd(dims, channels, channels, 3, padding=1)
        self.relation_dim = relation_dim
        if relation_dim > 0:
            self.film = nn.Linear(relation_dim, 2 * channels)
            # Start as identity modulation (scale=1, shift=0).
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        else:
            self.film = None

    def forward(self, x, rel=None):
        h = self.conv1(F.silu(self.norm1(x)))
        if self.film is not None and rel is not None:
            gamma, beta = self.film(rel).chunk(2, dim=1)
            gamma = gamma.view(gamma.shape[0], -1, 1, 1, 1)
            beta = beta.view(beta.shape[0], -1, 1, 1, 1)
            h = h * (1.0 + gamma) + beta
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return x + h


class ConstraintFieldCompiler3D(nn.Module):
    """Compile a penalty basis into a learnable constraint field.

    Args:
        penalty_channels:   number of basis input channels (default 4).
        out_channels:       number of compiled output channels
                            (default 4 -> [C_pos, C_neg, C_boundary,
                            C_fused]). Only 4 is fully interpretable; the
                            class supports >4 by padding extra learned
                            channels after C_fused.
        width:              base conv width (must be a multiple of 32 for
                            GroupNorm32; defaults to 32).
        num_res_blocks:     number of FiLM res blocks (default 2).
        basis_channel_names:tuple naming the basis channels so the
                            compiler can locate S_target / A_oar /
                            B_boundary / P_fused regardless of order.
        use_relation:       enable the RelationEncoder + FiLM modulation.
        dis_channels:       DIS channel count (for the RelationEncoder).
        relation_dim:       relation embedding size.
        use_cond_context:   if True, concatenate CT/SYN to the compiler
                            input for extra context (no GT dose).
        cond_channels:      number of extra context channels appended
                            when ``use_cond_context`` (e.g. CT + SYN = 2).
        dropout:            dropout inside res blocks.
        dims:               spatial dims (3).
    """

    def __init__(
        self,
        penalty_channels=4,
        out_channels=4,
        width=32,
        num_res_blocks=2,
        basis_channel_names=DEFAULT_BASIS_CHANNEL_NAMES,
        use_relation=True,
        dis_channels=11,
        relation_dim=128,
        use_cond_context=False,
        cond_channels=0,
        dropout=0.0,
        dims=3,
    ):
        super().__init__()
        self.penalty_channels = penalty_channels
        self.out_channels = max(4, int(out_channels))
        self.width = width
        self.use_relation = bool(use_relation)
        self.use_cond_context = bool(use_cond_context)
        self.cond_channels = int(cond_channels) if use_cond_context else 0
        self.basis_channel_names = tuple(basis_channel_names)
        self._name_to_idx = {n: i for i, n in enumerate(self.basis_channel_names)}

        self.relation_dim = relation_dim if self.use_relation else 0
        if self.use_relation:
            self.relation_encoder = RelationEncoder(
                dis_channels=dis_channels,
                penalty_channels=penalty_channels,
                relation_dim=relation_dim,
            )
        else:
            self.relation_encoder = None

        in_ch = penalty_channels + self.cond_channels
        self.stem = conv_nd(dims, in_ch, width, 3, padding=1)
        self.blocks = nn.ModuleList([
            _FiLMResBlock(width, relation_dim=self.relation_dim, dims=dims, dropout=dropout)
            for _ in range(num_res_blocks)
        ])

        # Head A: refined field deltas for [C_pos, C_neg, C_boundary]
        # plus any extra learned channels beyond the 4 canonical ones.
        self.n_extra = self.out_channels - 4
        self.field_head = conv_nd(dims, width, 3 + max(0, self.n_extra), 3, padding=1)
        # Head B: spatially-varying fusion weights alpha, beta, gamma.
        self.weight_head = conv_nd(dims, width, 3, 3, padding=1)

    def _basis(self, penalty, name):
        idx = self._name_to_idx.get(name)
        if idx is None or idx >= penalty.shape[1]:
            return th.zeros(
                penalty.shape[0], 1, *penalty.shape[2:],
                device=penalty.device, dtype=penalty.dtype,
            )
        return penalty[:, idx:idx + 1, ...]

    def forward(self, penalty, dis=None, cond=None, return_weights=False):
        """Compile the penalty basis into a constraint field.

        Args:
            penalty: (B, penalty_channels, D, H, W) basis input.
            dis:     (B, dis_channels, D, H, W) used by RelationEncoder.
            cond:    optional (B, cond_channels, D, H, W) extra context.
            return_weights: also return (alpha, beta, gamma) maps.

        Returns:
            compiled: (B, out_channels, D, H, W).
            (optional) dict with 'alpha','beta','gamma' if requested.
        """
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
        p = self._basis(penalty, "P_fused")  # coarse / synthetic prior

        c_pos = F.relu(f_pos + s)
        c_neg = F.relu(f_neg + a)
        c_boundary = f_bnd + b
        c_fused = alpha * c_pos - beta * c_neg + gamma * c_boundary + p

        outs = [c_pos, c_neg, c_boundary, c_fused]
        if self.n_extra > 0:
            outs.append(field[:, 3:3 + self.n_extra, ...])
        compiled = th.cat(outs, dim=1)

        if return_weights:
            return compiled, {"alpha": alpha, "beta": beta, "gamma": gamma}
        return compiled
