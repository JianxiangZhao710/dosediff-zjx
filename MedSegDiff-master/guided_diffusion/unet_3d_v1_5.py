"""
v1.5 — Risk-aware ControlNet-lite Penalty Adapter (A1 + A3).

Builds directly on v1.4 (``UNetModel_PenaltyAdapter_v1_4``) and keeps
*everything* about the backbone and the adapter unchanged:

  * v1.2 four-stream UNet backbone (X / CT / SYN / DIS), gated fusion,
    X-query ViT bottleneck, decoder — untouched.
  * The lightweight ``PenaltyAdapter3D`` multi-scale encoder — untouched.
  * The zero-initialised 1x1x1 Conv3D injectors into the X stream and the
    bottleneck — untouched (still 0-init, so init forward == v1.4 == v1.2).

The single new idea in v1.5 is a **risk-aware local modulation** applied
*before* the zero-conv residual is added back to the X stream:

    v1.4:   h_x = h_x +            ZeroConv(penalty_feat[level])
    v1.5:   h_x = h_x + W_risk  *  ZeroConv(penalty_feat[level])

where ``W_risk`` is a soft, continuous spatial weight map in
``[risk_base, 1]`` derived from the penalty components themselves. It
emphasises clinically high-risk regions (OAR shells, target-OAR
boundary transition zones, overlap / near-contact conflict regions) and
de-emphasises easy / low-risk regions, so the 3D penalty mainly acts
where it matters.

Risk map construction (``_build_risk_weight``):

  * Penalty channel order defaults to [S_target, A_oar, B_boundary, P_fused].
  * The risk components default to ["A_oar", "B_boundary", "P_fused"].
    A_oar is used directly (clamped to >= 0 as a magnitude); B_boundary
    and P_fused use their absolute value.
  * Each selected channel is soft-normalised per-sample (max-abs by
    default, preserving zero), the normalised maps are averaged and
    clamped to [0, 1] to give ``M_risk``.
  * ``W_risk = risk_base + (1 - risk_base) * M_risk`` keeps a weak global
    guidance (default risk_base=0.2) everywhere while boosting high-risk
    voxels.
  * Degradation: if none of the requested risk channels are present
    (e.g. single-channel penalty), the map falls back to abs(P_fused);
    if even that is unavailable, ``W_risk`` becomes all-ones, i.e. the
    model behaves exactly like v1.4.

Because the zero-convs remain 0-initialised, multiplying their (zero)
output by ``W_risk`` is still zero, so the model at initialisation is
*functionally identical* to v1.4 / v1.2. Penalty dropout (training only)
is preserved; the risk weight is computed from the *post-dropout*
penalty so dropped samples consistently collapse to ``W_risk = risk_base``.
"""
import torch as th
import torch.nn.functional as F

from .unet_3d_v1_4 import UNetModel_PenaltyAdapter_v1_4
from .nn import timestep_embedding, checkpoint


DEFAULT_PENALTY_CHANNEL_NAMES = ("S_target", "A_oar", "B_boundary", "P_fused")
DEFAULT_RISK_USE_CHANNELS = ("A_oar", "B_boundary", "P_fused")
# Channels whose raw sign is not meaningful for "risk magnitude" -> use abs().
_ABS_RISK_CHANNELS = frozenset({"B_boundary", "P_fused"})


class UNetModel_RiskAwarePenaltyAdapter_v1_5(UNetModel_PenaltyAdapter_v1_4):
    """v1.4 + soft risk-aware local modulation of the zero-conv residual.

    Additional kwargs (everything else matches v1.4):
        use_risk_aware_injection: bool, master switch. When False this
                                  class behaves byte-for-byte like v1.4.
        risk_base:                float in [0, 1]. Floor weight applied to
                                  non-risk regions (default 0.2).
        risk_use_channels:        tuple[str], penalty component names used
                                  to build M_risk (default
                                  ["A_oar", "B_boundary", "P_fused"]).
        risk_soft:                bool. True -> continuous soft mask
                                  (default). False -> hard 0/1 threshold
                                  at 0.5 (not recommended; provided as an
                                  ablation knob).
        penalty_channel_names:    tuple[str], channel ordering of the
                                  penalty input. Defaults to
                                  [S_target, A_oar, B_boundary, P_fused]
                                  truncated to ``penalty_channels`` (or
                                  ["P_fused"] when penalty_channels == 1).
    """

    def __init__(
        self,
        *args,
        use_risk_aware_injection=True,
        risk_base=0.2,
        risk_use_channels=DEFAULT_RISK_USE_CHANNELS,
        risk_soft=True,
        penalty_channel_names=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.use_risk_aware_injection = bool(use_risk_aware_injection)
        self.risk_base = float(risk_base)
        self.risk_soft = bool(risk_soft)
        self.risk_use_channels = tuple(risk_use_channels)

        # Resolve penalty channel naming.
        if penalty_channel_names is not None:
            names = tuple(penalty_channel_names)
        elif self.penalty_channels == 1:
            names = ("P_fused",)
        else:
            base = DEFAULT_PENALTY_CHANNEL_NAMES
            if self.penalty_channels <= len(base):
                names = base[: self.penalty_channels]
            else:
                names = base + tuple(
                    f"extra_{i}" for i in range(self.penalty_channels - len(base))
                )
        self.penalty_channel_names = names
        self._risk_name_to_idx = {n: i for i, n in enumerate(names)}

    # ------------------------------------------------------------------ #
    # Risk weight construction
    # ------------------------------------------------------------------ #
    def _normalize_soft(self, ch):
        """Per-sample max-abs normalisation of a non-negative channel.

        ch: (B, 1, D, H, W) — assumed already made non-negative by caller.
        Returns a map in [0, 1] (preserving zero). When ``risk_soft`` is
        False, the continuous map is hard-thresholded at 0.5.
        """
        b = ch.shape[0]
        amax = ch.reshape(b, -1).amax(dim=1).clamp_min(1e-6).view(b, 1, 1, 1, 1)
        out = (ch / amax).clamp(0.0, 1.0)
        if not self.risk_soft:
            out = (out > 0.5).to(out.dtype)
        return out

    def _build_risk_weight(self, penalty):
        """Construct W_risk in [risk_base, 1] from the penalty components.

        penalty: (B, P, D, H, W) — the (post-dropout) penalty input.
        Returns: (B, 1, D, H, W) weight map, or ``None`` when risk-aware
                 injection is disabled (caller then falls back to v1.4).
        """
        if not self.use_risk_aware_injection:
            return None

        P = penalty.shape[1]
        comps = []
        for name in self.risk_use_channels:
            idx = self._risk_name_to_idx.get(name)
            if idx is None or idx >= P:
                continue
            ch = penalty[:, idx:idx + 1, ...]
            if name in _ABS_RISK_CHANNELS:
                ch = ch.abs()
            else:
                # A_oar (and any other directional channel): treat as a
                # non-negative penalty magnitude.
                ch = ch.clamp_min(0.0)
            comps.append(self._normalize_soft(ch))

        if len(comps) == 0:
            # Degrade gracefully: abs(P_fused) if present, else single
            # channel, else fall back to v1.4 (W_risk == 1 everywhere).
            idx = self._risk_name_to_idx.get("P_fused")
            if idx is None and P == 1:
                idx = 0
            if idx is not None and idx < P:
                comps.append(self._normalize_soft(penalty[:, idx:idx + 1, ...].abs()))
            else:
                return th.ones(
                    penalty.shape[0], 1, *penalty.shape[2:],
                    device=penalty.device, dtype=penalty.dtype,
                )

        m_risk = th.stack(comps, dim=0).mean(dim=0).clamp(0.0, 1.0)  # (B,1,D,H,W)
        w_risk = self.risk_base + (1.0 - self.risk_base) * m_risk
        return w_risk

    @staticmethod
    def _resize_risk(w_risk, ref):
        """Resize ``w_risk`` (B,1,d,h,w) to the spatial size of ``ref``."""
        if w_risk.shape[2:] == ref.shape[2:]:
            return w_risk
        return F.interpolate(
            w_risk, size=ref.shape[2:], mode='trilinear', align_corners=False
        )

    # ------------------------------------------------------------------ #
    # Forward (mirrors v1.4, with risk-aware modulation of zero-conv)
    # ------------------------------------------------------------------ #
    def forward(self, x, timesteps, ct, syn, dis, penalty, y=None):
        # ---- Optional penalty dropout (training only) ----
        if penalty is None:
            penalty = th.zeros(
                x.shape[0], self.penalty_channels,
                x.shape[2], x.shape[3], x.shape[4],
                device=x.device, dtype=x.dtype,
            )
        elif self.training and self.use_penalty_dropout and self.penalty_dropout_p > 0.0:
            keep_mask = (
                th.rand(x.shape[0], device=x.device) > self.penalty_dropout_p
            ).to(x.dtype)
            keep_mask = keep_mask.view(-1, 1, 1, 1, 1)
            penalty = penalty * keep_mask

        # ---- v1.5: soft risk weight from the (post-dropout) penalty ----
        w_risk = self._build_risk_weight(penalty)  # (B,1,D,H,W) or None
        if w_risk is not None:
            w_risk = w_risk.type(self.dtype)

        # ---- Penalty Adapter: multi-scale features ----
        penalty_feats = self.penalty_adapter(penalty.type(self.dtype))

        # ---- Standard v1.2 forward, with risk-modulated penalty injection ----
        hs = []
        emb_in = timestep_embedding(timesteps, self.model_channels).type(self.dtype)
        emb = self.time_embed(emb_in)

        h = x.type(self.dtype)
        h_ct = ct.type(self.dtype)
        h_syn = syn.type(self.dtype)
        h_dis = dis.type(self.dtype)

        last_h_x = None
        last_h_ct = None
        last_h_syn = None
        last_h_dis = None

        for i, module in enumerate(self.input_blocks):
            h_x = module(h, emb)
            h_ct = self.input_blocks_CT[i](h_ct, emb)
            h_syn = self.input_blocks_SYN[i](h_syn, emb)
            h_dis = self.input_blocks_DIS[i](h_dis, emb)

            # ---- v1.5: risk-aware zero-conv penalty residual on X ----
            level = self._input_block_levels[i]
            zero_conv_i = self.penalty_zero_convs[i]
            residual = zero_conv_i(penalty_feats[level])
            if w_risk is not None:
                residual = residual * self._resize_risk(w_risk, residual)
            h_x = h_x + residual

            cond_proj_i = self.cond_proj[i]
            gate_proj_i = self.gate_proj[i]

            def _gated_fuse(h_x_, h_ct_, h_syn_, h_dis_, _cp=cond_proj_i, _gp=gate_proj_i):
                h_cond_ = _cp(th.cat([h_ct_, h_syn_, h_dis_], dim=1))
                gate_ = th.sigmoid(_gp(th.cat([h_x_, h_cond_], dim=1)))
                return h_x_ + gate_ * h_cond_

            fuse_params = (
                list(cond_proj_i.parameters()) + list(gate_proj_i.parameters())
            )
            h = checkpoint(
                _gated_fuse,
                (h_x, h_ct, h_syn, h_dis),
                fuse_params,
                self.use_checkpoint,
            )
            hs.append(h)

            last_h_x = h_x
            last_h_ct = h_ct
            last_h_syn = h_syn
            last_h_dis = h_dis

        # ----- X-query ViT cross-attention at bottleneck (4-condition) -----
        vit_params = (
            list(self.vit_cond_proj.parameters()) + list(self.fusion.parameters())
        )
        h = checkpoint(
            self._vit_bottleneck,
            (last_h_x, last_h_ct, last_h_syn, last_h_dis),
            vit_params,
            self.use_checkpoint,
        )

        # ---- v1.5: risk-aware zero-conv penalty residual at bottleneck ----
        residual_mid = self.penalty_zero_conv_middle(penalty_feats[-1])
        if w_risk is not None:
            residual_mid = residual_mid * self._resize_risk(w_risk, residual_mid)
        h = h + residual_mid

        # ----- Middle + Decoder (unchanged from v1.2) -----
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        return self.out(h)
