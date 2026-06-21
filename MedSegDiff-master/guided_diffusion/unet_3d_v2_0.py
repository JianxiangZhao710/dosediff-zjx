"""
v2.0 — Dynamic Relation-to-Field Constraint Router (UNet).

Builds directly on v1.5 (``UNetModel_RiskAwarePenaltyAdapter_v1_5``) and
upgrades the *fixed* "penalty field + fixed risk mask" injection into a
"learnable constraint-field compiler + dynamic time/risk-aware router".

Data flow (full v2.0):

    penalty_basis [S_target, A_oar, B_boundary, P_fused]
        -> ConstraintFieldCompiler3D  (+ RelationEncoder / FiLM)
        -> compiled_constraint_field  [C_pos, C_neg, C_boundary, C_fused]
        -> PenaltyAdapter3D (v1.4/v1.5, reused)  -> multi-scale feats
        -> ZeroConv (0-init, reused)             -> residual
        -> DynamicConstraintRouter3D modulation  -> router_weight * residual
        -> injected into the X stream of the v1.2 backbone

v1.5 injection:  h_x = h_x + W_risk          * ZeroConv(penalty_feat)
v2.0 injection:  h_x = h_x + router_weight_l * ZeroConv(compiled_feat_l)

What is preserved from v1.5 / v1.4 / v1.2
----------------------------------------
  * The v1.2 four-stream UNet backbone (X / CT / SYN / DIS), gated
    fusion, X-query ViT bottleneck and decoder — untouched.
  * The v1.4/v1.5 ``PenaltyAdapter3D`` multi-scale encoder and the
    zero-initialised 1x1x1 ZeroConv injectors — reused (and still
    0-init, so the initial forward adds exactly 0).
  * The v1.5 base risk map (``_build_risk_weight``) — kept, but it is now
    only a **prior input** to the router, not the final injection weight.
  * Penalty dropout, ``risk_base``, ``risk_use_channels``, freeze/unfreeze
    warmup helpers.

What is new in v2.0
-------------------
  * ``ConstraintFieldCompiler3D`` (+ ``RelationEncoder``): compiles the
    penalty basis into a learnable continuous constraint field.
  * ``DynamicConstraintRouter3D`` per injection level: predicts a
    time-/risk-aware per-voxel injection weight, initialised to exactly
    reproduce v1.5's risk-map weight (zero-init delta).

Ablation modes (``mode``)
-------------------------
    'v1_5'         : no compiler, no router -> fixed W_risk (== v1.5).
    'compiler_only': learnable compiled field, fixed W_risk injection.
    'router_only'  : raw penalty basis, dynamic router injection.
    'full_v2_0'    : compiler + dynamic router (default).

Checkpoint compatibility
------------------------
Because v2.0 inherits the v1.5 module names for the backbone, penalty
adapter and zero-convs, a v1.5 checkpoint loads cleanly with
``strict=False``: the backbone / adapter / zero-conv weights are reused,
while the new compiler / relation encoder / routers are freshly
initialised. The default ``compiler_out_channels == penalty_channels``
(both 4) keeps the adapter input channel count unchanged so its first
conv weights are reused as well.
"""
import torch as th
import torch.nn.functional as F

from .unet_3d_v1_5 import UNetModel_RiskAwarePenaltyAdapter_v1_5
from .nn import timestep_embedding, checkpoint
from .constraint_field_compiler_v2_0 import ConstraintFieldCompiler3D
from .dynamic_constraint_router_v2_0 import (
    DynamicConstraintRouter3D,
    router_total_variation,
)
from .penalty_adapter_v1_4 import PenaltyAdapter3D
from .nn import conv_nd
import torch.nn as nn


V2_MODES = ("v1_5", "compiler_only", "router_only", "full_v2_0")


def resolve_v2_mode(mode=None, use_constraint_compiler=None, use_dynamic_router=None):
    """Resolve an ablation ``mode`` string and the (compiler, router) flags.

    If ``mode`` is given it wins; otherwise the booleans select the mode.
    Returns ``(mode, use_compiler, use_router)``.
    """
    if mode is not None:
        if mode not in V2_MODES:
            raise ValueError(f"mode must be one of {V2_MODES}, got {mode!r}")
        use_compiler = mode in ("compiler_only", "full_v2_0")
        use_router = mode in ("router_only", "full_v2_0")
        return mode, use_compiler, use_router

    use_compiler = bool(use_constraint_compiler)
    use_router = bool(use_dynamic_router)
    if use_compiler and use_router:
        mode = "full_v2_0"
    elif use_compiler:
        mode = "compiler_only"
    elif use_router:
        mode = "router_only"
    else:
        mode = "v1_5"
    return mode, use_compiler, use_router


class UNetModel_DynamicConstraintRouter_v2_0(UNetModel_RiskAwarePenaltyAdapter_v1_5):
    """v1.5 + learnable constraint compiler + dynamic constraint router.

    Additional kwargs (everything else matches v1.5):
        mode:                      ablation mode (see ``V2_MODES``). When
                                   provided it overrides the boolean flags.
        use_constraint_compiler:   enable ConstraintFieldCompiler3D.
        use_dynamic_router:        enable DynamicConstraintRouter3D.
        use_relation_embedding:    enable the RelationEncoder inside the
                                   compiler.
        compiler_out_channels:     compiled-field channel count (default 4
                                   -> reuses the v1.5 adapter input).
        compiler_width:            compiler conv width (default
                                   ``model_channels``).
        compiler_num_res_blocks:   FiLM res blocks in the compiler.
        relation_dim:              relation embedding size.
        compiler_use_cond_context: feed CT/SYN as extra compiler context.
        router_time_dependent:     timestep-condition the router.
        router_delta_scale:        max router correction over the floor.
        router_hidden:             router conv hidden width.
    """

    def __init__(
        self,
        *args,
        mode=None,
        use_constraint_compiler=True,
        use_dynamic_router=True,
        use_relation_embedding=True,
        compiler_out_channels=4,
        compiler_width=None,
        compiler_num_res_blocks=2,
        relation_dim=128,
        compiler_use_cond_context=False,
        router_time_dependent=True,
        router_delta_scale=0.5,
        router_hidden=None,
        **kwargs,
    ):
        # dis_channels needed by the relation encoder; v1.5/v1.2 consume it
        # positionally/keyword via kwargs — capture without disturbing super.
        dis_channels = kwargs.get("dis_channels", 11)

        super().__init__(*args, **kwargs)

        mode, use_compiler, use_router = resolve_v2_mode(
            mode, use_constraint_compiler, use_dynamic_router
        )
        self.v2_mode = mode
        self.use_compiler = use_compiler
        self.use_router = use_router
        self.use_relation_embedding = bool(use_relation_embedding)
        self.compiler_out_channels = int(compiler_out_channels)
        self.router_time_dependent = bool(router_time_dependent)
        self._v2_dis_channels = dis_channels

        time_embed_dim = self.model_channels * 4

        # ---------------- Constraint Field Compiler ----------------
        if self.use_compiler:
            width = compiler_width if compiler_width is not None else self.model_channels
            cond_channels = 2 if compiler_use_cond_context else 0  # CT + SYN
            self.constraint_compiler = ConstraintFieldCompiler3D(
                penalty_channels=self.penalty_channels,
                out_channels=self.compiler_out_channels,
                width=width,
                num_res_blocks=compiler_num_res_blocks,
                basis_channel_names=self.penalty_channel_names,
                use_relation=self.use_relation_embedding,
                dis_channels=dis_channels,
                relation_dim=relation_dim,
                use_cond_context=compiler_use_cond_context,
                cond_channels=cond_channels,
                dims=self.dims if hasattr(self, "dims") else 3,
            )
            self.compiler_use_cond_context = bool(compiler_use_cond_context)

            # If the compiled-field channel count differs from the basis,
            # rebuild the penalty adapter + zero-convs so shapes line up.
            # (Default 4 == 4 -> reuse the inherited v1.5 modules / weights.)
            if self.compiler_out_channels != self.penalty_channels:
                self._rebuild_penalty_branch(self.compiler_out_channels)
        else:
            self.constraint_compiler = None
            self.compiler_use_cond_context = False

        # ---------------- Dynamic Constraint Router ----------------
        if self.use_router:
            dims = self.dims if hasattr(self, "dims") else 3
            self.routers = nn.ModuleList()
            for i in range(len(self.input_blocks)):
                level = self._input_block_levels[i]
                feat_ch = self._input_block_chans[i]
                adapter_ch = self._penalty_scale_channels[level]
                self.routers.append(
                    DynamicConstraintRouter3D(
                        feat_channels=feat_ch,
                        adapter_channels=adapter_ch,
                        time_embed_dim=time_embed_dim,
                        hidden=router_hidden,
                        risk_base=self.risk_base,
                        delta_scale=router_delta_scale,
                        time_dependent=self.router_time_dependent,
                        dims=dims,
                    )
                )
            bottleneck_ch = self._input_block_chans[-1]
            deepest_adapter_ch = self._penalty_scale_channels[-1]
            self.router_middle = DynamicConstraintRouter3D(
                feat_channels=bottleneck_ch,
                adapter_channels=deepest_adapter_ch,
                time_embed_dim=time_embed_dim,
                hidden=router_hidden,
                risk_base=self.risk_base,
                delta_scale=router_delta_scale,
                time_dependent=self.router_time_dependent,
                dims=dims,
            )
        else:
            self.routers = None
            self.router_middle = None

        # Auxiliary / regularisation state (populated each forward).
        self.collect_aux = False
        self._aux = {}
        self._router_mean_reg = th.zeros(())
        self._router_tv_reg = th.zeros(())
        self._last_m_risk = None

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def _rebuild_penalty_branch(self, new_penalty_channels):
        """Rebuild PenaltyAdapter3D + zero-convs for a new input channel
        count (used only when ``compiler_out_channels != penalty_channels``).
        Mirrors the v1.4 construction exactly so all downstream addressing
        (``_penalty_scale_channels`` etc.) stays valid.
        """
        dims = self.dims if hasattr(self, "dims") else 3
        self.penalty_adapter = PenaltyAdapter3D(
            penalty_channels=new_penalty_channels,
            model_channels=self.penalty_model_channels,
            channel_mult=self._penalty_channel_mult,
            num_res_blocks=self.penalty_adapter.num_res_blocks,
            dims=dims,
        )
        # Adapter scale channels are unchanged (depend on model channels).
        # Zero-convs only depend on adapter scale channels & block channels,
        # which are unchanged, so they can stay as-is. We keep them.
        self.penalty_channels = new_penalty_channels

    def adapter_parameters(self):
        """Extend v1.4's adapter-parameter set with the v2.0 modules so
        warm-up (``freeze_backbone``) keeps compiler + routers trainable.
        """
        yield from super().adapter_parameters()
        if self.constraint_compiler is not None:
            yield from self.constraint_compiler.parameters()
        if self.routers is not None:
            yield from self.routers.parameters()
        if self.router_middle is not None:
            yield from self.router_middle.parameters()

    # ------------------------------------------------------------------ #
    # Risk map helper
    # ------------------------------------------------------------------ #
    def _risk_map01_and_floor(self, penalty):
        """Return (M_risk_base in [0,1], W_floor in [risk_base,1]).

        Reuses the v1.5 ``_build_risk_weight`` (which already returns
        ``W = risk_base + (1-risk_base)*M_risk`` or ``None`` when risk-aware
        injection is disabled). We recover ``M_risk_base`` from it so the
        router receives a clean [0,1] prior.
        """
        w_risk = self._build_risk_weight(penalty)  # (B,1,...) or None
        if w_risk is None:
            ones = th.ones(
                penalty.shape[0], 1, *penalty.shape[2:],
                device=penalty.device, dtype=penalty.dtype,
            )
            return ones, ones
        w_risk = w_risk.type(self.dtype)
        denom = max(1e-6, 1.0 - self.risk_base)
        m_risk = ((w_risk - self.risk_base) / denom).clamp(0.0, 1.0)
        return m_risk, w_risk

    # ------------------------------------------------------------------ #
    # Forward
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

        # ---- Base risk map (prior for the router; fixed weight fallback) ----
        m_risk, w_floor = self._risk_map01_and_floor(penalty)
        if self.collect_aux:
            self._aux = {"m_risk": m_risk.detach()}
        self._last_m_risk = m_risk.detach()

        # ---- Stage 1: compile the penalty basis (or use it raw) ----
        if self.use_compiler and self.constraint_compiler is not None:
            cond = None
            if self.compiler_use_cond_context:
                cond = th.cat([ct.type(self.dtype), syn.type(self.dtype)], dim=1)
            adapter_input = self.constraint_compiler(
                penalty.type(self.dtype), dis=dis.type(self.dtype), cond=cond
            )
            if self.collect_aux:
                self._aux["compiled"] = adapter_input.detach()
        else:
            adapter_input = penalty.type(self.dtype)

        # ---- Stage 2: PenaltyAdapter multi-scale features (reused) ----
        penalty_feats = self.penalty_adapter(adapter_input)

        # ---- Standard v1.2 forward, with dynamic / fixed injection ----
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

        router_mean_acc = th.zeros((), device=x.device, dtype=self.dtype)
        router_tv_acc = th.zeros((), device=x.device, dtype=self.dtype)
        router_count = 0
        aux_router_weights = [] if self.collect_aux else None

        for i, module in enumerate(self.input_blocks):
            h_x = module(h, emb)
            h_ct = self.input_blocks_CT[i](h_ct, emb)
            h_syn = self.input_blocks_SYN[i](h_syn, emb)
            h_dis = self.input_blocks_DIS[i](h_dis, emb)

            # ---- v2.0 constraint injection on the X stream ----
            level = self._input_block_levels[i]
            zero_conv_i = self.penalty_zero_convs[i]
            residual = zero_conv_i(penalty_feats[level])

            if self.use_router and self.routers is not None:
                risk_l = self._resize_risk(m_risk, residual)
                w_l = self.routers[i](h_x, penalty_feats[level], risk_l, emb)
                router_mean_acc = router_mean_acc + w_l.mean()
                router_tv_acc = router_tv_acc + router_total_variation(w_l)
                router_count += 1
                if aux_router_weights is not None:
                    aux_router_weights.append(w_l.detach())
            else:
                # Fixed v1.5 weight (also covers v1_5 / compiler_only modes).
                w_l = self._resize_risk(w_floor, residual)

            h_x = h_x + w_l * residual

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

        # ---- v2.0 constraint injection at the bottleneck ----
        residual_mid = self.penalty_zero_conv_middle(penalty_feats[-1])
        if self.use_router and self.router_middle is not None:
            risk_mid = self._resize_risk(m_risk, residual_mid)
            w_mid = self.router_middle(h, penalty_feats[-1], risk_mid, emb)
            router_mean_acc = router_mean_acc + w_mid.mean()
            router_tv_acc = router_tv_acc + router_total_variation(w_mid)
            router_count += 1
            if aux_router_weights is not None:
                aux_router_weights.append(w_mid.detach())
        else:
            w_mid = self._resize_risk(w_floor, residual_mid)
        h = h + w_mid * residual_mid

        # ---- Router regularisation scalars (for FlowMatchingV20) ----
        if router_count > 0:
            self._router_mean_reg = router_mean_acc / router_count
            self._router_tv_reg = router_tv_acc / router_count
        else:
            self._router_mean_reg = th.zeros((), device=x.device, dtype=self.dtype)
            self._router_tv_reg = th.zeros((), device=x.device, dtype=self.dtype)

        if aux_router_weights is not None:
            self._aux["router_weights"] = aux_router_weights

        # ----- Middle + Decoder (unchanged from v1.2) -----
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        return self.out(h)
