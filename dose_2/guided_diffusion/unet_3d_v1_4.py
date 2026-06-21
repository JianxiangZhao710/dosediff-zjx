"""
v1.4 — ControlNet-lite Penalty Adapter on top of the v1.2 backbone.

Design (per v1.4 spec):

  * v1.2 backbone (4-stream X / CT / SYN / DIS encoder, gated residual
    fusion, ViT bottleneck, middle block, decoder) is kept *exactly*
    as-is. We do NOT concat the 3D penalty field into the CT/SYN/DIS
    fusion path.

  * A separate lightweight ``PenaltyAdapter3D`` consumes the penalty
    field (single- or multi-channel; configurable via
    ``penalty_channels``) and produces a multi-scale feature pyramid
    aligned with the main encoder's ``channel_mult`` levels.

  * For each input_block of the main encoder, the matching scale's
    penalty feature is passed through a **zero-initialized** 1x1x1
    Conv3D and added as a residual to the X-stream feature ``h_x``
    *before* the gated condition fusion. Because every zero-conv
    starts at exactly zero (weights AND bias zero), v1.4 at
    initialization is functionally identical to v1.2.

  * One additional zero-conv injection is applied at the bottleneck
    output (after the X-query ViT, before the middle block) using the
    deepest penalty feature. We deliberately do NOT inject into the
    decoder in this first version, to keep the new branch's influence
    well-localized.

  * Optional penalty condition dropout: with probability
    ``penalty_dropout_p`` (default 0.2 when enabled), the penalty input
    is replaced with zeros for that mini-batch. Dropout is only applied
    when ``self.training and self.use_penalty_dropout``.

  * The forward signature is extended with a ``penalty`` argument; the
    output is still the velocity field ``v`` of the same shape as
    ``x`` — flow-matching loss/sampling code does NOT need to change.
"""
import torch as th
import torch.nn as nn

from .nn import conv_nd, timestep_embedding, checkpoint
from .unet_3d_v1_2 import UNetModel_GatedXQueryViT_3D_v1_2
from .penalty_adapter_v1_4 import PenaltyAdapter3D


class UNetModel_PenaltyAdapter_v1_4(UNetModel_GatedXQueryViT_3D_v1_2):
    """v1.2 + ControlNet-lite penalty adapter.

    Additional kwargs (everything else matches v1.2):
        penalty_channels:           int, number of penalty input channels.
        penalty_model_channels:     int, base channel for the penalty adapter.
                                    Defaults to ``model_channels`` of the
                                    main UNet (clean alignment).
        penalty_num_res_blocks:     int, res blocks per level inside adapter.
                                    Default 1.
        penalty_dropout:            float, dropout inside adapter res blocks.
        use_penalty_dropout:        bool, whether to randomly zero-out the
                                    penalty input at training time.
        penalty_dropout_p:          float, prob. of zeroing the penalty
                                    input when ``use_penalty_dropout``.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        ct_channels,
        syn_channels=1,
        dis_channels=11,
        model_channels=64,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=3,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        # v1.1 / v1.2 shared ViT params
        vit_dim=1024,
        vit_heads=4,
        vit_mlp_dim=2048,
        vit_dim_head=64,
        vit_patch_size=(2, 2, 2),
        gate_init_bias=-2.0,
        # v1.4 penalty adapter params
        penalty_channels=1,
        penalty_model_channels=None,
        penalty_num_res_blocks=1,
        penalty_dropout=0.0,
        use_penalty_dropout=True,
        penalty_dropout_p=0.2,
    ):
        # Build v1.2 backbone unchanged.
        super().__init__(
            image_size=image_size,
            in_channels=in_channels,
            ct_channels=ct_channels,
            syn_channels=syn_channels,
            dis_channels=dis_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            num_classes=num_classes,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            vit_dim=vit_dim,
            vit_heads=vit_heads,
            vit_mlp_dim=vit_mlp_dim,
            vit_dim_head=vit_dim_head,
            vit_patch_size=vit_patch_size,
            gate_init_bias=gate_init_bias,
        )

        # --------- Penalty branch hyper-params ---------
        self.penalty_channels = penalty_channels
        self.use_penalty_dropout = bool(use_penalty_dropout)
        self.penalty_dropout_p = float(penalty_dropout_p)
        if penalty_model_channels is None:
            penalty_model_channels = model_channels
        self.penalty_model_channels = penalty_model_channels
        self._penalty_channel_mult = tuple(channel_mult)
        self._penalty_num_levels = len(self._penalty_channel_mult)

        # --------- Recompute input_blocks per-entry channels & levels ---------
        # Mirrors the construction loop of v1.2 __init__ exactly so that
        # we can address each entry of ``self.input_blocks`` by output
        # spatial level and channel count.
        input_block_chans = [int(channel_mult[0] * model_channels)]
        input_block_levels = [0]
        cur_level = 0
        ch = input_block_chans[0]
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                ch = int(mult * model_channels)
                input_block_chans.append(ch)
                input_block_levels.append(cur_level)
            if level != len(channel_mult) - 1:
                # Downsample step: spatial resolution drops one level,
                # channels are unchanged (matches v1.2's Downsample call).
                cur_level += 1
                input_block_chans.append(ch)
                input_block_levels.append(cur_level)
        self._input_block_chans = input_block_chans
        self._input_block_levels = input_block_levels

        # --------- Penalty Adapter (multi-scale 3D encoder) ---------
        self.penalty_adapter = PenaltyAdapter3D(
            penalty_channels=penalty_channels,
            model_channels=penalty_model_channels,
            channel_mult=self._penalty_channel_mult,
            num_res_blocks=penalty_num_res_blocks,
            dims=dims,
            dropout=penalty_dropout,
        )
        # Per-level adapter output channels (must match main UNet channels
        # at that level since ``penalty_model_channels`` defaults to
        # ``model_channels``; if user overrides it, the 1x1x1 zero-conv
        # below absorbs the channel mismatch).
        self._penalty_scale_channels = [
            int(m * penalty_model_channels) for m in self._penalty_channel_mult
        ]

        # --------- Per-input_block zero-conv injectors ---------
        # Each zero-conv: (penalty_scale_channels[level] -> input_block_chans[i]),
        # 1x1x1 kernel, weight AND bias zero-initialised so the very first
        # forward pass adds exactly 0. ``zero_module`` from .nn zeroes
        # parameters in-place; we additionally zero the bias explicitly
        # in case a future Conv3D variant separates parameters.
        self.penalty_zero_convs = nn.ModuleList()
        for i, c_out in enumerate(input_block_chans):
            level = input_block_levels[i]
            c_in = self._penalty_scale_channels[level]
            zc = conv_nd(dims, c_in, c_out, 1)
            nn.init.zeros_(zc.weight)
            if zc.bias is not None:
                nn.init.zeros_(zc.bias)
            self.penalty_zero_convs.append(zc)

        # --------- Bottleneck zero-conv (deepest level -> ch at bottleneck) ---------
        # ``ch`` here is the channel count at the deepest encoder level —
        # equal to the bottleneck feature channel. Build a separate
        # zero-conv mapping the deepest penalty feature to it.
        bottleneck_ch = ch
        deepest_level = self._penalty_num_levels - 1
        c_in_btl = self._penalty_scale_channels[deepest_level]
        self.penalty_zero_conv_middle = conv_nd(dims, c_in_btl, bottleneck_ch, 1)
        nn.init.zeros_(self.penalty_zero_conv_middle.weight)
        if self.penalty_zero_conv_middle.bias is not None:
            nn.init.zeros_(self.penalty_zero_conv_middle.bias)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def adapter_parameters(self):
        """Iterate over all parameters introduced by the v1.4 branch.

        Useful for adapter-warmup training (freeze main backbone, only
        train these).
        """
        yield from self.penalty_adapter.parameters()
        yield from self.penalty_zero_convs.parameters()
        yield from self.penalty_zero_conv_middle.parameters()

    def freeze_backbone(self):
        """Freeze every v1.2 backbone parameter (everything not in the
        adapter / zero-convs). Used for warm-up.
        """
        adapter_param_ids = set(id(p) for p in self.adapter_parameters())
        for p in self.parameters():
            if id(p) not in adapter_param_ids:
                p.requires_grad = False

    def unfreeze_backbone(self):
        """Re-enable gradients on the full model (used after warm-up)."""
        for p in self.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, x, timesteps, ct, syn, dis, penalty, y=None):
        """v1.4 forward.

        Args:
            x        : (B, in_channels,  D, H, W)  — noisy dose x_t.
            timesteps: (B,)                         — Flow Matching time (already scaled to [0, 1000]).
            ct       : (B, ct_channels,  D, H, W)
            syn      : (B, syn_channels, D, H, W)
            dis      : (B, dis_channels, D, H, W)
            penalty  : (B, penalty_channels, D, H, W) — clinical penalty prior.
            y        : reserved (unused, same as v1.2).

        Returns:
            (B, out_channels, D, H, W)  — predicted velocity v.
        """
        # ---- Optional penalty dropout (training only) ----
        if penalty is None:
            # Allow callers to skip the penalty branch entirely (e.g.
            # ablation). We synthesize a zero penalty with the right
            # channel count so the adapter forward shape stays valid.
            penalty = th.zeros(
                x.shape[0], self.penalty_channels,
                x.shape[2], x.shape[3], x.shape[4],
                device=x.device, dtype=x.dtype,
            )
        elif self.training and self.use_penalty_dropout and self.penalty_dropout_p > 0.0:
            # Per-sample Bernoulli mask: drop entire penalty volume of
            # individual samples in the batch.
            keep_mask = (
                th.rand(x.shape[0], device=x.device) > self.penalty_dropout_p
            ).to(x.dtype)
            keep_mask = keep_mask.view(-1, 1, 1, 1, 1)
            penalty = penalty * keep_mask

        # ---- Penalty Adapter: multi-scale features ----
        penalty_feats = self.penalty_adapter(penalty.type(self.dtype))
        # penalty_feats[level] has spatial size matching encoder level ``level``.

        # ---- Standard v1.2 forward, with penalty residual injection ----
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

        # ----- Gated 4-stream encoder (with per-block penalty residual) -----
        for i, module in enumerate(self.input_blocks):
            h_x = module(h, emb)
            h_ct = self.input_blocks_CT[i](h_ct, emb)
            h_syn = self.input_blocks_SYN[i](h_syn, emb)
            h_dis = self.input_blocks_DIS[i](h_dis, emb)

            # ---- v1.4: ControlNet-lite zero-conv penalty residual on X ----
            level = self._input_block_levels[i]
            zero_conv_i = self.penalty_zero_convs[i]
            h_x = h_x + zero_conv_i(penalty_feats[level])

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

        # ---- v1.4: zero-conv penalty residual at bottleneck ----
        h = h + self.penalty_zero_conv_middle(penalty_feats[-1])

        # ----- Middle + Decoder (unchanged from v1.2) -----
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        return self.out(h)
