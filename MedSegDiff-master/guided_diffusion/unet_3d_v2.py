"""
v2 — Conditional 3D Flow-UNet with **two streams** + multi-scale gated
residual condition control + window cross-attention at 32^3 and 16^3.

Differences vs v1.1 (``unet_3d_v1_1.py``):

1. Three-stream encoder (X / CT / DIS) -> **two-stream** encoder
   (X-stream + Condition-stream).
   - Condition input  : ``cat([ct, dis], dim=1)`` -> ``cond_in_channels``
                        channels (default 1 + 11 = 12 for OpenKBP).
   - Condition stem   : ``Conv3d(cond_in_channels -> ch, 3x3x3)``.
   - From there the Condition stream mirrors the X stream's
     ResBlock/Downsample topology.

2. Per-block gated residual control (same idea as v1.1):
       h_c_proj = cond_proj[i](h_c)             # 1x1x1 Conv3D, C -> C
       gate     = sigmoid(gate_proj[i](cat([h_x, h_c_proj], 1)))
       h_x      = h_x + gate * h_c_proj
   ``gate_proj`` final conv: weights zero-init, bias = ``gate_init_bias``
   (default -2.0) -> initial gate ~ 0.12.

3. Multi-scale window cross-attention (Q = X, K/V = condition) at the
   end of the 32^3 and 16^3 encoder scales:
       h_x = h_x + cross_attn_32(h_x, h_c_proj_32)   # window_size = 8
       h_x = h_x + cross_attn_16(h_x, h_c_proj_16)   # window_size = 8
   The ``WindowCrossAttention3D`` output projection is zero-initialised,
   so cross-attn starts from identity (no destabilising kick).

4. Decoder skips use the **condition-controlled** ``h_x`` (after gating
   AND any cross-attention applied at that block).

5. Forward signature is preserved: ``model(x, t, ct, dis) -> v``.
   I/O shapes are unchanged.

All other components (Flow Matching loss, EMA, scheduler, sampler,
penalty / DVH / SDM logic) are NOT touched.
"""

import torch as th
import torch.nn as nn

from .nn import (
    conv_nd,
    linear,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)
from .unet_3d import (
    TimestepEmbedSequential,
    Upsample,
    Downsample,
    ResBlock,
    AttentionBlock,
)
from .cross_attn_3d import WindowCrossAttention3D


class UNetModel_ControlSwinFlow_3D(nn.Module):
    """v2 velocity-field network: ControlSwinFlowUNet3D.

    Mirrors the constructor signature of ``UNetModel_GatedXQueryViT_3D``
    so that the training / inference scripts can swap by name.

    Args specific to v2:
        cross_attn_window_size: window edge length for window cross-attn
            at 32^3 and 16^3 scales (default 8 -> 64 windows of 512
            tokens at 32^3, 8 windows of 512 tokens at 16^3).
        cross_attn_heads: number of heads in the window cross-attn.
        cross_attn_scales: tuple of integer ds factors at which to apply
            window cross-attention (default ``(4, 8)`` corresponds to
            32^3 and 16^3 for a 128^3 input).
        gate_init_bias: bias init for the gate_proj's last conv
            (default -2.0 -> initial gate ~ 0.12).
    """

    def __init__(
        self,
        image_size,             # int or tuple (D, H, W)
        in_channels,            # noisy dose channel count (1)
        ct_channels,            # CT channel count (1)
        syn_channels=1,         # synthetic dose channel count (1)
        dis_channels=11,        # mask channel count (11 for OpenKBP)
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
        # ---- v2 specific ----
        cross_attn_window_size=8,
        cross_attn_heads=4,
        cross_attn_scales=(4, 8),     # ds factors -> 32^3 and 16^3 for 128^3 input
        gate_init_bias=-2.0,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.ct_channels = ct_channels
        self.syn_channels = syn_channels
        self.dis_channels = dis_channels
        self.cond_in_channels = ct_channels + syn_channels + dis_channels  # e.g. 13
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.cross_attn_scales = tuple(cross_attn_scales)
        self.cross_attn_window_size = cross_attn_window_size

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)

        # ----- Two-stream stems -----
        # X-stream: 1 -> ch
        self.input_blocks_X = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))
        ])
        # Condition-stream: cat(ct, dis) (e.g. 12) -> ch
        self.input_blocks_COND = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(dims, self.cond_in_channels, ch, 3, padding=1)
            )
        ])

        input_block_chans = [ch]
        # Track output spatial-downsample factor for every input block
        # (used to decide where to insert window cross-attn).
        output_ds_list = [1]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                out_ch = int(mult * model_channels)
                layers_X = [ResBlock(
                    ch, time_embed_dim, dropout, out_channels=out_ch,
                    dims=dims, use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm)]
                layers_C = [ResBlock(
                    ch, time_embed_dim, dropout, out_channels=out_ch,
                    dims=dims, use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm)]
                ch = out_ch
                if ds in attention_resolutions:
                    layers_X.append(AttentionBlock(
                        ch, num_heads=num_heads,
                        num_head_channels=num_head_channels,
                    ))
                    layers_C.append(AttentionBlock(
                        ch, num_heads=num_heads,
                        num_head_channels=num_head_channels,
                    ))
                self.input_blocks_X.append(TimestepEmbedSequential(*layers_X))
                self.input_blocks_COND.append(TimestepEmbedSequential(*layers_C))
                input_block_chans.append(ch)
                output_ds_list.append(ds)

            if level != len(channel_mult) - 1:
                out_ch = ch
                layers_X = [
                    ResBlock(
                        ch, time_embed_dim, dropout, out_channels=out_ch,
                        dims=dims, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm, down=True)
                    if resblock_updown else
                    Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                ]
                layers_C = [
                    ResBlock(
                        ch, time_embed_dim, dropout, out_channels=out_ch,
                        dims=dims, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm, down=True)
                    if resblock_updown else
                    Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                ]
                self.input_blocks_X.append(TimestepEmbedSequential(*layers_X))
                self.input_blocks_COND.append(TimestepEmbedSequential(*layers_C))
                ch = out_ch
                ds *= 2  # post-downsample factor
                input_block_chans.append(ch)
                output_ds_list.append(ds)

        self.input_block_chans = input_block_chans
        self.output_ds_list = output_ds_list

        # ----- Per-block gated residual control -----
        # cond_proj[i]: 1x1x1 Conv3D, C -> C (h_c -> h_c_proj, channels match h_x).
        # gate_proj[i]: 1x1x1 Conv3D, 2C -> C; weights zero-init, bias = gate_init_bias.
        self.cond_proj = nn.ModuleList()
        self.gate_proj = nn.ModuleList()
        for c in input_block_chans:
            cp = conv_nd(dims, c, c, 1)
            self.cond_proj.append(cp)
            gp = conv_nd(dims, 2 * c, c, 1)
            nn.init.zeros_(gp.weight)
            nn.init.constant_(gp.bias, gate_init_bias)
            self.gate_proj.append(gp)

        # ----- Window cross-attention modules at requested scales -----
        # We attach ONE cross-attention per requested scale, applied at
        # the LAST input block whose output ds matches that scale.
        self.cross_attn_at_block = {}  # block_idx -> nn.Module
        self.cross_attn_modules = nn.ModuleDict()
        for ds_target in self.cross_attn_scales:
            # last block whose output ds == ds_target
            last_idx = -1
            last_ch = None
            for i, (c, d) in enumerate(zip(input_block_chans, output_ds_list)):
                if d == ds_target:
                    last_idx = i
                    last_ch = c
            if last_idx < 0:
                continue
            mod = WindowCrossAttention3D(
                channels=last_ch,
                num_heads=cross_attn_heads,
                window_size=cross_attn_window_size,
            )
            key = f"ds{ds_target}"
            self.cross_attn_modules[key] = mod
            self.cross_attn_at_block[last_idx] = key

        # ----- Middle Block (unchanged) -----
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims,
                     use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(ch, num_heads=num_heads,
                           num_head_channels=num_head_channels),
            ResBlock(ch, time_embed_dim, dropout, dims=dims,
                     use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm),
        )

        # ----- Output (decoder) blocks (unchanged structure) -----
        self.output_blocks = nn.ModuleList([])
        # Local copy because output_blocks construction pops from the
        # channel list in reverse order.
        ds_dec = ds
        ich_stack = list(input_block_chans)
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = ich_stack.pop()
                layers = [
                    ResBlock(
                        ch + ich, time_embed_dim, dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds_dec in attention_resolutions:
                    layers.append(AttentionBlock(
                        ch, num_heads=num_heads_upsample,
                        num_head_channels=num_head_channels,
                    ))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(ch, time_embed_dim, dropout,
                                 out_channels=out_ch, dims=dims,
                                 use_checkpoint=use_checkpoint,
                                 use_scale_shift_norm=use_scale_shift_norm,
                                 up=True)
                        if resblock_updown else
                        Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds_dec //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    # ---------- helpers ----------
    @staticmethod
    def _gated_fuse(h_x, h_c, cond_proj, gate_proj):
        """Gated residual control. h_x channel == h_c channel."""
        h_c_proj = cond_proj(h_c)
        gate = th.sigmoid(gate_proj(th.cat([h_x, h_c_proj], dim=1)))
        return h_x + gate * h_c_proj, h_c_proj

    # ---------- forward ----------
    def forward(self, x, timesteps, ct, syn, dis, y=None):
        """v2 forward.

        Args:
            x        : (B, in_channels,  D, H, W)  noisy dose x_t.
            timesteps: (B,)                         flow-matching time t * 1000.
            ct       : (B, ct_channels,  D, H, W)
            syn      : (B, syn_channels, D, H, W)
            dis      : (B, dis_channels, D, H, W)

        Returns:
            (B, out_channels, D, H, W) predicted velocity v.
        """
        assert x.shape[-3:] == ct.shape[-3:] == syn.shape[-3:] == dis.shape[-3:], (
            f"v2 forward: spatial mismatch x={tuple(x.shape)} "
            f"ct={tuple(ct.shape)} syn={tuple(syn.shape)} dis={tuple(dis.shape)}"
        )
        B = x.shape[0]
        x_in_shape = x.shape

        emb_in = timestep_embedding(timesteps, self.model_channels).type(self.dtype)
        emb = self.time_embed(emb_in)

        h_x = x.type(self.dtype)
        h_c = th.cat([ct, syn, dis], dim=1).type(self.dtype)

        hs = []

        for i, (x_block, c_block) in enumerate(
            zip(self.input_blocks_X, self.input_blocks_COND)
        ):
            h_x = x_block(h_x, emb)
            h_c = c_block(h_c, emb)

            cp = self.cond_proj[i]
            gp = self.gate_proj[i]

            # ---- gated residual control ----
            if self.use_checkpoint:
                fuse_params = list(cp.parameters()) + list(gp.parameters())
                h_x = checkpoint(
                    lambda hx_, hc_, _cp=cp, _gp=gp: _gated_fuse_fn(hx_, hc_, _cp, _gp),
                    (h_x, h_c),
                    fuse_params,
                    self.use_checkpoint,
                )
            else:
                h_c_proj_dbg = cp(h_c)
                gate_dbg = th.sigmoid(gp(th.cat([h_x, h_c_proj_dbg], dim=1)))
                assert h_c_proj_dbg.shape == h_x.shape, (
                    f"cond_proj shape mismatch: h_c_proj{tuple(h_c_proj_dbg.shape)} "
                    f"vs h_x{tuple(h_x.shape)}"
                )
                assert gate_dbg.shape == h_x.shape, (
                    f"gate shape mismatch: gate{tuple(gate_dbg.shape)} "
                    f"vs h_x{tuple(h_x.shape)}"
                )
                h_x = h_x + gate_dbg * h_c_proj_dbg

            # ---- optional window cross-attention at this block's scale ----
            if i in self.cross_attn_at_block:
                key = self.cross_attn_at_block[i]
                attn_mod = self.cross_attn_modules[key]
                # cond_proj is a 1x1x1 conv, cheap to recompute even with
                # gradient checkpointing on.
                h_c_proj = cp(h_c)
                if self.use_checkpoint:
                    attn_params = list(attn_mod.parameters())
                    attn_out = checkpoint(
                        attn_mod,
                        (h_x, h_c_proj),
                        attn_params,
                        self.use_checkpoint,
                    )
                else:
                    attn_out = attn_mod(h_x, h_c_proj)
                    assert attn_out.shape == h_x.shape, (
                        f"cross_attn output {tuple(attn_out.shape)} != "
                        f"h_x {tuple(h_x.shape)}"
                    )
                h_x = h_x + attn_out

            # Skip stores condition-controlled h_x.
            hs.append(h_x)

        # ----- Middle block (unchanged) -----
        h = self.middle_block(h_x, emb)

        # ----- Decoder (unchanged) -----
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        out = self.out(h)
        assert out.shape[-3:] == x_in_shape[-3:] and out.shape[1] == self.out_channels, (
            f"v2 forward: output shape {tuple(out.shape)} mismatches "
            f"input spatial {tuple(x_in_shape)}"
        )
        return out


def _gated_fuse_fn(h_x, h_c, cp, gp):
    """Pure-tensor function used inside ``checkpoint`` (returns one tensor)."""
    h_c_proj = cp(h_c)
    gate = th.sigmoid(gp(th.cat([h_x, h_c_proj], dim=1)))
    return h_x + gate * h_c_proj
