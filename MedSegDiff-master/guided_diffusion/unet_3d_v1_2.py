"""
v1.2 — Conditional 3D UNet velocity-field network, extending v1.1 with
full 4-stream condition handling in the encoder and ViT bottleneck.

Changes vs ``unet_3d_v1_1.py`` (UNetModel_GatedXQueryViT_3D):

1. Encoder: properly track ALL FOUR streams (X / CT / SYN / DIS) instead
   of only X / CT / DIS.  The ``last_h_syn`` feature is now captured
   for use in the ViT bottleneck.

2. Bottleneck ViT: uses ``ViT_fusion_3D_XQuery_v1_2`` which calls
   ``forward_4(ct_feat, syn_feat, dis_feat, main_feat)`` so that the
   synthetic dose has its own dedicated patch embedding and contributes
   to the cross-attention key/value alongside CT and DIS.

3. The ``cond_proj`` and ``gate_proj`` logic (gated residual control)
   is unchanged — they still fuse CT + SYN + DIS -> C then modulate X.

4. Forward signature and all I/O shapes are preserved.

The only semantic difference is that the synthetic dose now gets
explicit per-token representation inside the bottleneck ViT, rather
than being flattened into the condition projection before the ViT.
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
from .vit_v1_2 import ViT_fusion_3D_XQuery_v1_2


class UNetModel_GatedXQueryViT_3D_v1_2(nn.Module):
    """v1.2 velocity-field network.

    Inherits the v1.1 backbone and gated fusion, but:
      - Tracks ``last_h_syn`` separately in the 4-stream encoder.
      - Uses ``ViT_fusion_3D_XQuery_v1_2`` so the bottleneck ViT
        sees explicit CT / SYN / DIS / Main features.

    Args mirror ``UNetModel_GatedXQueryViT_3D`` so existing training /
    inference code can swap by name.

    Additional behaviour:
        - ``last_h_syn`` is now captured for ViT bottleneck (v1.1 only
          tracked CT, DIS).
        - Bottleneck ViT is ``ViT_fusion_3D_XQuery_v1_2`` (calls forward_4).
    """

    def __init__(
        self,
        image_size,           # (D, H, W) or int
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
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
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

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)

        # ----- Four-stream input blocks (X / CT / SYN / DIS) -----
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))
        ])
        self.input_blocks_CT = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, ct_channels, ch, 3, padding=1))
        ])
        self.input_blocks_SYN = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, syn_channels, ch, 3, padding=1))
        ])
        self.input_blocks_DIS = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, dis_channels, ch, 3, padding=1))
        ])

        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                out_ch = int(mult * model_channels)
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                                   use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                layers_CT = [ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                                      use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                layers_SYN = [ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                                       use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                layers_DIS = [ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                                       use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                ch = out_ch
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels))
                    layers_CT.append(AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels))
                    layers_SYN.append(AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels))
                    layers_DIS.append(AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.input_blocks_CT.append(TimestepEmbedSequential(*layers_CT))
                self.input_blocks_SYN.append(TimestepEmbedSequential(*layers_SYN))
                self.input_blocks_DIS.append(TimestepEmbedSequential(*layers_DIS))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                layers = [
                    ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                             use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True)
                    if resblock_updown else
                    Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                ]
                layers_CT = [
                    ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                             use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True)
                    if resblock_updown else
                    Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                ]
                layers_SYN = [
                    ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                             use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True)
                    if resblock_updown else
                    Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                ]
                layers_DIS = [
                    ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                             use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True)
                    if resblock_updown else
                    Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                ]
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.input_blocks_CT.append(TimestepEmbedSequential(*layers_CT))
                self.input_blocks_SYN.append(TimestepEmbedSequential(*layers_SYN))
                self.input_blocks_DIS.append(TimestepEmbedSequential(*layers_DIS))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        # ----- Per-block gated condition fusion (unchanged from v1.1) -----
        # cond_proj[i]: cat(h_ct, h_syn, h_dis) (3C) -> h_cond (C)
        # gate_proj[i]: cat(h_x, h_cond) (2C) -> gate logits (C). bias init -> gate_init_bias.
        self.cond_proj = nn.ModuleList()
        self.gate_proj = nn.ModuleList()
        for c in input_block_chans:
            cp = conv_nd(dims, 3 * c, c, 1)
            self.cond_proj.append(cp)
            gp = conv_nd(dims, 2 * c, c, 1)
            nn.init.zeros_(gp.weight)
            nn.init.constant_(gp.bias, gate_init_bias)
            self.gate_proj.append(gp)

        # ----- Bottleneck: ViT with 4-condition input -----
        ds_factor = 2 ** (len(channel_mult) - 1)
        if isinstance(image_size, int):
            feature_size = (image_size // ds_factor, image_size // ds_factor, image_size // ds_factor)
        else:
            feature_size = tuple([s // ds_factor for s in image_size])

        # Project cat([last_h_ct, last_h_syn, last_h_dis], 1) (3*ch) -> ch
        self.vit_cond_proj = conv_nd(dims, 3 * ch, ch, 1)

        # v1.2: use ViT_fusion_3D_XQuery_v1_2 which exposes forward_4
        self.fusion = ViT_fusion_3D_XQuery_v1_2(
            image_size=feature_size,
            patch_size=vit_patch_size,
            dim=vit_dim,
            heads=vit_heads,
            mlp_dim=vit_mlp_dim,
            channels=ch,
            dim_head=vit_dim_head,
        )

        # ----- Middle Block (unchanged) -----
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels),
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm),
        )

        # ----- Output Blocks (unchanged) -----
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich, time_embed_dim, dropout, out_channels=int(model_channels * mult),
                             dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads_upsample, num_head_channels=num_head_channels))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                                 use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, up=True)
                        if resblock_updown else
                        Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def _vit_bottleneck(self, last_h_x, last_h_ct, last_h_syn, last_h_dis):
        """v1.2 bottleneck: call ViT forward_4 with explicit SYN tensor."""
        cond_feat = self.vit_cond_proj(th.cat([last_h_ct, last_h_syn, last_h_dis], dim=1))
        # forward_4: CT as Q, CT+SYN+DIS as K, Main as V
        return last_h_x + self.fusion.forward_4(last_h_ct, last_h_syn, last_h_dis, last_h_x)

    def forward(self, x, timesteps, ct, syn, dis, y=None):
        """v1.2 forward.

        Args:
            x        : (B, in_channels,  D, H, W)  — noisy dose x_t.
            timesteps: (B,)                         — Flow Matching time (already scaled to [0, 1000]).
            ct       : (B, ct_channels,  D, H, W)
            syn      : (B, syn_channels, D, H, W)  ← NEW in v1.2
            dis      : (B, dis_channels, D, H, W)

        Returns:
            (B, out_channels, D, H, W)  — predicted velocity v.
        """
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

        # ----- Gated 4-stream encoder -----
        for i, module in enumerate(self.input_blocks):
            h_x = module(h, emb)
            h_ct = self.input_blocks_CT[i](h_ct, emb)
            h_syn = self.input_blocks_SYN[i](h_syn, emb)
            h_dis = self.input_blocks_DIS[i](h_dis, emb)

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
            last_h_syn = h_syn   # v1.2: now tracked separately
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

        # ----- Middle + Decoder (unchanged) -----
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        return self.out(h)
