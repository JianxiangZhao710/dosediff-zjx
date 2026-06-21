"""
v1.4 — ControlNet-lite 3D Penalty Adapter.

A lightweight multi-scale 3D encoder that takes a penalty field
(single-channel fused or multi-channel [S_target, A_oar, B_boundary, P_fused])
and produces one feature map per resolution level of the main UNet
encoder so that those features can be injected into the X stream as
zero-conv-gated residuals (ControlNet-lite style).

Design constraints (per v1.4 spec):
    - Use only Conv3D / lightweight ResBlock / Downsample.
    - No attention, no time conditioning (penalty is constant per sample).
    - Outputs aligned with main UNet encoder pyramid defined by
      ``channel_mult`` and ``model_channels``.
    - The adapter itself does NOT predict dose / velocity. It only
      provides multi-scale residual condition guidance. Zero-conv
      injection happens outside this module (in the v1.4 UNet) so that
      the adapter's contribution starts at exactly zero.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import conv_nd, normalization
from .unet_3d import Downsample


class _AdapterResBlock(nn.Module):
    """Lightweight 3D residual block (no time embedding).

    Two GroupNorm + SiLU + Conv3D layers with a skip connection.
    Channel-preserving (in_channels == out_channels).
    """

    def __init__(self, channels, dims=3, dropout=0.0):
        super().__init__()
        self.norm1 = normalization(channels)
        self.conv1 = conv_nd(dims, channels, channels, 3, padding=1)
        self.norm2 = normalization(channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return x + h


class PenaltyAdapter3D(nn.Module):
    """ControlNet-lite multi-scale 3D adapter for the penalty field.

    The output spatial pyramid mirrors the main UNet encoder defined by
    ``model_channels`` and ``channel_mult``: level ``l`` has spatial
    size downsampled by ``2 ** l`` and channels ``channel_mult[l] *
    model_channels``.

    Args:
        penalty_channels:  number of input penalty channels (e.g. 1 for
                           a fused field, or 4 for
                           [S_target, A_oar, B_boundary, P_fused]).
        model_channels:    base channel count (must match main UNet for
                           clean alignment, but can be smaller to save
                           memory; channel mismatches are handled by the
                           outside zero-conv since it is 1x1x1 anyway).
        channel_mult:      per-level channel multipliers, same convention
                           as the main UNet (e.g. (1, 2, 4, 4)).
        num_res_blocks:    number of ``_AdapterResBlock`` per level
                           (default 1 — keep this lightweight).
        dims:              spatial dims (3 for 3D).
        dropout:           dropout inside the adapter res blocks.
    """

    def __init__(
        self,
        penalty_channels=1,
        model_channels=32,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=1,
        dims=3,
        dropout=0.0,
    ):
        super().__init__()

        self.dims = dims
        self.penalty_channels = penalty_channels
        self.model_channels = model_channels
        self.channel_mult = tuple(channel_mult)
        self.num_levels = len(self.channel_mult)
        self.num_res_blocks = num_res_blocks
        self.scale_channels = [int(m * model_channels) for m in self.channel_mult]

        # Stem: penalty_channels -> base channels at level 0.
        ch0 = self.scale_channels[0]
        self.stem = conv_nd(dims, penalty_channels, ch0, 3, padding=1)

        # Per-level: optional 1x1 channel adapter (when prev != current),
        # then num_res_blocks lightweight res blocks.
        self.level_blocks = nn.ModuleList()
        # Down-samplers between consecutive levels (n - 1 of them).
        self.downs = nn.ModuleList()

        prev_ch = ch0
        for level, ch in enumerate(self.scale_channels):
            blocks = nn.ModuleList()
            if prev_ch != ch:
                blocks.append(conv_nd(dims, prev_ch, ch, 1))
            else:
                blocks.append(nn.Identity())
            for _ in range(num_res_blocks):
                blocks.append(_AdapterResBlock(ch, dims=dims, dropout=dropout))
            self.level_blocks.append(blocks)

            if level != self.num_levels - 1:
                # Strided 3x3 conv downsample (matches main UNet).
                self.downs.append(
                    Downsample(ch, True, dims=dims, out_channels=ch)
                )
            prev_ch = ch

    def forward(self, penalty):
        """Run the multi-scale adapter encoder.

        Args:
            penalty: (B, penalty_channels, D, H, W)

        Returns:
            list of length ``num_levels``. Element ``l`` has shape
            ``(B, scale_channels[l], D / 2**l, H / 2**l, W / 2**l)``.
        """
        h = self.stem(penalty)
        feats = []
        for level in range(self.num_levels):
            for block in self.level_blocks[level]:
                h = block(h)
            feats.append(h)
            if level != self.num_levels - 1:
                h = self.downs[level](h)
        return feats
