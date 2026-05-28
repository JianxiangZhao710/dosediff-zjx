"""
3D window-based cross-attention used by v2 (UNetModel_ControlSwinFlow_3D).

Q is taken from the X-stream feature map; K and V are taken from the
condition feature map. Attention is computed within non-overlapping
3D windows (window_size^3 tokens per window) for memory efficiency.

Output projection is zero-initialised, so a freshly built block returns 0
and the UNet's residual ``h_x = h_x + cross_attn(h_x, cond)`` starts from
identity, matching the gentle warm-up behaviour of v1.1's gate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowCrossAttention3D(nn.Module):
    """Multi-head cross-attention within fixed 3D windows.

    Args:
        channels: input/output channel count (must equal x and cond channels).
        num_heads: number of attention heads.
        window_size: cubic window edge length. Each window contributes
            ``window_size ** 3`` tokens.
        dim_head: per-head channel count. If ``None``, picked to keep
            ``num_heads * dim_head`` close to ``channels``.
        dropout: attention dropout probability.
    """

    def __init__(self, channels, num_heads=4, window_size=8, dim_head=None,
                 dropout=0.0):
        super().__init__()
        if dim_head is None:
            dim_head = max(channels // num_heads, 32)
        self.channels = channels
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = num_heads * dim_head
        self.window_size = window_size
        self.dropout_p = dropout

        self.norm_x = nn.LayerNorm(channels)
        self.norm_c = nn.LayerNorm(channels)

        self.to_q = nn.Linear(channels, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(channels, self.inner_dim * 2, bias=False)
        self.to_out = nn.Linear(self.inner_dim, channels)

        # Zero-init output projection -> initial cross-attn output = 0
        # so the surrounding UNet residual starts from identity.
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def _to_windows(self, t):
        """(B, C, D, H, W) -> (B * nD * nH * nW, ws**3, C)."""
        B, C, D, H, W = t.shape
        ws = self.window_size
        nD, nH, nW = D // ws, H // ws, W // ws
        # group dims by (window_idx, intra_window_idx)
        t = t.view(B, C, nD, ws, nH, ws, nW, ws)
        # -> (B, nD, nH, nW, ws, ws, ws, C)
        t = t.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        return t.view(B * nD * nH * nW, ws * ws * ws, C)

    def _from_windows(self, t, B, C, D, H, W):
        """(B * nD * nH * nW, ws**3, C) -> (B, C, D, H, W)."""
        ws = self.window_size
        nD, nH, nW = D // ws, H // ws, W // ws
        t = t.view(B, nD, nH, nW, ws, ws, ws, C)
        t = t.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        return t.view(B, C, D, H, W)

    def forward(self, x, cond):
        """
        Args:
            x:    (B, C, D, H, W) — query feature (X-stream).
            cond: (B, C, D, H, W) — key/value feature (condition).

        Returns:
            (B, C, D, H, W) — cross-attended output, same shape as ``x``.
        """
        assert x.shape == cond.shape, \
            f"WindowCrossAttention3D: x{tuple(x.shape)} != cond{tuple(cond.shape)}"
        B, C, D, H, W = x.shape
        ws = self.window_size
        assert D % ws == 0 and H % ws == 0 and W % ws == 0, (
            f"WindowCrossAttention3D: spatial {(D, H, W)} not divisible "
            f"by window_size={ws}"
        )

        xw = self.norm_x(self._to_windows(x))   # (Bw, N, C)
        cw = self.norm_c(self._to_windows(cond))

        Bw, N, _ = xw.shape
        q = self.to_q(xw).view(Bw, N, self.num_heads, self.dim_head).transpose(1, 2)
        kv = self.to_kv(cw).view(Bw, N, 2, self.num_heads, self.dim_head)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, Bw, h, N, d)
        k, v = kv[0], kv[1]

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )  # (Bw, h, N, d)
        out = out.transpose(1, 2).contiguous().view(Bw, N, self.inner_dim)
        out = self.to_out(out)                  # (Bw, N, C)

        return self._from_windows(out, B, C, D, H, W)
