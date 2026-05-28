"""
v1.3 — Cross-attention 3D ViT fusion for 3-condition setting (CT / SYN / t).

Drops DIS entirely vs v1.1 / v1.2.

forward_3: Q from CT, K/V from CT + SYN (no DIS).
forward     (2-input): passthrough to forward_3 for compatibility.
forward_4   (v1.2 signature): unsupported, raises NotImplementedError.
"""
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from .vit import PreNorm, FeedForward, Attention, triple


class ViT_fusion_3D_XQuery_v1_3(nn.Module):
    """Cross-attention fusion at the UNet bottleneck — 3-condition variant.

    Args:
        image_size: bottleneck spatial size (D, H, W). e.g. (16, 16, 16) for 128^3 input.
        patch_size: token patch size (pd, ph, pw). e.g. (2, 2, 2).
        dim:        transformer hidden dim. e.g. 1024.
        heads:      attention heads.
        mlp_dim:    FFN hidden dim.
        channels:   spatial channel count of the input feature maps.
        dim_head:   per-head channel count.
    """

    def __init__(self, image_size, patch_size, dim, heads, mlp_dim,
                 channels=1, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        self.image_depth, self.image_height, self.image_width = triple(image_size)
        self.patch_depth, self.patch_height, self.patch_width = triple(patch_size)

        assert self.image_depth % self.patch_depth == 0
        assert self.image_height % self.patch_height == 0
        assert self.image_width % self.patch_width == 0

        num_patches = (
            (self.image_depth // self.patch_depth)
            * (self.image_height // self.patch_height)
            * (self.image_width // self.patch_width)
        )
        patch_dim = channels * self.patch_depth * self.patch_height * self.patch_width

        # Separate patch embeddings for Q and K/V streams.
        self.to_patch_q = nn.Sequential(
            Rearrange(
                'b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)',
                p1=self.patch_depth, p2=self.patch_height, p3=self.patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.to_patch_kv = nn.Sequential(
            Rearrange(
                'b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)',
                p1=self.patch_depth, p2=self.patch_height, p3=self.patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # Independent positional embeddings for Q and K/V sequences.
        self.pos_q = nn.Parameter(torch.randn(1, num_patches, dim))
        self.pos_kv = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.cross_attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

        self.return_linear = nn.Linear(dim, patch_dim)
        self.reshape_back = Rearrange(
            'b (d h w) (p1 p2 p3 c) -> b c (d p1) (h p2) (w p3)',
            d=self.image_depth // self.patch_depth,
            h=self.image_height // self.patch_height,
            w=self.image_width // self.patch_width,
            p1=self.patch_depth, p2=self.patch_height, p3=self.patch_width,
        )

    def forward(self, x_feat, cond_feat):
        """Compatibility shim: calls forward_3(x_feat, cond_feat).

        x_feat and cond_feat are both projected tensors ready for cross-attention.
        Here cond_feat is the fused CT + SYN.
        """
        return self.forward_3(x_feat, cond_feat)

    def forward_3(self, x_ct, x_ct_syn_fused):
        """
        3-input entry point used by UNetModel_GatedXQueryViT_3D_v1_3.

        Args:
            x_ct          : (B, C, D, H, W)  — CT feature, used as Q
            x_ct_syn_fused: (B, C, D, H, W)  — projected CT + SYN fusion, used as K and V

        Returns:
            (B, C, D, H, W) — ViT-refined feature
        """
        assert x_ct.shape == x_ct_syn_fused.shape, (
            f"ViT_v1_3: x_ct {x_ct.shape} != cond {x_ct_syn_fused.shape}"
        )

        q_tokens = self.to_patch_q(x_ct)
        kv_tokens = self.to_patch_kv(x_ct_syn_fused)

        q_tokens = self.dropout(q_tokens + self.pos_q)
        kv_tokens = self.dropout(kv_tokens + self.pos_kv)

        attn_out = self.cross_attn(q_tokens, kv_tokens, kv_tokens) + q_tokens
        out_tokens = self.ff(attn_out) + attn_out

        out_tokens = self.return_linear(out_tokens)
        out = self.reshape_back(out_tokens)
        return out

    def forward_4(self, x_ct, x_syn, x_dis, x_main):
        """Unsupported in v1.3 — DIS is not a valid condition."""
        raise NotImplementedError(
            "forward_4 is not supported in v1.3 (DIS is not a condition). "
            "Use forward_3(x_ct, x_ct_syn_fused) instead."
        )
