"""
v1.1 — X-query cross-attention 3D ViT fusion module.

Differences vs ``ViT_fusion_3D`` (v1, in vit.py):
    - v1   : forward(x_q=CT, x_k=DIS, x_v=Main_X)   -- Q from CT, K from DIS, V from X
    - v1.1 : forward(x_feat,  cond_feat)            -- Q from X, K and V both from condition

The condition feature is expected to be projected (e.g. from ``cat([h_ct, h_dis], dim=1)``)
*outside* this module by the UNet's ``vit_cond_proj``.

Patch size, token count, dim and other hyperparams are kept identical to v1.
"""
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from .vit import PreNorm, FeedForward, Attention, triple


class ViT_fusion_3D_XQuery(nn.Module):
    """Cross-attention fusion at the UNet bottleneck.

    Args:
        image_size: bottleneck spatial size (D, H, W). e.g. (16, 16, 16) for 128^3 input.
        patch_size: token patch size (pd, ph, pw). e.g. (2, 2, 2).
        dim:        transformer hidden dim. e.g. 1024.
        heads:      attention heads.
        mlp_dim:    FFN hidden dim.
        channels:   spatial channel count of the input feature maps (both x and cond).
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

        # 1) Separate patch embeddings for X and condition.
        self.to_patch_x = nn.Sequential(
            Rearrange(
                'b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)',
                p1=self.patch_depth, p2=self.patch_height, p3=self.patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.to_patch_cond = nn.Sequential(
            Rearrange(
                'b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)',
                p1=self.patch_depth, p2=self.patch_height, p3=self.patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # 2) Independent positional embeddings for the two token sequences.
        self.pos_x = nn.Parameter(torch.randn(1, num_patches, dim))
        self.pos_cond = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # 3) Standard PreNorm(cross-attention) + PreNorm(FFN).
        #    The shared ``Attention`` from vit.py already has independent Wq/Wk/Wv.
        self.cross_attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

        # 4) Reconstruct back to 3D feature map.
        self.return_linear = nn.Linear(dim, patch_dim)
        self.reshape_back = Rearrange(
            'b (d h w) (p1 p2 p3 c) -> b c (d p1) (h p2) (w p3)',
            d=self.image_depth // self.patch_depth,
            h=self.image_height // self.patch_height,
            w=self.image_width // self.patch_width,
            p1=self.patch_depth, p2=self.patch_height, p3=self.patch_width,
        )

    def forward(self, x_feat, cond_feat):
        """
        Args:
            x_feat   : (B, C, D, H, W)  — X-stream bottleneck feature
            cond_feat: (B, C, D, H, W)  — projected condition (same channels as x_feat)

        Returns:
            (B, C, D, H, W) — ViT-refined X feature (residual-friendly; UNet adds it back to x_feat)
        """
        assert x_feat.shape == cond_feat.shape, (
            f"ViT_XQuery: x_feat {x_feat.shape} != cond_feat {cond_feat.shape}"
        )

        x_tokens = self.to_patch_x(x_feat)                # (B, N, dim)
        cond_tokens = self.to_patch_cond(cond_feat)       # (B, N, dim)

        x_tokens = self.dropout(x_tokens + self.pos_x)
        cond_tokens = self.dropout(cond_tokens + self.pos_cond)

        # Cross attention: Q from X, K and V from cond.
        #   PreNorm only normalises its first arg (Q), matching the original ViT_fusion_3D pattern.
        attn_out = self.cross_attn(x_tokens, cond_tokens, cond_tokens) + x_tokens
        out_tokens = self.ff(attn_out) + attn_out

        out_tokens = self.return_linear(out_tokens)       # (B, N, patch_dim)
        out = self.reshape_back(out_tokens)               # (B, C, D, H, W)
        return out

    def forward_4(self, x_ct, x_syn, x_dis, x_main):
        """
        4-input entry point used by v1.2 (and v1 bottleneck fusion).

        Args:
            x_ct   : (B, C, D, H, W)  — CT feature, used as Q
            x_syn  : (B, 1, D, H, W)  — synthetic dose feature
            x_dis  : (B, 11, D, H, W) — DIS mask feature
            x_main : (B, C, D, H, W)  — Main(X) feature, used as V

        Returns:
            (B, C, D, H, W) — ViT-refined Main feature
        """
        # Fused condition = CT + SYN + DIS (same as v1 ViT_fusion_3D)
        x_k = x_ct + x_syn + x_dis

        # Q from CT, K/V from fused condition / Main
        q_tokens = self.to_patch_x(x_ct)                  # (B, N, dim)
        k_tokens = self.to_patch_cond(x_k)               # (B, N, dim)
        v_tokens = self.to_patch_cond(x_main)             # (B, N, dim)

        q_tokens = self.dropout(q_tokens + self.pos_x)
        k_tokens = self.dropout(k_tokens + self.pos_cond)
        v_tokens = self.dropout(v_tokens + self.pos_cond)

        attn_out = self.cross_attn(q_tokens, k_tokens, v_tokens) + q_tokens
        out_tokens = self.ff(attn_out) + attn_out

        out_tokens = self.return_linear(out_tokens)
        out = self.reshape_back(out_tokens)
        return out
