"""
v1.2 — Thin wrapper around ``ViT_fusion_3D_XQuery`` that exposes
``forward_4(ct_feat, syn_feat, dis_feat, main_feat)`` explicitly.

Since ``ViT_fusion_3D_XQuery`` (v1.1) now has ``forward_4`` natively,
this module simply re-exports it under a v1.2 name so the UNet can
swap the ViT class by name.

All arguments are passed directly to ``ViT_fusion_3D_XQuery``.
"""
import torch.nn as nn
from .vit_v1_1 import ViT_fusion_3D_XQuery


class ViT_fusion_3D_XQuery_v1_2(ViT_fusion_3D_XQuery):
    """v1.2 ViT fusion — same transformer as v1.1, but the UNet calls
    ``forward_4`` (added to the base class) instead of the 2-input ``forward``.
    """

    def forward_4(self, x_ct, x_syn, x_dis, x_main):
        """Explicit 4-input entry point — used by UNetModel_GatedXQueryViT_3D_v1_2."""
        return super().forward_4(x_ct, x_syn, x_dis, x_main)
