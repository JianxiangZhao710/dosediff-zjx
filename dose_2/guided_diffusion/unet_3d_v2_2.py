"""
v2.2 — Dynamic Relation-to-Field Constraint Router (UNet).

Identical to v2.0 / v2.1 except the constraint compiler is replaced with
``ConstraintFieldCompiler3D_v22`` (basis gating + bounded boundary).

Router, PenaltyAdapter, backbone, losses and checkpoint loading are
unchanged from v2.0.  Older checkpoints load with ``strict=False``.
"""
from .unet_3d_v2_0 import (
    UNetModel_DynamicConstraintRouter_v2_0,
    resolve_v2_mode,
    V2_MODES,
    router_total_variation,
)
from .constraint_field_compiler_v2_2 import (
    ConstraintFieldCompiler3D_v22,
    V22_GATE_MODES,
)

V2_2_MODES = V2_MODES + ("full_v2_2",)


def resolve_v2_2_mode(mode=None, use_constraint_compiler=None, use_dynamic_router=None):
    """Like ``resolve_v2_mode``; ``full_v2_2`` aliases ``full_v2_0`` flags."""
    user_mode = mode
    if mode == "full_v2_2":
        mode = "full_v2_0"
    resolved_mode, use_compiler, use_router = resolve_v2_mode(
        mode, use_constraint_compiler, use_dynamic_router,
    )
    if user_mode == "full_v2_2":
        resolved_mode = "full_v2_2"
    return resolved_mode, use_compiler, use_router


class UNetModel_DynamicConstraintRouter_v2_2(UNetModel_DynamicConstraintRouter_v2_0):
    """v2.0 UNet + v2.2 basis-gated constraint compiler."""

    def __init__(self, *args, **kwargs):
        compiler_width = kwargs.get("compiler_width")
        compiler_num_res_blocks = kwargs.get("compiler_num_res_blocks", 2)
        relation_dim = kwargs.get("relation_dim", 128)
        compiler_use_cond_context = kwargs.get("compiler_use_cond_context", False)
        compiler_gate_mode = kwargs.pop("compiler_gate_mode", "gate")
        compiler_boundary_eps = kwargs.pop("compiler_boundary_eps", 1e-3)
        dis_channels = kwargs.get("dis_channels", 11)
        requested_mode = kwargs.get("mode")
        if requested_mode == "full_v2_2":
            kwargs = dict(kwargs)
            kwargs["mode"] = "full_v2_0"

        super().__init__(*args, **kwargs)

        if self.use_compiler:
            width = compiler_width if compiler_width is not None else self.model_channels
            cond_channels = 2 if compiler_use_cond_context else 0
            self.constraint_compiler = ConstraintFieldCompiler3D_v22(
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
                gate_mode=compiler_gate_mode,
                boundary_eps=compiler_boundary_eps,
            )

        if requested_mode == "full_v2_2":
            self.v2_mode = "full_v2_2"
