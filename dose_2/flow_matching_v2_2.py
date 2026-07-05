"""
v2.2 — Flow-Matching wrapper.

Loss, sampling and optional regularisers are **identical** to v2.0
(``FlowMatchingV20``).  v2.2 changes live only inside
``ConstraintFieldCompiler3D_v22``; no new loss terms are added.
"""
from flow_matching_v2_0 import FlowMatchingV20


class FlowMatchingV22(FlowMatchingV20):
    """Flow-Matching wrapper for v2.2 (basis-gated compiler)."""

    pass
