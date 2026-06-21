"""
v2.0 — dataset for the Dynamic Relation-to-Field Constraint Router.

The v2.0 first release deliberately **reuses the v1.5 data contract**
unchanged. The returned tuple is exactly:

    (ct, syn, dis, penalty, dose)

with the same crop / normalisation as v1.4/v1.5. v2.0's new behaviour
(constraint compiler + dynamic router) lives entirely inside the
network, so no data-format change is required.

Penalty basis channel order (default):
    [S_target, A_oar, B_boundary, P_fused]
mapped on disk to:
    S_target.nii.gz / A_oar.nii.gz / B_boundary.nii.gz / dose_synthetic.nii.gz

P_fused semantics (IMPORTANT)
-----------------------------
In this project ``P_fused`` currently reuses ``dose_synthetic.nii.gz``.
For v2.0 this is treated explicitly as a **coarse / synthetic prior**,
NOT as ground-truth dose. The compiler uses it only as the coarse-prior
term of ``C_fused``; ground-truth ``dose.nii.gz`` is never used to build
any penalty / compiled field (no label leakage).

If a genuine fused-penalty file ``P_fused.nii.gz`` exists, prefer it via
``--penalty_files`` so the coarse-prior channel reflects a real fused
penalty instead of the synthetic dose. The SYN stream and the penalty
P_fused/coarse-prior channel may coexist, but the redundancy of this
dual synthetic-prior path should be checked with the
``disable_relation_embedding`` / ``router_only`` ablations.
"""
import os

from .dose_loader_3d_v1_5 import (
    Dataset_PSDM_3D_Train_v1_5,
    DEFAULT_V15_PENALTY_FILES,
    DEFAULT_V15_PENALTY_NAMES,
    infer_penalty_name,
)


# v2.0 reuses the v1.5 defaults; aliased here for clarity / discoverability.
DEFAULT_V20_PENALTY_FILES = DEFAULT_V15_PENALTY_FILES
DEFAULT_V20_PENALTY_NAMES = DEFAULT_V15_PENALTY_NAMES

# A genuine fused-penalty file, preferred over dose_synthetic when present.
PREFERRED_FUSED_PENALTY_FILE = "P_fused.nii.gz"


def resolve_v20_penalty_files(patient_probe_dir=None, penalty_files=None):
    """Pick the penalty file list, preferring a real ``P_fused.nii.gz``.

    If ``penalty_files`` is given, it is returned unchanged. Otherwise the
    v1.5 defaults are used, but if every probed patient directory contains
    a real ``P_fused.nii.gz`` we substitute it for ``dose_synthetic.nii.gz``
    in the P_fused slot.
    """
    if penalty_files is not None:
        return tuple(penalty_files)

    files = list(DEFAULT_V20_PENALTY_FILES)
    if patient_probe_dir and os.path.isdir(patient_probe_dir):
        sample_dirs = [
            os.path.join(patient_probe_dir, d)
            for d in sorted(os.listdir(patient_probe_dir))
            if os.path.isdir(os.path.join(patient_probe_dir, d))
        ][:8]
        if sample_dirs and all(
            os.path.exists(os.path.join(d, PREFERRED_FUSED_PENALTY_FILE))
            for d in sample_dirs
        ):
            files = [
                PREFERRED_FUSED_PENALTY_FILE if infer_penalty_name(f) == "P_fused" else f
                for f in files
            ]
    return tuple(files)


class Dataset_PSDM_3D_Train_v2_0(Dataset_PSDM_3D_Train_v1_5):
    """v2.0 dataset — identical contract to v1.5.

    Exists as a thin subclass for naming clarity and to centralise the
    v2.0 default penalty-file resolution (real ``P_fused.nii.gz`` is
    preferred over the synthetic-dose coarse prior when available).
    """

    def __init__(
        self,
        data_root,
        patch_size=(64, 128, 128),
        mask_names=None,
        use_synthetic_dose=True,
        penalty_mode="multi",
        penalty_files=None,
        penalty_normalize=True,
        penalty_channel_names=None,
        prefer_real_fused=True,
    ):
        if penalty_mode == "multi" and penalty_files is None and prefer_real_fused:
            penalty_files = resolve_v20_penalty_files(data_root, None)

        super().__init__(
            data_root=data_root,
            patch_size=patch_size,
            mask_names=mask_names,
            use_synthetic_dose=use_synthetic_dose,
            penalty_mode=penalty_mode,
            penalty_files=penalty_files,
            penalty_normalize=penalty_normalize,
            penalty_channel_names=penalty_channel_names,
        )
        print(f"[v2.0] penalty_files = {list(self.penalty_files)}")
        print(f"[v2.0] penalty_channel_names = {list(self.penalty_channel_names)}")
