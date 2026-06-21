"""
v1.5 — dataset for the Risk-aware ControlNet-lite Penalty Adapter.

Identical sampling / cropping / normalisation to the v1.4 dataset
(``Dataset_PSDM_3D_Train_v1_4``); the only additions are:

  * v1.5-specific default penalty file list mapped to the *actual*
    component files present in each patient directory:

        S_target.nii.gz        -> S_target
        A_oar.nii.gz           -> A_oar
        B_boundary.nii.gz      -> B_boundary
        dose_synthetic.nii.gz  -> P_fused   (the fused / synthetic dose)

    so the default penalty channel order is
    [S_target, A_oar, B_boundary, P_fused].

  * A ``penalty_channel_names`` attribute describing the channel ordering,
    consumed by ``UNetModel_RiskAwarePenaltyAdapter_v1_5`` to build the
    soft risk weight map.

The returned tuple is exactly the v1.4 one:
    (ct, syn, dis, penalty, dose)
"""
import os

from .dose_loader_3d_v1_4 import Dataset_PSDM_3D_Train_v1_4


# Default v1.5 multi-channel penalty: real on-disk file names.
DEFAULT_V15_PENALTY_FILES = (
    'S_target.nii.gz',
    'A_oar.nii.gz',
    'B_boundary.nii.gz',
    'dose_synthetic.nii.gz',  # P_fused
)
DEFAULT_V15_PENALTY_NAMES = (
    'S_target',
    'A_oar',
    'B_boundary',
    'P_fused',
)


def infer_penalty_name(fname):
    """Map a penalty file name to a canonical component name."""
    base = os.path.basename(fname).lower()
    if 's_target' in base:
        return 'S_target'
    if 'a_oar' in base:
        return 'A_oar'
    if 'b_boundary' in base:
        return 'B_boundary'
    if 'synthetic' in base or 'p_fused' in base or 'fused' in base:
        return 'P_fused'
    return os.path.splitext(os.path.splitext(base)[0])[0]


class Dataset_PSDM_3D_Train_v1_5(Dataset_PSDM_3D_Train_v1_4):
    """v1.4 dataset with v1.5 default penalty files + channel naming."""

    def __init__(
        self,
        data_root,
        patch_size=(64, 128, 128),
        mask_names=None,
        use_synthetic_dose=True,
        penalty_mode='multi',
        penalty_files=None,
        penalty_normalize=True,
        penalty_channel_names=None,
    ):
        if penalty_mode == 'multi' and penalty_files is None:
            penalty_files = DEFAULT_V15_PENALTY_FILES

        super().__init__(
            data_root=data_root,
            patch_size=patch_size,
            mask_names=mask_names,
            use_synthetic_dose=use_synthetic_dose,
            penalty_mode=penalty_mode,
            penalty_files=penalty_files,
            penalty_normalize=penalty_normalize,
        )

        # Resolve channel names (used by the risk-aware UNet).
        if penalty_channel_names is not None:
            self.penalty_channel_names = tuple(penalty_channel_names)
        elif penalty_mode == 'single':
            self.penalty_channel_names = ('P_fused',)
        else:
            self.penalty_channel_names = tuple(
                infer_penalty_name(f) for f in self.penalty_files
            )

        print(f"[v1.5] penalty_channel_names = {list(self.penalty_channel_names)}")
