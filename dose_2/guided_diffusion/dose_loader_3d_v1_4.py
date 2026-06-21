"""
v1.4 — OpenKBP 3D dataset extended with a penalty field for the
ControlNet-lite penalty adapter.

Drop-in replacement for ``Dataset_PSDM_3D_Train`` that returns one
extra tensor per sample: a (penalty_channels, D, H, W) penalty volume
covering the same spatial crop as ct / syn / dis / dose.

Two penalty I/O modes are supported:

  * ``mode='single'`` (default):
      The penalty volume is **the same file as the synthetic dose**
      (``dose_synthetic.nii.gz``). In this project's setup, the
      synthetic dose IS the (single-channel) penalty / clinical prior:
      v1.2 already consumes it as the ``SYN`` stream, and v1.4 also
      routes it through the new penalty adapter. To guarantee the two
      pathways see exactly the same volume (including normalisation
      and the random Z crop), the dataset simply re-uses the already
      loaded ``syn`` tensor as the penalty input.
      ``penalty_channels=1``.

  * ``mode='multi'``:
      A list of files (default ['penalty_S_target.nii.gz',
      'penalty_A_oar.nii.gz', 'penalty_B_boundary.nii.gz',
      'penalty_P_fused.nii.gz']). Loaded separately and stacked along
      channel dim. ``penalty_channels=len(files)``. Each channel is
      max-abs normalised (gentle, preserves zero). Missing files are
      filled with zeros — combined with the ``penalty_dropout`` switch
      in the model, this acts as a clean ablation knob.
"""
import os

import numpy as np
import nibabel as nib
import torch

from .dose_loader_3d import Dataset_PSDM_3D_Train, OPENKBP_MASK_NAMES  # noqa: F401


DEFAULT_MULTI_PENALTY_FILES = (
    'penalty_S_target.nii.gz',
    'penalty_A_oar.nii.gz',
    'penalty_B_boundary.nii.gz',
    'penalty_P_fused.nii.gz',
)


class Dataset_PSDM_3D_Train_v1_4(Dataset_PSDM_3D_Train):
    """Same crop / normalisation as v1.2, plus a penalty volume."""

    def __init__(
        self,
        data_root,
        patch_size=(64, 128, 128),
        mask_names=None,
        use_synthetic_dose=True,
        penalty_mode='single',
        penalty_files=None,
        penalty_normalize=True,
    ):
        super().__init__(
            data_root=data_root,
            patch_size=patch_size,
            mask_names=mask_names,
            use_synthetic_dose=use_synthetic_dose,
        )
        if penalty_mode not in ('single', 'multi'):
            raise ValueError(f"penalty_mode must be 'single' or 'multi', got {penalty_mode}")
        self.penalty_mode = penalty_mode
        if penalty_mode == 'single':
            # Single-channel penalty = synthetic dose (dose_synthetic.nii.gz),
            # reused from the SYN stream. No separate file is read.
            self.penalty_files = ()  # informational only
            self.penalty_channels = 1
        else:
            self.penalty_files = tuple(penalty_files) if penalty_files else DEFAULT_MULTI_PENALTY_FILES
            self.penalty_channels = len(self.penalty_files)
        self.penalty_normalize = penalty_normalize

        if penalty_mode == 'single':
            print(f"[v1.4] Penalty mode=single (penalty = dose_synthetic.nii.gz, reused from SYN stream), channels=1")
        else:
            print(
                f"[v1.4] Penalty mode=multi, channels={self.penalty_channels}, "
                f"files={list(self.penalty_files)}"
            )

    def _load_penalty_volume(self, patient_dir, ref_shape):
        """Load and stack the configured penalty file(s); return (P, D, H, W).

        Missing files are filled with zeros. Each channel is optionally
        scaled to roughly [-1, 1] (see ``penalty_normalize``); since we
        don't know the upstream penalty range a priori, we apply a
        gentle per-volume max normalisation that preserves zero.
        """
        channels = []
        for fname in self.penalty_files:
            fpath = os.path.join(patient_dir, fname)
            arr = self.load_nii(fpath)
            if arr is None:
                arr = np.zeros(ref_shape, dtype=np.float32)
            else:
                arr = arr.transpose(2, 0, 1).astype(np.float32)
                if arr.shape != ref_shape:
                    arr = np.zeros(ref_shape, dtype=np.float32)
            if self.penalty_normalize:
                amax = float(np.max(np.abs(arr)))
                if amax > 1e-6:
                    arr = arr / amax
            channels.append(arr)
        return np.stack(channels, axis=0)  # (P, D, H, W)

    def __getitem__(self, index):
        patient_id = self.patient_list[index]
        patient_dir = os.path.join(self.data_root, patient_id)

        # ---- Load CT / dose / syn / dis exactly like the parent class ----
        ct_nii = nib.load(os.path.join(patient_dir, 'ct.nii.gz'))
        ct_data = ct_nii.get_fdata().astype(np.float32).transpose(2, 0, 1)

        dose_data = nib.load(os.path.join(patient_dir, 'dose.nii.gz')).get_fdata().astype(np.float32)
        dose_data = dose_data.transpose(2, 0, 1)

        if self.use_synthetic_dose:
            syn_path = os.path.join(patient_dir, 'dose_synthetic.nii.gz')
            syn_data = self.load_nii(syn_path)
            if syn_data is None:
                syn_data = np.zeros_like(dose_data)
            else:
                syn_data = syn_data.transpose(2, 0, 1)
        else:
            syn_data = np.zeros_like(dose_data)

        empty_shape = ct_data.shape  # (D, H, W)

        mask_channels = []
        for name in self.mask_names:
            m = self.load_nii(os.path.join(patient_dir, '{}.nii.gz'.format(name)))
            if m is None:
                m = np.zeros(empty_shape, dtype=np.float32)
            else:
                m = m.transpose(2, 0, 1)
                if m.shape != empty_shape:
                    m = np.zeros(empty_shape, dtype=np.float32)
            m = (m > 0).astype(np.float32)
            mask_channels.append(m)
        dis_stack = np.stack(mask_channels, axis=0)  # (11, D, H, W)

        # ---- Normalise CT / dose / syn ----
        ct_data = self.normalize_ct(ct_data)
        dose_data = self.normalize_dose(dose_data)
        syn_data = self.normalize_dose(syn_data)

        # ---- Penalty source ----
        # Single mode: penalty volume == synthetic dose (already loaded and
        #              dose-normalised). We assemble a (1, D, H, W) stack
        #              AFTER the shared Z crop below to guarantee identity
        #              with the SYN stream.
        # Multi  mode: load each penalty file independently with max-abs
        #              normalisation; we apply the same Z crop afterwards.
        penalty_stack_full = None
        if self.penalty_mode == 'multi':
            penalty_stack_full = self._load_penalty_volume(patient_dir, empty_shape)

        # ---- Z-axis crop / pad to target depth (shared across all volumes) ----
        current_depth = ct_data.shape[0]
        target_depth = self.patch_size[0]

        if current_depth > target_depth:
            start_z = np.random.randint(0, current_depth - target_depth)
            end_z = start_z + target_depth

            ct_crop = ct_data[start_z:end_z, :, :]
            dose_crop = dose_data[start_z:end_z, :, :]
            syn_crop = syn_data[start_z:end_z, :, :]
            dis_crop = dis_stack[:, start_z:end_z, :, :]
            if penalty_stack_full is not None:
                penalty_crop = penalty_stack_full[:, start_z:end_z, :, :]
        elif current_depth < target_depth:
            pad_z = target_depth - current_depth
            ct_crop = np.pad(ct_data, ((0, pad_z), (0, 0), (0, 0)), 'constant', constant_values=-1.0)
            dose_crop = np.pad(dose_data, ((0, pad_z), (0, 0), (0, 0)), 'constant', constant_values=-1.0)
            syn_crop = np.pad(syn_data, ((0, pad_z), (0, 0), (0, 0)), 'constant', constant_values=-1.0)
            dis_crop = np.pad(dis_stack, ((0, 0), (0, pad_z), (0, 0), (0, 0)), 'constant', constant_values=0)
            if penalty_stack_full is not None:
                penalty_crop = np.pad(penalty_stack_full, ((0, 0), (0, pad_z), (0, 0), (0, 0)),
                                      'constant', constant_values=0)
        else:
            ct_crop = ct_data
            dose_crop = dose_data
            syn_crop = syn_data
            dis_crop = dis_stack
            if penalty_stack_full is not None:
                penalty_crop = penalty_stack_full

        # ---- Single-mode penalty: re-use syn_crop directly ----
        if self.penalty_mode == 'single':
            # syn_crop currently shape (D, H, W); promote to (1, D, H, W).
            penalty_crop = syn_crop[np.newaxis, ...].copy()

        ct_crop = ct_crop[np.newaxis, ...]
        dose_crop = dose_crop[np.newaxis, ...]
        syn_crop = syn_crop[np.newaxis, ...]

        return (
            torch.from_numpy(ct_crop),                          # (1,  D, H, W)
            torch.from_numpy(syn_crop),                         # (1,  D, H, W)
            torch.from_numpy(dis_crop),                         # (11, D, H, W)
            torch.from_numpy(penalty_crop.astype(np.float32)),  # (P,  D, H, W)
            torch.from_numpy(dose_crop),                        # (1,  D, H, W) target
        )
