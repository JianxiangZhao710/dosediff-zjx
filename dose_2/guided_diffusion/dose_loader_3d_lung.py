"""
Lung-cancer dataset adapter for the v2.0 Dynamic Constraint Router.

This dataset maps the on-disk ``lung_cancer_processed`` layout (plus the
copied ABPSF penalty components) onto the v2.0 data contract:

    (ct, syn, dis, penalty, dose[, body])

Key differences vs. the OpenKBP loaders
---------------------------------------
* **Variable per-case size** is handled by cropping / padding *all three*
  spatial axes to ``patch_size`` (default 128^3). Training uses a random
  crop, validation a deterministic center crop. Smaller volumes are
  zero/-1 padded so every sample is exactly ``patch_size``.
* **Masks live in sub-directories** (``resolved/resolved_*.nii.gz``); the
  Body mask is *excluded* from the network input (per request) and is
  only returned (optionally) for body-region MAE evaluation.
* **DIS channel layout** keeps the hard-coded semantics used by the
  weighted MSE / RelationEncoder (``dis[:,0:3]`` = PTV, ``dis[:,3:10]`` =
  OAR):

      ch0  : PTV (Target)
      ch1-2: empty (zeros, no extra PTV levels)
      ch3  : Lung_L
      ch4  : Lung_R
      ch5  : Heart
      ch6  : Esophagus
      ch7  : SpinalCord
      ch8  : Trachea
      ch9-10: empty (zeros; Body excluded)

* **syn / penalty priors**: ``final_total.nii.gz`` (a [0,1] dose-attention
  field) is used *both* as the SYN stream (mapped ``2*x-1`` to the dose
  range, **no GT leakage**) and as the penalty ``P_fused`` coarse-prior
  channel. The penalty basis is ``[S_target, A_oar, B_boundary,
  final_total->P_fused]`` with gentle per-volume max-abs normalisation.
"""
import os

import numpy as np
import nibabel as nib
import torch

from .dose_loader_3d import Dataset_PSDM_3D_Train


# Default penalty basis (files copied into each case dir).
DEFAULT_LUNG_PENALTY_FILES = (
    "S_target.nii.gz",
    "A_oar.nii.gz",
    "B_boundary.nii.gz",
    "final_total.nii.gz",  # -> P_fused coarse prior
)
DEFAULT_LUNG_PENALTY_NAMES = ("S_target", "A_oar", "B_boundary", "P_fused")

# Mask sources (relative to a patient dir). ``None`` => zero channel.
DEFAULT_LUNG_PTV_SOURCE = "resolved/resolved_Target_Total_from_RS"
DEFAULT_LUNG_OAR_SOURCES = (
    "resolved/resolved_Lung_L",
    "resolved/resolved_Lung_R",
    "resolved/resolved_Heart",
    "resolved/resolved_Esophagus",
    "resolved/resolved_SpinalCord",
    "resolved/resolved_Trachea",
)
DEFAULT_LUNG_BODY_SOURCE = "resolved/resolved_Body"

DIS_CHANNELS = 11  # keep OpenKBP-compatible width (0:3 PTV, 3:10 OAR, 10 spare)


class Dataset_Lung_v2_0(Dataset_PSDM_3D_Train):
    """Lung-cancer dataset for v2.0 training / validation.

    Args:
        data_root:    split dir containing one sub-dir per patient.
        patch_size:   (D, H, W) target patch, multiple of 16.
        is_train:     random crop (True) vs. deterministic center crop.
        return_body:  also return the Body mask (eval-only, not an input).
        penalty_files / penalty_channel_names: penalty basis override.
        ptv_source / oar_sources / body_source: mask path overrides.
    """

    def __init__(
        self,
        data_root,
        patch_size=(128, 128, 128),
        is_train=True,
        return_body=False,
        penalty_files=None,
        penalty_channel_names=None,
        penalty_normalize=True,
        ptv_source=DEFAULT_LUNG_PTV_SOURCE,
        oar_sources=DEFAULT_LUNG_OAR_SOURCES,
        body_source=DEFAULT_LUNG_BODY_SOURCE,
        syn_from_final_total=True,
    ):
        # The base class builds patient_list and provides normalize_* / load_nii.
        super().__init__(
            data_root=data_root,
            patch_size=patch_size,
            mask_names=None,
            use_synthetic_dose=False,
        )
        self.is_train = bool(is_train)
        self.return_body = bool(return_body)
        self.penalty_normalize = bool(penalty_normalize)
        self.syn_from_final_total = bool(syn_from_final_total)

        self.penalty_files = tuple(penalty_files) if penalty_files else DEFAULT_LUNG_PENALTY_FILES
        if penalty_channel_names:
            self.penalty_channel_names = tuple(penalty_channel_names)
        else:
            self.penalty_channel_names = DEFAULT_LUNG_PENALTY_NAMES
        self.penalty_channels = len(self.penalty_files)

        self.ptv_source = ptv_source
        self.oar_sources = tuple(oar_sources)
        self.body_source = body_source

        # final_total is the SYN / P_fused prior source (last penalty file).
        self.final_total_file = self.penalty_files[-1]

        print(
            f"[lung-v2.0] root={data_root} | cases={len(self.patient_list)} | "
            f"patch={tuple(patch_size)} | train={self.is_train} | "
            f"penalty_files={list(self.penalty_files)} | "
            f"penalty_names={list(self.penalty_channel_names)} | "
            f"syn_from_final_total={self.syn_from_final_total} | "
            f"return_body={self.return_body}"
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _load_dhw(self, path, ref_shape=None, binary=False):
        """Load a NIfTI as (D, H, W) float32; zeros if missing / mismatched."""
        arr = self.load_nii(path)
        if arr is None:
            if ref_shape is None:
                return None
            return np.zeros(ref_shape, dtype=np.float32)
        arr = arr.transpose(2, 0, 1).astype(np.float32)
        if ref_shape is not None and arr.shape != ref_shape:
            return np.zeros(ref_shape, dtype=np.float32)
        if binary:
            arr = (arr > 0).astype(np.float32)
        return arr

    def _plan_starts(self, shape_dhw):
        starts = []
        for s, p in zip(shape_dhw, self.patch_size):
            if s > p:
                st = np.random.randint(0, s - p + 1) if self.is_train else (s - p) // 2
            else:
                st = 0
            starts.append(int(st))
        return starts

    def _crop_pad(self, arr, starts, pad_value, channel_first=False):
        spatial = arr.shape[1:] if channel_first else arr.shape
        sl = [slice(st, min(st + p, s)) for st, p, s in zip(starts, self.patch_size, spatial)]
        out = arr[(slice(None),) + tuple(sl)] if channel_first else arr[tuple(sl)]
        spatial_out = out.shape[1:] if channel_first else out.shape
        pads = [(0, p - o) for p, o in zip(self.patch_size, spatial_out)]
        if channel_first:
            pads = [(0, 0)] + pads
        if any(pa[1] > 0 for pa in pads):
            out = np.pad(out, pads, "constant", constant_values=pad_value)
        return out

    # ------------------------------------------------------------------ #
    # Item
    # ------------------------------------------------------------------ #
    def __getitem__(self, index):
        patient_id = self.patient_list[index]
        pdir = os.path.join(self.data_root, patient_id)

        ct = self._load_dhw(os.path.join(pdir, "ct.nii.gz"))
        if ct is None:
            raise FileNotFoundError(f"ct.nii.gz missing for {patient_id}")
        ref_shape = ct.shape

        dose = self._load_dhw(os.path.join(pdir, "dose.nii.gz"), ref_shape)

        # ---- DIS stack (Body excluded) ----
        dis = np.zeros((DIS_CHANNELS,) + ref_shape, dtype=np.float32)
        ptv = self._load_dhw(os.path.join(pdir, self.ptv_source + ".nii.gz"), ref_shape, binary=True)
        dis[0] = ptv
        for i, src in enumerate(self.oar_sources):
            dis[3 + i] = self._load_dhw(os.path.join(pdir, src + ".nii.gz"), ref_shape, binary=True)

        # ---- Penalty basis [S_target, A_oar, B_boundary, P_fused] ----
        pen_channels = []
        for fname in self.penalty_files:
            arr = self._load_dhw(os.path.join(pdir, fname), ref_shape)
            if self.penalty_normalize:
                amax = float(np.max(np.abs(arr)))
                if amax > 1e-6:
                    arr = arr / amax
            pen_channels.append(arr)
        penalty = np.stack(pen_channels, axis=0)

        # ---- SYN stream from final_total ([0,1] -> [-1,1]); else zeros ----
        if self.syn_from_final_total:
            ft = self._load_dhw(os.path.join(pdir, self.final_total_file), ref_shape)
            syn = ft * 2.0 - 1.0
        else:
            syn = np.full(ref_shape, -1.0, dtype=np.float32)

        # ---- Optional Body mask (eval only) ----
        body = None
        if self.return_body:
            body = self._load_dhw(os.path.join(pdir, self.body_source + ".nii.gz"), ref_shape, binary=True)

        # ---- Normalise CT / dose ----
        ct = self.normalize_ct(ct)
        dose = self.normalize_dose(dose)

        # ---- Shared crop / pad to patch_size on all 3 axes ----
        starts = self._plan_starts(ref_shape)
        ct = self._crop_pad(ct, starts, pad_value=-1.0)
        dose = self._crop_pad(dose, starts, pad_value=-1.0)
        syn = self._crop_pad(syn, starts, pad_value=-1.0)
        dis = self._crop_pad(dis, starts, pad_value=0.0, channel_first=True)
        penalty = self._crop_pad(penalty, starts, pad_value=0.0, channel_first=True)
        if body is not None:
            body = self._crop_pad(body, starts, pad_value=0.0)

        out = [
            torch.from_numpy(ct[np.newaxis, ...].astype(np.float32)),       # (1,  D,H,W)
            torch.from_numpy(syn[np.newaxis, ...].astype(np.float32)),      # (1,  D,H,W)
            torch.from_numpy(dis.astype(np.float32)),                       # (11, D,H,W)
            torch.from_numpy(penalty.astype(np.float32)),                   # (P,  D,H,W)
            torch.from_numpy(dose[np.newaxis, ...].astype(np.float32)),     # (1,  D,H,W)
        ]
        if self.return_body:
            out.append(torch.from_numpy(body[np.newaxis, ...].astype(np.float32)))
        return tuple(out)
