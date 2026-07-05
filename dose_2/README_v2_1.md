# v2.1 — Compiler Architecture Fix (no new losses)

v2.1 is an **additive** release: all v2.0 files are untouched. Only the
constraint compiler forward pass changes (pure architecture).

## Motivation

Inspection on the lung test set showed v2.0 **channel collapse**:

- `C_pos` / `C_neg` ≈ 0 everywhere (ReLU allowed `f` to cancel basis)
- `C_fused` ≈ `P_fused` direct bypass (`+ p` term)

## Changes (architecture only, no new loss)

| ID | Change | v2.0 | v2.1 |
|----|--------|------|------|
| **A** | Break P_fused shortcut | `C_fused = α·C_pos − β·C_neg + γ·C_boundary + P_fused` | `C_fused = α·C_pos − β·C_neg + γ·C_boundary` |
| **B** | Anti channel-death | `C_pos = ReLU(f+s)`, `C_neg = ReLU(f+a)` | `C_pos = softplus(f+s)`, `C_neg = softplus(f+a)` |

`P_fused` remains in the 4-channel penalty **basis** (stem input). SYN stream
is unchanged. No alignment / semantic loss is added.

## New files

| File | Role |
|------|------|
| `guided_diffusion/constraint_field_compiler_v2_1.py` | `ConstraintFieldCompiler3D_v21` |
| `guided_diffusion/unet_3d_v2_1.py` | `UNetModel_DynamicConstraintRouter_v2_1` |
| `flow_matching_v2_1.py` | `FlowMatchingV21` (same loss as v2.0) |
| `scripts/dose_train_3d_v2_1_lung.py` | Lung training |
| `scripts/train_v2_1_lung.sh` | Launch script |

## Training

```bash
cd /root/dose-zjx/dose_2
bash scripts/train_v2_1_lung.sh

# Optional: warm-start backbone+router from v2.0 (compiler re-inits)
bash scripts/train_v2_1_lung.sh \
  --resume_from trained_models/v2_0_lung/lung_full_v2_0_ep400/ema_best_mae.pth
```

Checkpoints: `trained_models/v2_1_lung/`

## Ablation modes

Same as v2.0 plus `full_v2_1` (default). `--mode full_v2_1` enables v2.1
compiler + dynamic router.

## Inspect compiled fields

```bash
conda run -n mednext python scripts/inspect_compiled_field_lung.py \
  --version v2_1 \
  --model_path trained_models/v2_1_lung/<run>/ema_best_mae.pth \
  --data_dir /root/autodl-tmp/lung_cancer_processed/processed/test \
  --output_dir predictions/aux_inspect_v21/test_top5 \
  --max_patients 5 --no_save_nii
```

## Checkpoint compatibility

Loading a v2.0 checkpoint with `strict=False`: backbone, adapter, router
weights transfer; `constraint_compiler.*` is re-initialised (forward
semantics differ).
