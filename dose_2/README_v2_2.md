# v2.2 — Basis-Gated Compiler (no new losses)

v2.2 is an **additive** release: v2.0 / v2.1 files are untouched. Only the
constraint compiler forward pass changes (pure architecture).

## Motivation

v2.1 fixed channel death and the `P_fused` shortcut, but inspect showed the
network could still **cancel** basis inside PTV/OAR via additive residual:

- `C_pos = softplus(f + S_target)` with `f ≈ -S` in PTV → output ≈ 0
- `C_boundary = f + B` unbounded → large negative values in PTV

## Changes (architecture only)

| Channel | v2.1 | v2.2 (default `gate`) |
|---------|------|------------------------|
| **C_pos** | `softplus(f + S)` | `softplus(f) * S` |
| **C_neg** | `softplus(f + A)` | `softplus(f) * A` |
| **C_boundary** | `f + B` | `tanh(f) * (|B| + ε)` |
| **C_fused** | `α·pos − β·neg + γ·bnd` (no P bypass) | same |

`P_fused` remains in the 4-channel penalty **basis** (stem input only).

### Gate ablations (`--compiler_gate_mode`)

| Mode | Description |
|------|-------------|
| `gate` (default) | Scheme A — multiply by basis |
| `gate1p` | Scheme B — multiply by `(1 + basis)` |
| `gate_prior` | Scheme C — adds `softplus(log_η) * P_fused` to `C_fused` only |

## New files

| File | Role |
|------|------|
| `guided_diffusion/constraint_field_compiler_v2_2.py` | `ConstraintFieldCompiler3D_v22` |
| `guided_diffusion/unet_3d_v2_2.py` | `UNetModel_DynamicConstraintRouter_v2_2` |
| `flow_matching_v2_2.py` | `FlowMatchingV22` (same loss as v2.0) |
| `scripts/dose_train_3d_v2_2_lung.py` | Single-GPU training |
| `scripts/dose_train_3d_v2_2_lung_ddp.py` | 2-GPU DDP training |
| `scripts/train_v2_2_lung.sh` | Single-GPU launch |
| `scripts/train_v2_2_lung_2gpu.sh` | 2-GPU launch |

## Training

Recommended: warm-start **v2.0 best backbone**, **reset compiler** (random init), freeze backbone 20 epochs.

```bash
cd /root/dose-zjx/dose_2

# 2-GPU (recommended) — scheme 3: backbone warm-start + compiler reset
bash scripts/train_v2_2_lung_2gpu.sh

# Single GPU
bash scripts/train_v2_2_lung.sh

# Ablation: gate1p or weak P_fused on C_fused
bash scripts/train_v2_2_lung_2gpu.sh --compiler_gate_mode gate1p
bash scripts/train_v2_2_lung_2gpu.sh --compiler_gate_mode gate_prior
```

Checkpoints: `trained_models/v2_2_lung/`

### Useful flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--compiler_gate_mode` | `gate` | gating scheme (see above) |
| `--compiler_boundary_eps` | `1e-3` | floor on boundary band magnitude |
| `--freeze_backbone_epochs` | `0` | train compiler+adapter+router only for N epochs |
| `--reset_compiler` | off | skip `constraint_compiler.*` when loading `--resume_from` |

## Inspect compiled fields

```bash
python scripts/inspect_compiled_field_lung.py \
  --version v2_2 \
  --model_path trained_models/v2_2_lung/<run>/ema_best_mae.pth \
  --data_dir /root/autodl-tmp/lung_cancer_processed/processed/test \
  --output_dir predictions/aux_inspect_v22/test_top5 \
  --max_patients 5 --no_save_nii
```

Expected after convergence:

- `C_pos` PTV mean > 0, background ≈ 0
- `C_neg` OAR mean > 0, PTV ≈ 0
- `C_fused` not ≈ `P_fused` (no shortcut)

## Checkpoint compatibility

`strict=False` load from v2.0 / v2.1: backbone, adapter, router transfer;
`constraint_compiler.*` re-inits (forward semantics differ). For `gate_prior`,
`log_fused_prior_scale` is a new scalar parameter.
