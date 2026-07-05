#!/usr/bin/env python3
"""
dose_train_3d_v2_0_lung.py
==========================
Single-GPU v2.0 (Dynamic Relation-to-Field Constraint Router) training on
the lung-cancer dataset.

Inputs per case (see ``dose_loader_3d_lung.Dataset_Lung_v2_0``):
    ct          : CT
    syn         : final_total mapped to [-1,1] (coarse dose-attention prior)
    dis (11ch)  : Target (PTV, ch0) + 6 OARs (ch3-8); Body excluded
    penalty(4ch): [S_target, A_oar, B_boundary, final_total->P_fused]
    dose        : RTDose ground truth (target)

Training is single-process (no DDP) and shows live tqdm logs: per-batch
loss + loss components (base / router_mean / router_tv / grad / hf) and
LR; per-epoch summary + optional body-region MAE validation.

Example
-------
    cd /root/dose-zjx/dose_2
    python scripts/dose_train_3d_v2_0_lung.py \
        --data_root_train /root/autodl-tmp/lung_cancer_processed/processed/training \
        --data_root_val   /root/autodl-tmp/lung_cancer_processed/processed/val \
        --epoch 600 --bs 1 --gpu 0
"""
import sys

sys.path.append("../")
sys.path.append("./")

import os
import argparse
import contextlib

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

from guided_diffusion.dose_loader_3d_lung import (
    Dataset_Lung_v2_0,
    DEFAULT_LUNG_PENALTY_FILES,
    DEFAULT_LUNG_PENALTY_NAMES,
)
from guided_diffusion.unet_3d_v2_0 import (
    UNetModel_DynamicConstraintRouter_v2_0,
    resolve_v2_mode,
    V2_MODES,
)
from guided_diffusion.unet_3d_v1_5 import DEFAULT_RISK_USE_CHANNELS
from flow_matching_v2_0 import FlowMatchingV20


# ----------------------------- CLI ARGS -----------------------------
def build_args():
    p = argparse.ArgumentParser(description="Single-GPU v2.0 lung dose training.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--epoch", type=int, default=600)
    p.add_argument("--num_workers", type=int, default=4)

    # schedule
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--lr_max", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # main UNet
    p.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--model_channels", type=int, default=32)
    p.add_argument("--channel_mult", type=int, nargs="+", default=[1, 2, 4, 4])

    # penalty branch
    p.add_argument("--penalty_files", type=str, nargs="*", default=list(DEFAULT_LUNG_PENALTY_FILES))
    p.add_argument("--penalty_channel_names", type=str, nargs="*", default=list(DEFAULT_LUNG_PENALTY_NAMES))
    p.add_argument("--penalty_model_channels", type=int, default=None)
    p.add_argument("--penalty_num_res_blocks", type=int, default=1)
    p.add_argument("--penalty_dropout_p", type=float, default=0.2)
    p.add_argument("--disable_penalty_dropout", action="store_true")

    # v1.5 risk prior
    p.add_argument("--disable_risk_aware_injection", action="store_true")
    p.add_argument("--risk_base", type=float, default=0.2)
    p.add_argument("--risk_use_channels", type=str, nargs="+", default=list(DEFAULT_RISK_USE_CHANNELS))
    p.add_argument("--risk_hard", action="store_true")

    # v2.0 modules / ablation
    p.add_argument("--mode", type=str, choices=list(V2_MODES), default="full_v2_0")
    p.add_argument("--disable_constraint_compiler", action="store_true")
    p.add_argument("--disable_dynamic_router", action="store_true")
    p.add_argument("--disable_relation_embedding", action="store_true")
    p.add_argument("--compiler_out_channels", type=int, default=4)
    p.add_argument("--compiler_width", type=int, default=None)
    p.add_argument("--compiler_num_res_blocks", type=int, default=2)
    p.add_argument("--relation_dim", type=int, default=128)
    p.add_argument("--compiler_use_cond_context", action="store_true")
    p.add_argument("--router_time_independent", action="store_true")
    p.add_argument("--router_delta_scale", type=float, default=0.5)
    p.add_argument("--router_hidden", type=int, default=None)

    # v2.0 optional losses
    p.add_argument("--lambda_grad_loss", type=float, default=0.0)
    p.add_argument("--lambda_hf_loss", type=float, default=0.0)
    p.add_argument("--router_reg_weight", type=float, default=0.0)
    p.add_argument("--router_smooth_weight", type=float, default=0.0)
    p.add_argument("--disable_morphology_loss", action="store_true")
    p.add_argument("--hf_kernel", type=int, default=5)
    # Additive body-L1 (improves Dose Score without disturbing the
    # structure-weighted MSE / DVH). Default 0 == original behaviour.
    p.add_argument("--lambda_global_l1", type=float, default=0.0)
    p.add_argument("--global_l1_whole_volume", action="store_true",
                   help="apply global L1 over the whole patch instead of the CT-derived body mask")

    # data
    p.add_argument("--data_root_train", type=str,
                   default="/root/autodl-tmp/lung_cancer_processed/processed/training")
    p.add_argument("--data_root_val", type=str,
                   default="/root/autodl-tmp/lung_cancer_processed/processed/val")

    # EMA / resume / save / val
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume_ema", type=str, default="")
    p.add_argument("--resume_epoch", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="")
    p.add_argument("--save_name_suffix", type=str, default="")
    p.add_argument("--val_every", type=int, default=25)
    p.add_argument("--val_steps", type=int, default=20)
    p.add_argument("--save_every", type=int, default=50)
    return p.parse_args()


# ----------------------------- EMA -----------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.buffers = {n: b.detach().clone() for n, b in model.named_buffers() if b.is_floating_point()}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)
        for n, b in model.named_buffers():
            if b.is_floating_point():
                self.buffers[n] = b.detach().clone()

    @contextlib.contextmanager
    def average_parameters(self, model):
        backup = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        try:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.requires_grad and n in self.shadow:
                        p.data.copy_(self.shadow[n])
            yield
        finally:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.requires_grad and n in backup:
                        p.data.copy_(backup[n])

    def state_dict_for_save(self, model):
        sd = {k: v.detach().clone() for k, v in self.shadow.items()}
        for n, b in model.named_buffers():
            sd[n] = b.detach().clone()
        return sd


def main():
    args = build_args()

    mode, use_compiler, use_router = resolve_v2_mode(
        args.mode,
        use_constraint_compiler=not args.disable_constraint_compiler,
        use_dynamic_router=not args.disable_dynamic_router,
    )
    use_relation = (not args.disable_relation_embedding) and use_compiler
    if args.disable_morphology_loss:
        args.lambda_grad_loss = 0.0
        args.lambda_hf_loss = 0.0

    patch_size = tuple(args.patch_size)
    if any(s % 16 != 0 for s in patch_size):
        raise ValueError(f"patch_size 必须为 16 的倍数, 当前: {patch_size}")
    if args.model_channels % 32 != 0:
        raise ValueError(f"--model_channels 必须为 32 的倍数, 当前: {args.model_channels}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    print(f"[Device] {device}")

    penalty_channels = len(args.penalty_files)

    # ----------------------------- DATA -----------------------------
    train_data = Dataset_Lung_v2_0(
        data_root=args.data_root_train,
        patch_size=patch_size,
        is_train=True,
        return_body=False,
        penalty_files=args.penalty_files,
        penalty_channel_names=args.penalty_channel_names,
    )
    train_loader = DataLoader(
        train_data, batch_size=args.bs, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    val_enabled = bool(args.data_root_val) and args.val_every > 0 and os.path.isdir(args.data_root_val)
    if val_enabled:
        val_data = Dataset_Lung_v2_0(
            data_root=args.data_root_val,
            patch_size=patch_size,
            is_train=False,
            return_body=True,  # body returned for eval-only MAE
            penalty_files=args.penalty_files,
            penalty_channel_names=args.penalty_channel_names,
        )
        val_loader = DataLoader(
            val_data, batch_size=1, shuffle=False,
            num_workers=max(1, args.num_workers // 2), pin_memory=True,
        )
    else:
        val_loader = None

    # ----------------------------- SAVE DIR -----------------------------
    save_name = (
        f"v2_0_lung_{mode}_mc{args.model_channels}_pc{penalty_channels}"
        f"_bs{args.bs}_epoch{args.epoch}"
    )
    if args.save_name_suffix:
        save_name += "_" + args.save_name_suffix
    save_dir = args.save_dir or os.path.join("trained_models", "v2_0_lung", save_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Save] checkpoints -> {save_dir}")

    # ----------------------------- MODEL -----------------------------
    use_risk = not args.disable_risk_aware_injection
    model = UNetModel_DynamicConstraintRouter_v2_0(
        image_size=patch_size,
        in_channels=1, ct_channels=1, syn_channels=1, dis_channels=11,
        model_channels=args.model_channels, out_channels=1,
        num_res_blocks=2, attention_resolutions=(8, 16),
        channel_mult=tuple(args.channel_mult), dims=3,
        use_checkpoint=True, use_fp16=False,
        penalty_channels=penalty_channels,
        penalty_model_channels=args.penalty_model_channels,
        penalty_num_res_blocks=args.penalty_num_res_blocks,
        use_penalty_dropout=(not args.disable_penalty_dropout),
        penalty_dropout_p=args.penalty_dropout_p,
        use_risk_aware_injection=use_risk,
        risk_base=args.risk_base,
        risk_use_channels=tuple(args.risk_use_channels),
        risk_soft=(not args.risk_hard),
        penalty_channel_names=tuple(args.penalty_channel_names),
        mode=mode,
        use_relation_embedding=use_relation,
        compiler_out_channels=args.compiler_out_channels,
        compiler_width=args.compiler_width,
        compiler_num_res_blocks=args.compiler_num_res_blocks,
        relation_dim=args.relation_dim,
        compiler_use_cond_context=args.compiler_use_cond_context,
        router_time_dependent=(not args.router_time_independent),
        router_delta_scale=args.router_delta_scale,
        router_hidden=args.router_hidden,
    ).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_adapter = sum(p.numel() for p in model.adapter_parameters())
    print(f"[Model] v2.0 mode={mode} | total={n_total/1e6:.2f}M | "
          f"adapter+compiler+routers={n_adapter/1e6:.2f}M")
    print(f"[v2.0] compiler={use_compiler} router={use_router} relation={use_relation} "
          f"risk_base={args.risk_base} risk_use={args.risk_use_channels}")

    # ----- optional resume (strict=False) -----
    start_epoch = args.resume_epoch
    if args.resume_from:
        if not os.path.isfile(args.resume_from):
            raise FileNotFoundError(f"--resume_from not found: {args.resume_from}")
        sd = torch.load(args.resume_from, map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[Resume] from {args.resume_from} (strict=False) "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        del sd

    flow_model = FlowMatchingV20(
        model,
        lambda_grad_loss=args.lambda_grad_loss,
        lambda_hf_loss=args.lambda_hf_loss,
        router_reg_weight=args.router_reg_weight,
        router_smooth_weight=args.router_smooth_weight,
        hf_kernel=args.hf_kernel,
        lambda_global_l1=args.lambda_global_l1,
        global_l1_use_body_mask=(not args.global_l1_whole_volume),
    )

    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None and args.resume_ema and os.path.isfile(args.resume_ema):
        es = torch.load(args.resume_ema, map_location="cpu")
        n = 0
        for name, p in ema.shadow.items():
            if name in es:
                p.copy_(es[name].to(p.device)); n += 1
        print(f"[Resume] EMA restored {n}/{len(ema.shadow)} params")
        del es

    # ----------------------------- OPTIM + LR -----------------------------
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr_max, weight_decay=args.weight_decay,
    )
    total_epochs = args.epoch
    warmup_epochs = min(max(1, int(total_epochs * args.warmup_ratio)), max(1, total_epochs - 1))
    cosine_epochs = max(1, total_epochs - warmup_epochs)
    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=args.min_lr),
        ],
        milestones=[warmup_epochs],
    )
    for _ in range(start_epoch):
        lr_scheduler.step()
    print(f"[LR] LinearWarmup({warmup_epochs}) + Cosine({cosine_epochs}), "
          f"lr_max={args.lr_max}, min_lr={args.min_lr}, start_epoch={start_epoch}")

    # ----------------------------- VAL -----------------------------
    @torch.no_grad()
    def evaluate(use_ema):
        if not val_enabled:
            return float("nan"), float("nan")
        model.eval()
        cm = ema.average_parameters(model) if (use_ema and ema is not None) else contextlib.nullcontext()
        raw = flow_model._raw_net()
        mae_sum, mae_cnt = 0.0, 0
        risk_sum, risk_cnt = 0.0, 0
        with cm:
            for batch in tqdm(val_loader, desc="[Val]", leave=False):
                ct, syn, dis, penalty, dose_gt, body = [b.to(device).float() for b in batch]
                pred = flow_model.sample(ct, syn, dis, penalty, steps=args.val_steps)
                pred_gy = torch.clamp((pred + 1.0) * 40.0, 0.0, 80.0)
                gt_gy = torch.clamp((dose_gt + 1.0) * 40.0, 0.0, 80.0)
                risk_map = getattr(raw, "_last_m_risk", None)
                for b in range(ct.shape[0]):
                    bm = body[b]
                    s = bm.sum()
                    if s.item() > 0:
                        mae_sum += ((pred_gy[b] - gt_gy[b]).abs() * bm).sum().item() / s.item()
                        mae_cnt += 1
                    if risk_map is not None:
                        rm = (risk_map[b] > 0.5).float()
                        rs = rm.sum()
                        if rs.item() > 0:
                            risk_sum += ((pred_gy[b] - gt_gy[b]).abs() * rm).sum().item() / rs.item()
                            risk_cnt += 1
        model.train()
        mae = mae_sum / mae_cnt if mae_cnt > 0 else float("inf")
        risk_mae = risk_sum / risk_cnt if risk_cnt > 0 else float("nan")
        return mae, risk_mae

    # ----------------------------- TRAIN LOOP -----------------------------
    best_mae = float("inf")
    best_epoch = -1
    grad_accum = max(1, args.grad_accum_steps)
    n_batches = len(train_loader)

    epoch_bar = tqdm(range(start_epoch, total_epochs), desc="Epochs", position=0)
    for epoch in epoch_bar:
        model.train()
        cur_lr = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad(set_to_none=True)

        losses = []
        comp_acc = {}
        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}",
                         position=1, leave=False)
        for i, batch in enumerate(batch_bar):
            ct, syn, dis, penalty, dose = [b.to(device, non_blocking=True).float() for b in batch]
            loss = flow_model.get_loss(dose, ct, syn, dis, penalty)
            (loss / grad_accum).backward()

            if (i + 1) % grad_accum == 0 or (i + 1) == n_batches:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)

            losses.append(loss.item())
            for k, v in flow_model.last_components.items():
                comp_acc.setdefault(k, []).append(float(v))

            # ---- live tqdm log ----
            post = {"loss": f"{loss.item():.4f}", "lr": f"{cur_lr:.2e}"}
            for k in ("base", "global_l1", "router_mean", "router_tv", "grad", "hf"):
                if k in flow_model.last_components:
                    post[k] = f"{float(flow_model.last_components[k]):.4f}"
            batch_bar.set_postfix(post)
        batch_bar.close()

        lr_scheduler.step()
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        comp_str = "  ".join(f"{k}={np.mean(v):.4f}" for k, v in comp_acc.items() if k != "total")
        epoch_bar.write(f"[Epoch {epoch+1}/{total_epochs}] lr={cur_lr:.2e} "
                        f"train_loss={mean_loss:.4f}  [{comp_str}]")

        run_val = val_enabled and ((epoch + 1) % args.val_every == 0)
        if run_val:
            val_mae, val_risk = evaluate(use_ema=(ema is not None))
            tag = "EMA" if ema is not None else "raw"
            epoch_bar.write(f"  [Val] epoch {epoch+1}  MAE({tag})={val_mae:.4f} Gy  "
                            f"risk-MAE={val_risk:.4f} Gy  (best={best_mae:.4f} @ep{best_epoch})")
            epoch_bar.set_postfix(val_mae=f"{val_mae:.3f}", best=f"{best_mae:.3f}")
            if val_mae < best_mae:
                best_mae, best_epoch = val_mae, epoch + 1
                torch.save(model.state_dict(), os.path.join(save_dir, "model_best_mae.pth"))
                if ema is not None:
                    torch.save(ema.state_dict_for_save(model), os.path.join(save_dir, "ema_best_mae.pth"))
                epoch_bar.write(f"  [Val] *new best* MAE={best_mae:.4f} Gy @ep{best_epoch} -> saved")

        if (epoch + 1) % max(1, args.save_every) == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch{epoch+1}.pth"))
            if ema is not None:
                torch.save(ema.state_dict_for_save(model), os.path.join(save_dir, f"ema_epoch{epoch+1}.pth"))

    torch.save(model.state_dict(), os.path.join(save_dir, "model_final.pth"))
    if ema is not None:
        torch.save(ema.state_dict_for_save(model), os.path.join(save_dir, "ema_final.pth"))
    print(f"Training finished. Best val MAE = {best_mae:.4f} Gy @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
