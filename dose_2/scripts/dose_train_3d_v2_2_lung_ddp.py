#!/usr/bin/env python3
"""
dose_train_3d_v2_2_lung_ddp.py
================================
2-GPU (DDP) v2.2 lung training (compiler: basis gating + bounded boundary).

Same loss as v2.0 (``FlowMatchingV22`` / structure-weighted MSE). No new loss terms.

Launch (2 GPUs):
    cd /root/dose-zjx/dose_2
    torchrun --nproc_per_node=2 scripts/dose_train_3d_v2_2_lung_ddp.py ...
or
    bash scripts/train_v2_2_lung_2gpu.sh
"""
import sys

sys.path.append("../")
sys.path.append("./")

import os
import argparse
import contextlib

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as Data
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

from guided_diffusion.dose_loader_3d_lung import (
    Dataset_Lung_v2_0,
    DEFAULT_LUNG_PENALTY_FILES,
    DEFAULT_LUNG_PENALTY_NAMES,
)
from guided_diffusion.unet_3d_v2_2 import (
    UNetModel_DynamicConstraintRouter_v2_2,
    resolve_v2_2_mode,
    V2_2_MODES,
)
from guided_diffusion.constraint_field_compiler_v2_2 import V22_GATE_MODES
from guided_diffusion.unet_3d_v1_5 import DEFAULT_RISK_USE_CHANNELS
from flow_matching_v2_2 import FlowMatchingV22
from warmstart_utils import load_warmstart_checkpoint, adapter_param_ids, mask_backbone_grads


def build_args():
    p = argparse.ArgumentParser(description="DDP v2.2 lung training (2-GPU).")
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--epoch", type=int, default=400)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--local_rank", type=int, default=-1)

    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--lr_max", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--model_channels", type=int, default=32)
    p.add_argument("--channel_mult", type=int, nargs="+", default=[1, 2, 4, 4])

    p.add_argument("--penalty_files", type=str, nargs="*", default=list(DEFAULT_LUNG_PENALTY_FILES))
    p.add_argument("--penalty_channel_names", type=str, nargs="*", default=list(DEFAULT_LUNG_PENALTY_NAMES))
    p.add_argument("--penalty_model_channels", type=int, default=None)
    p.add_argument("--penalty_num_res_blocks", type=int, default=1)
    p.add_argument("--penalty_dropout_p", type=float, default=0.2)
    p.add_argument("--disable_penalty_dropout", action="store_true")

    p.add_argument("--disable_risk_aware_injection", action="store_true")
    p.add_argument("--risk_base", type=float, default=0.2)
    p.add_argument("--risk_use_channels", type=str, nargs="+", default=list(DEFAULT_RISK_USE_CHANNELS))
    p.add_argument("--risk_hard", action="store_true")

    p.add_argument("--mode", type=str, choices=list(V2_2_MODES), default="full_v2_2")
    p.add_argument("--compiler_gate_mode", type=str, choices=list(V22_GATE_MODES), default="gate")
    p.add_argument("--compiler_boundary_eps", type=float, default=1e-3)
    p.add_argument("--freeze_backbone_epochs", type=int, default=0)
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

    p.add_argument("--lambda_grad_loss", type=float, default=0.0)
    p.add_argument("--lambda_hf_loss", type=float, default=0.0)
    p.add_argument("--hf_kernel", type=int, default=5)
    p.add_argument("--router_reg_weight", type=float, default=0.0)
    p.add_argument("--router_smooth_weight", type=float, default=0.0)
    p.add_argument("--disable_morphology_loss", action="store_true")
    p.add_argument("--lambda_global_l1", type=float, default=0.0)
    p.add_argument("--global_l1_whole_volume", action="store_true")

    p.add_argument("--data_root_train", type=str,
                   default="/root/autodl-tmp/lung_cancer_processed/processed/training")
    p.add_argument("--data_root_val", type=str,
                   default="/root/autodl-tmp/lung_cancer_processed/processed/val")

    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--reset_compiler", action="store_true",
                   help="warm-start backbone/adapter/router but keep compiler randomly initialized")
    p.add_argument("--resume_ema", type=str, default="")
    p.add_argument("--resume_epoch", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="")
    p.add_argument("--save_name_suffix", type=str, default="")
    p.add_argument("--val_every", type=int, default=50)
    p.add_argument("--val_steps", type=int, default=20)
    p.add_argument("--save_every", type=int, default=50)
    return p.parse_args()


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


args = build_args()

mode, use_compiler, use_router = resolve_v2_2_mode(
    args.mode,
    use_constraint_compiler=not args.disable_constraint_compiler,
    use_dynamic_router=not args.disable_dynamic_router,
)
use_relation = (not args.disable_relation_embedding) and use_compiler
if args.disable_morphology_loss:
    args.lambda_grad_loss = 0.0
    args.lambda_hf_loss = 0.0

if args.local_rank == -1:
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend="nccl", init_method="env://")
device = torch.device("cuda", args.local_rank)
world_size = dist.get_world_size()
rank = dist.get_rank()
is_main = (rank == 0)

if is_main:
    print(f"[rank {rank}] LOCAL_RANK={args.local_rank} device={device} world_size={world_size}", flush=True)

patch_size = tuple(args.patch_size)
if any(s % 16 != 0 for s in patch_size):
    raise ValueError(f"patch_size 必须为 16 的倍数, 当前: {patch_size}")
if args.model_channels % 32 != 0:
    raise ValueError(f"--model_channels 必须为 32 的倍数, 当前: {args.model_channels}")

penalty_channels = len(args.penalty_files)
start_epoch = args.resume_epoch
total_epochs = args.epoch

save_name = (
    f"v2_2_lung_{mode}_gm{args.compiler_gate_mode}_mc{args.model_channels}_pc{penalty_channels}"
    f"_bs{world_size * args.bs}_epoch{total_epochs}"
)
if args.save_name_suffix:
    save_name += "_" + args.save_name_suffix
save_dir = args.save_dir or os.path.join("trained_models", "v2_2_lung", save_name)
if is_main:
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Save] checkpoints -> {save_dir}", flush=True)

train_data = Dataset_Lung_v2_0(
    data_root=args.data_root_train,
    patch_size=patch_size,
    is_train=True,
    return_body=False,
    penalty_files=args.penalty_files,
    penalty_channel_names=args.penalty_channel_names,
)
train_sampler = Data.distributed.DistributedSampler(train_data)
train_loader = DataLoader(
    train_data, batch_size=args.bs, sampler=train_sampler,
    shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True,
)

val_enabled = bool(args.data_root_val) and args.val_every > 0 and os.path.isdir(args.data_root_val)
if val_enabled:
    val_data = Dataset_Lung_v2_0(
        data_root=args.data_root_val,
        patch_size=patch_size,
        is_train=False,
        return_body=True,
        penalty_files=args.penalty_files,
        penalty_channel_names=args.penalty_channel_names,
    )
    val_sampler = Data.distributed.DistributedSampler(val_data, shuffle=False)
    val_loader = DataLoader(
        val_data, batch_size=1, sampler=val_sampler,
        shuffle=False, num_workers=max(1, args.num_workers // 2), pin_memory=True,
    )
else:
    val_loader = None

use_risk = not args.disable_risk_aware_injection
model = UNetModel_DynamicConstraintRouter_v2_2(
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
    compiler_gate_mode=args.compiler_gate_mode,
    compiler_boundary_eps=args.compiler_boundary_eps,
).to(device)

if is_main:
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[Model] v2.2 DDP mode={mode} | total={n_total/1e6:.2f}M", flush=True)
    print(f"[v2.2] compiler=v22(gate={args.compiler_gate_mode}) router={use_router} "
          f"relation={use_relation} freeze_backbone_epochs={args.freeze_backbone_epochs} "
          f"reset_compiler={args.reset_compiler}", flush=True)

if args.resume_from:
    if not os.path.isfile(args.resume_from):
        raise FileNotFoundError(f"--resume_from not found: {args.resume_from}")
    load_warmstart_checkpoint(
        model, args.resume_from, reset_compiler=args.reset_compiler,
        log_fn=print if is_main else lambda *a, **k: None,
    )
elif is_main:
    print("[Resume] none -> training FROM SCRATCH", flush=True)

flow_model = FlowMatchingV22(
    model,
    lambda_grad_loss=args.lambda_grad_loss,
    lambda_hf_loss=args.lambda_hf_loss,
    router_reg_weight=args.router_reg_weight,
    router_smooth_weight=args.router_smooth_weight,
    hf_kernel=args.hf_kernel,
    lambda_global_l1=args.lambda_global_l1,
    global_l1_use_body_mask=(not args.global_l1_whole_volume),
)

ddp_model = DDP(
    model,
    device_ids=[args.local_rank],
    output_device=args.local_rank,
    find_unused_parameters=False,
)
flow_model.net = ddp_model

ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
if ema is not None and args.resume_ema and os.path.isfile(args.resume_ema) and not args.reset_compiler:
    es = torch.load(args.resume_ema, map_location="cpu")
    with torch.no_grad():
        n = 0
        for name, p in ema.shadow.items():
            if name in es:
                p.copy_(es[name].to(p.device))
                n += 1
    if is_main:
        print(f"[Resume] EMA restored {n}/{len(ema.shadow)} params", flush=True)
    del es
elif ema is not None and args.reset_compiler and is_main:
    print("[Resume] reset_compiler=True -> EMA shadow uses fresh compiler weights", flush=True)

optimizer = optim.AdamW(
    [p for p in ddp_model.parameters() if p.requires_grad],
    lr=args.lr_max, weight_decay=args.weight_decay,
)
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
if is_main:
    print(f"[LR] LinearWarmup({warmup_epochs}) + Cosine({cosine_epochs}), "
          f"lr_max={args.lr_max}, min_lr={args.min_lr}", flush=True)


@torch.no_grad()
def evaluate_metrics(use_ema):
    if not val_enabled:
        return float("nan"), float("nan")

    ddp_model.eval()
    cm = ema.average_parameters(model) if (use_ema and ema is not None) else contextlib.nullcontext()
    raw = flow_model._raw_net()

    mae_sum = torch.zeros(1, device=device)
    mae_count = torch.zeros(1, device=device)
    risk_sum = torch.zeros(1, device=device)
    risk_count = torch.zeros(1, device=device)

    with cm:
        iterator = val_loader
        if is_main:
            iterator = tqdm(val_loader, desc="[Val]", leave=False)

        for batch in iterator:
            ct, syn, dis, penalty, dose_gt, body = [b.to(device).float() for b in batch]
            pred = flow_model.sample(ct, syn, dis, penalty, steps=args.val_steps)
            pred_gy = torch.clamp((pred + 1.0) * 40.0, 0.0, 80.0)
            gt_gy = torch.clamp((dose_gt + 1.0) * 40.0, 0.0, 80.0)
            risk_map = getattr(raw, "_last_m_risk", None)

            for b in range(ct.shape[0]):
                bm = body[b]
                s = bm.sum()
                if s.item() > 0:
                    mae_sum += ((pred_gy[b] - gt_gy[b]).abs() * bm).sum() / s
                    mae_count += 1
                if risk_map is not None:
                    rm = (risk_map[b] > 0.5).float()
                    rs = rm.sum()
                    if rs.item() > 0:
                        risk_sum += ((pred_gy[b] - gt_gy[b]).abs() * rm).sum() / rs
                        risk_count += 1

    for t in (mae_sum, mae_count, risk_sum, risk_count):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    ddp_model.train()

    mae = (mae_sum / mae_count).item() if mae_count.item() > 0 else float("inf")
    risk_mae = (risk_sum / risk_count).item() if risk_count.item() > 0 else float("nan")
    return mae, risk_mae


best_mae = float("inf")
best_epoch = -1
grad_accum = max(1, args.grad_accum_steps)
n_batches = len(train_loader)
adapter_ids = adapter_param_ids(model)

for epoch in range(start_epoch, total_epochs):
    backbone_frozen = args.freeze_backbone_epochs > 0 and epoch < args.freeze_backbone_epochs
    if backbone_frozen is False and epoch == args.freeze_backbone_epochs and args.freeze_backbone_epochs > 0 and is_main:
        print(f"[Freeze] backbone grad enabled at epoch {epoch + 1}", flush=True)
    train_sampler.set_epoch(epoch)
    current_lr = optimizer.param_groups[0]["lr"]
    ddp_model.train()
    optimizer.zero_grad(set_to_none=True)

    train_losses = []
    comp_acc = {}
    loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}") if is_main else train_loader

    for i, batch in enumerate(loader):
        ct, syn, dis, penalty, dose = [b.to(device, non_blocking=True).float() for b in batch]
        loss_raw = flow_model.get_loss(dose, ct, syn, dis, penalty)
        (loss_raw / grad_accum).backward()
        if backbone_frozen:
            mask_backbone_grads(model, adapter_ids)

        loss_detached = loss_raw.detach()
        dist.all_reduce(loss_detached, op=dist.ReduceOp.SUM)
        loss_detached = loss_detached / world_size
        train_losses.append(loss_detached.item())

        for k, v in flow_model.last_components.items():
            comp_acc.setdefault(k, []).append(float(v))

        if (i + 1) % grad_accum == 0 or (i + 1) == n_batches:
            clip_params = (
                [p for p in model.adapter_parameters() if p.grad is not None]
                if backbone_frozen
                else [p for p in ddp_model.parameters() if p.grad is not None]
            )
            torch.nn.utils.clip_grad_norm_(clip_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        if is_main:
            post = {"loss": f"{loss_detached.item():.4f}", "lr": f"{current_lr:.2e}"}
            for k in ("base", "global_l1", "router_mean", "router_tv", "grad", "hf"):
                if k in flow_model.last_components:
                    post[k] = f"{float(flow_model.last_components[k]):.4f}"
            loader.set_postfix(post)

    lr_scheduler.step()

    if is_main:
        mean_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        comp_str = "  ".join(f"{k}={np.mean(v):.4f}" for k, v in comp_acc.items() if k != "total")
        print(f"Epoch {epoch+1}/{total_epochs}  lr={current_lr:.2e}  train_loss={mean_loss:.4f}  [{comp_str}]", flush=True)

    if val_enabled and ((epoch + 1) % args.val_every == 0):
        val_mae, val_risk = evaluate_metrics(use_ema=(ema is not None))
        if is_main:
            tag = "EMA" if ema is not None else "raw"
            print(f"  [Val] epoch {epoch+1}  MAE({tag})={val_mae:.4f} Gy  "
                  f"risk-MAE={val_risk:.4f} Gy  (best={best_mae:.4f} @ep{best_epoch})", flush=True)
            if val_mae < best_mae:
                best_mae, best_epoch = val_mae, epoch + 1
                torch.save(model.state_dict(), os.path.join(save_dir, "model_best_mae.pth"))
                if ema is not None:
                    torch.save(ema.state_dict_for_save(model), os.path.join(save_dir, "ema_best_mae.pth"))
                print(f"  [Val] *new best* MAE={best_mae:.4f} Gy @ep{best_epoch} -> saved", flush=True)

    if is_main and ((epoch + 1) % max(1, args.save_every) == 0):
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch{epoch+1}.pth"))
        if ema is not None:
            torch.save(ema.state_dict_for_save(model), os.path.join(save_dir, f"ema_epoch{epoch+1}.pth"))

if is_main:
    torch.save(model.state_dict(), os.path.join(save_dir, "model_final.pth"))
    if ema is not None:
        torch.save(ema.state_dict_for_save(model), os.path.join(save_dir, "ema_final.pth"))
    print(f"Training finished. Best val MAE = {best_mae:.4f} Gy @ epoch {best_epoch}", flush=True)

dist.destroy_process_group()
