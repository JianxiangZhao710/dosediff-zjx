import sys
sys.path.append("../")
sys.path.append("./")

import os
import argparse
import contextlib
import torch
import numpy as np
import torch.distributed as dist
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

from guided_diffusion.dose_loader_3d import Dataset_PSDM_3D_Train
from guided_diffusion.unet_3d import UNetModel_MS_Former_3D
from guided_diffusion.unet_3d_v1_1 import UNetModel_GatedXQueryViT_3D
from flow_matching import FlowMatching


# Model registry: model_name -> class
MODEL_REGISTRY = {
    'v1': UNetModel_MS_Former_3D,                       # baseline: addition fusion + Q=CT,K=DIS,V=X ViT
    'v1_1_gated_xquery_vit': UNetModel_GatedXQueryViT_3D,  # gated fusion + Q=X,K/V=condition ViT
}


# ----------------------------- CLI ARGS -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default="0", help='which gpu is used (informational, controlled by CUDA_VISIBLE_DEVICES)')
parser.add_argument('--bs', type=int, default=1, help='batch size per gpu')
parser.add_argument('--epoch', type=int, default=600, help='total epochs')
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--steps", type=int, default=50, help='legacy arg, sampling steps (use --val_steps instead)')

# Training schedule
parser.add_argument("--grad_accum_steps", type=int, default=4, help="gradient accumulation steps (effective batch = world_size * bs * grad_accum_steps)")
parser.add_argument("--warmup_ratio", type=float, default=0.05, help="warmup ratio for total epochs")
parser.add_argument("--lr_max", type=float, default=1e-4, help="peak learning rate after warmup")
parser.add_argument("--min_lr", type=float, default=1e-6, help="minimum lr for cosine annealing")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient norm clip")

# Model
parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 128, 128], help="patch size (D H W), must be multiples of 16")
parser.add_argument("--model_channels", type=int, default=64, help="base channel count of UNet (must be multiple of 32 for GroupNorm)")
parser.add_argument("--model_name", type=str, default="v1", choices=list(MODEL_REGISTRY.keys()),
                    help="which velocity-field network to use. v1=baseline (UNetModel_MS_Former_3D), "
                         "v1_1_gated_xquery_vit=gated encoder + X-query ViT (UNetModel_GatedXQueryViT_3D)")

# Data paths
parser.add_argument("--data_root_train", type=str,
                    default='/data0/zhaojianxiang/preprocessed_data/train-pats_preprocess/',
                    help="OpenKBP training set (with Mask_*.nii.gz + dose.nii.gz)")
parser.add_argument("--data_root_val", type=str,
                    default='/data0/zhaojianxiang/preprocessed_data/validation-pats_preprocess/',
                    help="OpenKBP validation set; pass empty string to disable validation")

# EMA
parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate (set 0 to disable EMA)")

# Resume
parser.add_argument("--resume_from", type=str, default="",
                    help="path to model state_dict (e.g. model_best_mae.pth or ema_best_mae.pth). "
                         "Loaded into the trainable model with strict=False")
parser.add_argument("--resume_ema", type=str, default="",
                    help="path to EMA state_dict. If empty and --resume_from is given, EMA is initialized from --resume_from")
parser.add_argument("--save_name_suffix", type=str, default="",
                    help="append suffix to save_dir to avoid overwriting previous runs")

# Validation / checkpointing
parser.add_argument("--val_every", type=int, default=25, help="run validation every N epochs (0 disables)")
parser.add_argument("--val_steps", type=int, default=10, help="Flow Matching Euler steps for validation sampling")
parser.add_argument("--save_every", type=int, default=25, help="save checkpoint every N epochs")

args = parser.parse_args()


# ----------------------------- EMA UTIL -----------------------------
class EMA:
    """Lightweight Polyak / exponential moving average of model parameters.

    Maintains a CPU-or-GPU shadow copy of each trainable parameter and (if any)
    persistent buffer. Updated after every optimizer.step().
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999, device=None):
        self.decay = decay
        self.device = device
        self.shadow_params = {}
        self.shadow_buffers = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow_params[name] = p.detach().clone()
        # store float buffers too (e.g. running stats); int buffers are skipped
        for name, b in model.named_buffers():
            if b.is_floating_point():
                self.shadow_buffers[name] = b.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            s = self.shadow_params.get(name)
            if s is None:
                continue
            s.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for name, b in model.named_buffers():
            if not b.is_floating_point():
                continue
            s = self.shadow_buffers.get(name)
            if s is None:
                continue
            s.copy_(b.detach())  # buffers tracked as-is

    @contextlib.contextmanager
    def average_parameters(self, model: torch.nn.Module):
        """Temporarily swap model weights with EMA weights; restore on exit."""
        backup_params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        try:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.requires_grad and n in self.shadow_params:
                        p.data.copy_(self.shadow_params[n])
            yield
        finally:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.requires_grad and n in backup_params:
                        p.data.copy_(backup_params[n])

    def state_dict_for_save(self, model: torch.nn.Module = None):
        """Return a state_dict containing EMA params (and current model buffers if model is given).

        With model given, the returned dict is fully compatible with
        ``model.load_state_dict(..., strict=True)`` for inference.
        """
        sd = {k: v.detach().clone() for k, v in self.shadow_params.items()}
        if model is not None:
            for name, b in model.named_buffers():
                sd[name] = b.detach().clone()
        return sd


# ----------------------------- DISTRIBUTED SETUP -----------------------------
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


# ----------------------------- BASIC CHECKS -----------------------------
patch_size = tuple(args.patch_size)
if any(s % 16 != 0 for s in patch_size):
    raise ValueError(f"patch_size 必须为 16 的倍数, 当前: {patch_size}")
if args.model_channels % 32 != 0:
    raise ValueError(f"--model_channels 必须为 32 的倍数 (GroupNorm32), 当前: {args.model_channels}")

train_bs = args.bs
all_epochs = args.epoch

save_name = 'MedSegDiff_Flow_3D_OpenKBP_11ch_mc{}_bs{}_epoch{}'.format(
    args.model_channels, world_size * args.bs, args.epoch
)
if args.save_name_suffix:
    save_name = save_name + '_' + args.save_name_suffix
# v1 keeps the legacy flat layout (backwards compatible);
# other model variants are namespaced under their model_name to avoid overlap.
if args.model_name == 'v1':
    save_dir = os.path.join('trained_models', save_name)
else:
    save_dir = os.path.join('trained_models', args.model_name, save_name)
if is_main:
    os.makedirs(save_dir, exist_ok=True)


# ----------------------------- DATA -----------------------------
train_data = Dataset_PSDM_3D_Train(data_root=args.data_root_train, patch_size=patch_size)
train_sampler = Data.distributed.DistributedSampler(train_data)
train_dataloader = DataLoader(
    train_data, batch_size=train_bs, sampler=train_sampler,
    shuffle=False, num_workers=4, pin_memory=True,
)

val_enabled = bool(args.data_root_val) and args.val_every > 0 and os.path.isdir(args.data_root_val)
if val_enabled:
    val_data = Dataset_PSDM_3D_Train(data_root=args.data_root_val, patch_size=patch_size)
    val_sampler = Data.distributed.DistributedSampler(val_data, shuffle=False)
    val_dataloader = DataLoader(
        val_data, batch_size=1, sampler=val_sampler,
        shuffle=False, num_workers=2, pin_memory=True,
    )
else:
    val_dataloader = None

if is_main:
    print(f"Train size: {len(train_data)}", flush=True)
    if val_enabled:
        print(f"Val   size: {len(val_data)}  (val_every={args.val_every} epochs, val_steps={args.val_steps})", flush=True)
    else:
        print(f"Val disabled (data_root_val='{args.data_root_val}', val_every={args.val_every})", flush=True)


# ----------------------------- MODEL -----------------------------
dis_channels = 11  # 11 OpenKBP masks
ModelClass = MODEL_REGISTRY[args.model_name]
if is_main:
    print(f"[Model] using model_name='{args.model_name}' -> {ModelClass.__name__}", flush=True)

model = ModelClass(
    image_size=patch_size,
    in_channels=1,
    ct_channels=1,
    dis_channels=dis_channels,
    model_channels=args.model_channels,
    out_channels=1,
    num_res_blocks=2,
    attention_resolutions=(8, 16),
    channel_mult=(1, 2, 4, 4),
    dims=3,
    use_checkpoint=True,
    use_fp16=False,
)
model = model.to(device)

# ----- Optional: resume from a saved checkpoint -----
if args.resume_from:
    if not os.path.isfile(args.resume_from):
        raise FileNotFoundError(f"--resume_from not found: {args.resume_from}")
    if is_main:
        print(f"[Resume] Loading model weights from: {args.resume_from}", flush=True)
    state_dict = torch.load(args.resume_from, map_location='cpu')
    # checkpoints saved during training also strip 'module.' already; be defensive
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if is_main:
        if missing:
            print(f"[Resume] missing keys ({len(missing)}): {missing[:3]}{' ...' if len(missing) > 3 else ''}", flush=True)
        if unexpected:
            print(f"[Resume] unexpected keys ({len(unexpected)}): {unexpected[:3]}{' ...' if len(unexpected) > 3 else ''}", flush=True)
    del state_dict

flow_model = FlowMatching(model)

ddp_model = DDP(
    model,
    device_ids=[args.local_rank],
    output_device=args.local_rank,
    find_unused_parameters=False,
)
flow_model.net = ddp_model

# EMA (each rank keeps an identical copy — all-reduce keeps params in sync)
ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
if ema is not None and args.resume_ema:
    if not os.path.isfile(args.resume_ema):
        raise FileNotFoundError(f"--resume_ema not found: {args.resume_ema}")
    if is_main:
        print(f"[Resume] Loading EMA shadow weights from: {args.resume_ema}", flush=True)
    ema_state = torch.load(args.resume_ema, map_location='cpu')
    # ema state_dict only has trainable params (no buffers); update shadow_params in place
    with torch.no_grad():
        n_loaded = 0
        for name, p in ema.shadow_params.items():
            if name in ema_state:
                p.copy_(ema_state[name].to(p.device))
                n_loaded += 1
        if is_main:
            print(f"[Resume] EMA shadow restored for {n_loaded}/{len(ema.shadow_params)} params", flush=True)
    del ema_state


# ----------------------------- OPTIMIZER + LR -----------------------------
optimizer = optim.AdamW(ddp_model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)

warmup_epochs = max(1, int(args.epoch * args.warmup_ratio))
warmup_epochs = min(warmup_epochs, max(1, args.epoch - 1))
cosine_epochs = max(1, args.epoch - warmup_epochs)
warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=args.min_lr)
lr_scheduler = SequentialLR(
    optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_epochs],
)

if is_main:
    print(
        f"LR schedule: LinearWarmup({warmup_epochs} ep) + Cosine({cosine_epochs} ep), "
        f"lr_max={args.lr_max}, min_lr={args.min_lr}, weight_decay={args.weight_decay}",
        flush=True,
    )
    print(
        f"EMA: {'decay=' + str(args.ema_decay) if ema is not None else 'disabled'}",
        flush=True,
    )


# ----------------------------- VAL FUNCTION -----------------------------
@torch.no_grad()
def evaluate_mae(use_ema: bool):
    """Compute mean absolute dose error inside Mask_possible_dose_mask.

    Returns aggregated MAE (Gy) across all val patients (averaged over ranks).
    """
    if not val_enabled:
        return float('nan')

    ddp_model.eval()

    cm = ema.average_parameters(model) if (use_ema and ema is not None) else contextlib.nullcontext()

    mae_sum = torch.zeros(1, device=device)
    mae_count = torch.zeros(1, device=device)

    with cm:
        iterator = val_dataloader
        if is_main:
            iterator = tqdm(val_dataloader, desc="[Val ]", leave=False)

        for ct, dis_cond, dose_gt in iterator:
            ct = ct.to(device, non_blocking=True).float()
            dis_cond = dis_cond.to(device, non_blocking=True).float()
            dose_gt = dose_gt.to(device, non_blocking=True).float()

            pred = flow_model.sample(ct, dis_cond, steps=args.val_steps)

            pred_gy = torch.clamp((pred + 1.0) * 40.0, 0.0, 80.0)
            gt_gy = torch.clamp((dose_gt + 1.0) * 40.0, 0.0, 80.0)

            # possible_dose_mask is channel index 10 of dis (see OPENKBP_MASK_NAMES)
            body_mask = dis_cond[:, 10:11, ...]

            for b in range(ct.shape[0]):
                bm_sum = body_mask[b].sum()
                if bm_sum.item() > 0:
                    mae_b = (torch.abs(pred_gy[b] - gt_gy[b]) * body_mask[b]).sum() / bm_sum
                    mae_sum += mae_b
                    mae_count += 1

    dist.all_reduce(mae_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(mae_count, op=dist.ReduceOp.SUM)
    ddp_model.train()
    if mae_count.item() == 0:
        return float('inf')
    return (mae_sum / mae_count).item()


# ----------------------------- TRAIN LOOP -----------------------------
best_mae = float('inf')
best_mae_epoch = -1
best_mae_use_ema = False

for epoch in range(all_epochs):
    train_sampler.set_epoch(epoch)
    current_lr = optimizer.param_groups[0]['lr']
    ddp_model.train()

    train_epoch_loss = []
    if is_main:
        loader = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{all_epochs}")
    else:
        loader = train_dataloader

    grad_accum_steps = max(1, int(args.grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)

    n_batches = len(train_dataloader)

    for i, (ct, dis_cond, rtdose) in enumerate(loader):
        ct = ct.to(device, non_blocking=True).float()
        dis_cond = dis_cond.to(device, non_blocking=True).float()
        rtdose = rtdose.to(device, non_blocking=True).float()

        loss_raw = flow_model.get_loss(rtdose, ct, dis_cond)
        loss_scaled = loss_raw / grad_accum_steps
        loss_scaled.backward()

        loss_detached = loss_raw.detach()
        dist.all_reduce(loss_detached, op=dist.ReduceOp.SUM)
        loss_detached = loss_detached / world_size
        train_epoch_loss.append(loss_detached.item())

        is_accum_step = (i + 1) % grad_accum_steps == 0
        is_last_batch = (i + 1) == n_batches
        if is_accum_step or is_last_batch:
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # EMA update once per optimizer step
            if ema is not None:
                ema.update(model)

        if is_main:
            loader.set_postfix(loss=loss_detached.item(), lr=current_lr)

    lr_scheduler.step()

    if is_main:
        mean_loss = float(np.mean(train_epoch_loss))
        print(f"Epoch {epoch + 1}/{all_epochs}  lr={current_lr:.2e}  train_loss={mean_loss:.4f}", flush=True)

    # ----- Validation -----
    run_val = val_enabled and ((epoch + 1) % args.val_every == 0)
    if run_val:
        # Validate with EMA weights if available, else with current weights
        val_mae = evaluate_mae(use_ema=(ema is not None))

        if is_main:
            tag = "EMA" if ema is not None else "raw"
            print(f"  [Val] epoch {epoch + 1}  MAE({tag})={val_mae:.4f} Gy  (best={best_mae:.4f} @epoch{best_mae_epoch})", flush=True)

            if val_mae < best_mae:
                best_mae = val_mae
                best_mae_epoch = epoch + 1
                best_mae_use_ema = (ema is not None)
                # save best raw + best EMA
                torch.save(model.state_dict(), os.path.join(save_dir, 'model_best_mae.pth'))
                if ema is not None:
                    torch.save(ema.state_dict_for_save(model), os.path.join(save_dir, 'ema_best_mae.pth'))
                print(f"  [Val] *new best* MAE={best_mae:.4f} Gy at epoch {best_mae_epoch} -> saved", flush=True)

    # ----- Periodic checkpoint -----
    if is_main and ((epoch + 1) % max(1, args.save_every) == 0):
        torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch{epoch + 1}.pth'))
        if ema is not None:
            torch.save(ema.state_dict_for_save(model), os.path.join(save_dir, f'ema_epoch{epoch + 1}.pth'))

if is_main:
    # final save
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_final.pth'))
    if ema is not None:
        torch.save(ema.state_dict_for_save(model), os.path.join(save_dir, 'ema_final.pth'))
    print(f"Training finished. Best val MAE = {best_mae:.4f} Gy @ epoch {best_mae_epoch} "
          f"({'EMA' if best_mae_use_ema else 'raw'})", flush=True)

dist.destroy_process_group()
