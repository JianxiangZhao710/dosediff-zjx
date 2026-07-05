"""Warm-start helpers for v2.x training scripts."""
import torch


COMPILER_PREFIX = "constraint_compiler."


def normalize_state_dict(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def filter_reset_compiler(state_dict, reset_compiler=False):
    if not reset_compiler:
        return state_dict, 0
    skipped = sum(1 for k in state_dict if k.startswith(COMPILER_PREFIX))
    filtered = {k: v for k, v in state_dict.items() if not k.startswith(COMPILER_PREFIX)}
    return filtered, skipped


def load_warmstart_checkpoint(model, path, reset_compiler=False, log_fn=print):
    sd = normalize_state_dict(torch.load(path, map_location="cpu"))
    sd, n_skip = filter_reset_compiler(sd, reset_compiler=reset_compiler)
    if reset_compiler:
        log_fn(f"[Resume] reset_compiler: skipped {n_skip} {COMPILER_PREFIX}* keys (random init)")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    log_fn(f"[Resume] from {path} (strict=False) missing={len(missing)} unexpected={len(unexpected)}")
    return missing, unexpected


def adapter_param_ids(model):
    return {id(p) for p in model.adapter_parameters()}


def mask_backbone_grads(model, adapter_ids):
    """Zero backbone grads without requires_grad=False (checkpoint-safe)."""
    for p in model.parameters():
        if id(p) not in adapter_ids and p.grad is not None:
            p.grad = None
