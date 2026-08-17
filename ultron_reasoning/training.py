import math
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch.optim import AdamW

from .checkpoint import save_checkpoint


@dataclass
class TrainConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: int = 2000
    max_steps: int = 100000
    grad_clip: float = 1.0
    grad_accumulation_steps: int = 1
    log_every: int = 10
    save_every: int = 1000
    bf16: bool = True


def build_optimizer(model, cfg: TrainConfig):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "norm" not in name.lower() and "bias" not in name.lower():
            decay.append(p)
        else:
            no_decay.append(p)
    return AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.learning_rate,
        betas=cfg.betas,
        eps=cfg.eps,
    )


def cosine_lr(step, cfg):
    if step < cfg.warmup_steps:
        return (step + 1) / max(1, cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def _device(device):
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def train(model, loader: Iterable, cfg: TrainConfig, device="cuda", output_dir="checkpoints", resume: Optional[str] = None):
    device = _device(device)
    model.to(device)
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not cfg.bf16)
    start_step = 0

    if resume:
        from .checkpoint import load_checkpoint
        start_step = load_checkpoint(resume, model, optimizer, map_location=device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        use_amp = cfg.bf16 and device.type == "cuda"
        context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else torch.enable_grad()
        with context:
            loss = model(input_ids, labels=labels)["loss"] / cfg.grad_accumulation_steps
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        micro += 1
        if micro < cfg.grad_accumulation_steps:
            continue
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        step = start_step + 1
        scale = cosine_lr(step - 1, cfg)
        for group in optimizer.param_groups:
            group["lr"] = cfg.learning_rate * scale
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        micro = 0
        if step % cfg.log_every == 0:
            print(f"step={step} loss={loss.item() * cfg.grad_accumulation_steps:.4f} lr={cfg.learning_rate * scale:.3e}")
        if step > 0 and step % cfg.save_every == 0:
            save_checkpoint(f"{output_dir}/step-{step}.pt", model, optimizer, step=step)
        start_step = step
        if step >= cfg.max_steps:
            break
    return model
