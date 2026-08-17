"""Optimizer and learning-rate utilities for Ultron pretraining."""
import math
from typing import Iterable

import torch


def build_adamw(model: torch.nn.Module, lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith("bias") or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=betas, eps=eps,
    )


class CosineWithWarmup(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1, last_epoch=-1):
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = (step + 1) / self.warmup_steps
        else:
            progress = min(1.0, (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
            scale = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]


def clip_gradients(model: torch.nn.Module, max_norm: float = 1.0) -> float:
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
