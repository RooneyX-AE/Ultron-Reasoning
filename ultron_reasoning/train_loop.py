"""Minimal end-to-end causal-LM training engine for Ultron-Reasoning."""
from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from .mixed_precision import autocast_context
from .optim import build_adamw, CosineWithWarmup, clip_gradients


@dataclass
class TrainState:
    step: int = 0
    tokens: int = 0
    best_loss: float = float("inf")


class Trainer:
    def __init__(self, model, device=None, lr=3e-4, weight_decay=0.1,
                 total_steps=1000, warmup_steps=50, grad_accumulation=1,
                 grad_clip=1.0, bf16=True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = build_adamw(model, lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineWithWarmup(self.optimizer, warmup_steps, total_steps)
        self.grad_accumulation = max(1, grad_accumulation)
        self.grad_clip = grad_clip
        self.bf16 = bf16 and self.device.type in ("cuda", "cpu")
        self.state = TrainState()

    def train_step(self, batch):
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch.get("labels", input_ids).to(self.device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device, non_blocking=True)
        with autocast_context(self.device, self.bf16):
            out = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = out["loss"] / self.grad_accumulation
        loss.backward()
        return float(loss.detach()) * self.grad_accumulation

    def fit(self, loader: Iterable, max_steps: Optional[int] = None, log_every=10):
        target = max_steps or self.scheduler.total_steps
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        micro = 0
        for batch in loader:
            loss = self.train_step(batch)
            micro += 1
            if micro < self.grad_accumulation:
                continue
            grad_norm = clip_gradients(self.model, self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.state.step += 1
            self.state.tokens += int(batch["input_ids"].numel())
            self.state.best_loss = min(self.state.best_loss, loss)
            if self.state.step % log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"step={self.state.step} loss={loss:.4f} lr={lr:.3e} grad_norm={grad_norm:.3f}")
            micro = 0
            if self.state.step >= target:
                break
        return self.state
