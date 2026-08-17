"""Minimal RLVR/GRPO-style primitives for verifiable reasoning experiments.

This module intentionally keeps reward computation separate from policy optimization.
A production run should plug in task-specific verifiers and distributed rollout workers.
"""
from dataclasses import dataclass
from typing import Callable, Sequence

import torch


@dataclass
class Rollout:
    prompt: str
    response: str
    reward: float


def normalize_group_rewards(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if rewards.numel() == 1:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)


def grouped_advantages(rewards: Sequence[float], group_size: int) -> torch.Tensor:
    r = torch.tensor(rewards, dtype=torch.float32)
    if r.numel() % group_size:
        raise ValueError("reward count must be divisible by group_size")
    return torch.cat([normalize_group_rewards(g) for g in r.split(group_size)])


def verify_rollouts(rollouts: Sequence[Rollout], verifier: Callable[[str, str], float]):
    return [Rollout(r.prompt, r.response, float(verifier(r.prompt, r.response))) for r in rollouts]
