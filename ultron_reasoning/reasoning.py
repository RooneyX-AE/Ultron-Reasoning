from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn.functional as F


@dataclass
class ReasoningSample:
    prompt: str
    response: str
    reward: float = 0.0


def sequence_logprob(model, input_ids, labels):
    out = model(input_ids, labels=None)
    logits = out["logits"][:, :-1]
    target = labels[:, 1:]
    logp = F.log_softmax(logits, dim=-1)
    token_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return token_lp.sum(dim=-1)


def rejection_sample(samples: Iterable[ReasoningSample], verifier: Callable[[ReasoningSample], float], keep_ratio=0.5):
    scored = []
    for sample in samples:
        sample.reward = float(verifier(sample))
        scored.append(sample)
    scored.sort(key=lambda x: x.reward, reverse=True)
    keep = max(1, int(len(scored) * keep_ratio)) if scored else 0
    return scored[:keep]


def masked_policy_loss(logprobs, old_logprobs, advantages, clip_eps=0.2):
    ratio = torch.exp(logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return -torch.minimum(unclipped, clipped).mean()
