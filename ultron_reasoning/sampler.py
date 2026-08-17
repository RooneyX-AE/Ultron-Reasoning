"""Distributed-friendly token sampling utilities."""
import torch
import torch.nn.functional as F


def top_k_top_p_filter(logits, top_k=0, top_p=1.0):
    out = logits.clone()
    if top_k > 0:
        k = min(top_k, out.size(-1))
        threshold = torch.topk(out, k, dim=-1).values[..., -1, None]
        out = out.masked_fill(out < threshold, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(out, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        remove = cumulative - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        out = torch.full_like(out, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return out


def sample(logits, temperature=1.0, top_k=0, top_p=1.0):
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = top_k_top_p_filter(logits / temperature, top_k, top_p)
    return torch.multinomial(F.softmax(logits, dim=-1), 1)
