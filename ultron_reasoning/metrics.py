"""Training and language-model metrics."""
import math


def perplexity(loss: float) -> float:
    return math.exp(min(float(loss), 20.0))


def accuracy_from_logits(logits, labels, ignore_index=-100):
    pred = logits.argmax(dim=-1)
    mask = labels.ne(ignore_index)
    total = int(mask.sum())
    if total == 0:
        return 0.0
    return float((pred.eq(labels) & mask).sum()) / total


def tokens_per_second(num_tokens: int, elapsed_seconds: float) -> float:
    return float(num_tokens) / max(float(elapsed_seconds), 1e-9)
