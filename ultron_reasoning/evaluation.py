import math
from typing import Callable, Iterable

import torch


def perplexity(model, batches: Iterable, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model.eval().to(device)
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in batches:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            out = model(ids, labels=labels)
            tokens = (labels != -100).sum().item()
            total_loss += out["loss"].item() * tokens
            total_tokens += tokens
    return math.exp(total_loss / max(1, total_tokens))


def exact_match(predictions, references):
    pairs = zip(predictions, references)
    return sum(p.strip() == r.strip() for p, r in pairs) / max(1, len(predictions))


def verify_numeric_answer(prediction: str, expected: str) -> float:
    """Minimal deterministic verifier for exact numeric final answers."""
    import re
    nums_p = re.findall(r"[-+]?\d+(?:\.\d+)?", prediction.replace(",", ""))
    nums_e = re.findall(r"[-+]?\d+(?:\.\d+)?", expected.replace(",", ""))
    if not nums_e or not nums_p:
        return 0.0
    return 1.0 if nums_p[-1] == nums_e[-1] else 0.0
