"""Mixed-precision helpers with CPU-safe fallbacks."""
import contextlib
import torch


def autocast_context(device: torch.device, enabled: bool = True, dtype=torch.bfloat16):
    if not enabled:
        return contextlib.nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    if device.type == "cpu":
        return torch.autocast(device_type="cpu", dtype=dtype)
    return contextlib.nullcontext()


def make_grad_scaler(enabled: bool = False):
    # BF16 does not normally require gradient scaling; kept as a compatibility hook.
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)
