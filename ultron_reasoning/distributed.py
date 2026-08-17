"""Small distributed-training helpers. Works with single GPU as a no-op."""
import os
import torch
import torch.distributed as dist


def init_distributed():
    if not dist.is_available() or dist.is_initialized():
        return False
    if "RANK" not in os.environ:
        return False
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return True


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process():
    return rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def all_reduce_mean(value: torch.Tensor):
    if not dist.is_initialized():
        return value
    out = value.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out / world_size()
