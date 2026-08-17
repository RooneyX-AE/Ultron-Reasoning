import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ultron_reasoning import UltronConfig, UltronForCausalLM
from ultron_reasoning.data import PackedTokenDataset, make_loader
from ultron_reasoning.training import TrainConfig, train


def main():
    parser = argparse.ArgumentParser(description="Train Ultron-Reasoning 1B")
    parser.add_argument("--tokens", nargs="+", required=True, help="pre-tokenized token files")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1000)
    args = parser.parse_args()

    dataset = PackedTokenDataset(args.tokens, args.seq_len)
    loader = make_loader(dataset, args.batch_size)
    model = UltronForCausalLM(UltronConfig())
    cfg = TrainConfig(
        learning_rate=args.lr,
        max_steps=args.steps,
        grad_accumulation_steps=args.grad_accumulation,
        save_every=args.save_every,
    )
    Path(args.output).mkdir(parents=True, exist_ok=True)
    train(model, loader, cfg, device=args.device, output_dir=args.output)


if __name__ == "__main__":
    main()
