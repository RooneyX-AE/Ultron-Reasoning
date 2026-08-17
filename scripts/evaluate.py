import argparse
import torch

from ultron_reasoning import UltronConfig, UltronForCausalLM
from ultron_reasoning.checkpoint import load_checkpoint
from ultron_reasoning.data import PackedTokenDataset, make_loader
from ultron_reasoning.evaluation import perplexity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokens", nargs="+", required=True)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UltronForCausalLM(UltronConfig()).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    ds = PackedTokenDataset(args.tokens, args.seq_len)
    metric = perplexity(model, make_loader(ds, args.batch_size, shuffle=False), device=device)
    print(f"perplexity={metric:.4f}")


if __name__ == "__main__":
    main()
