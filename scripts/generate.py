import argparse
import torch

from ultron_reasoning import UltronConfig, UltronForCausalLM
from ultron_reasoning.generation import generate
from ultron_reasoning.tokenizer import ByteTokenizer
from ultron_reasoning.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UltronForCausalLM(UltronConfig()).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    tokenizer = ByteTokenizer()
    ids = torch.tensor([tokenizer.encode(args.prompt, add_bos=True, add_eos=False)], device=device)
    out = generate(model, ids, args.max_new_tokens, eos_token_id=tokenizer.eos_token_id, temperature=args.temperature)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
