"""Normalize JSONL instruction/reasoning records into a simple text corpus."""
import argparse
import json
from pathlib import Path


def convert(src: Path, dst: Path):
    count = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "text" in obj:
                text = obj["text"]
            elif "prompt" in obj and "response" in obj:
                text = f"<|user|>\n{obj['prompt']}\n<|assistant|>\n{obj['response']}"
            else:
                raise ValueError("record must contain text or prompt/response")
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} records to {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.input, args.output)
