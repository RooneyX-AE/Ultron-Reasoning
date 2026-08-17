"""Streaming JSONL causal-language-model dataset."""
import json
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch.utils.data import IterableDataset


class JsonlTextDataset(IterableDataset):
    def __init__(self, path: str, tokenizer, seq_len: int = 2048, stride: Optional[int] = None):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or seq_len

    def __iter__(self) -> Iterator[dict]:
        buffer = []
        for line in self.path.open("r", encoding="utf-8"):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text")
            if not isinstance(text, str):
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            buffer.extend(ids)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.stride:]
                tokens = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": tokens[:-1], "labels": tokens[1:]}
