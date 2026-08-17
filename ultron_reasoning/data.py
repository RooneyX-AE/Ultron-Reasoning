from pathlib import Path
from typing import Iterable, Iterator

import torch
from torch.utils.data import DataLoader, Dataset


class PackedTokenDataset(Dataset):
    def __init__(self, token_files: Iterable[str], seq_len: int):
        self.seq_len = seq_len
        tokens = []
        for filename in token_files:
            raw = Path(filename).read_bytes()
            tokens.extend(raw)
        if len(tokens) < seq_len + 1:
            raise ValueError("dataset does not contain enough tokens")
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.length = (len(self.tokens) - 1) // seq_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.tokens[start:start + self.seq_len]
        y = self.tokens[start + 1:start + self.seq_len + 1]
        return {"input_ids": x, "labels": y}


def make_loader(dataset, batch_size, shuffle=True, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
