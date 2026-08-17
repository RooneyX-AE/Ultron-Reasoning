from pathlib import Path
from typing import List


class ByteTokenizer:
    """Deterministic byte-level fallback tokenizer for smoke tests and bootstrapping.

    Production training should replace this with the project BPE/SentencePiece tokenizer.
    IDs 0..255 represent raw bytes; special tokens are configurable above that range.
    """
    def __init__(self, bos_token_id=256, eos_token_id=257, pad_token_id=0):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.vocab_size = 258

    def encode(self, text: str, add_bos=False, add_eos=True) -> List[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        data = bytes(i for i in ids if 0 <= i <= 255)
        return data.decode("utf-8", errors="replace")

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text(
            '{"type":"byte","vocab_size":258,"bos_token_id":256,"eos_token_id":257}',
            encoding="utf-8",
        )
