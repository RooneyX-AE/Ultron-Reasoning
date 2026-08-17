import torch
from torch.utils.data import DataLoader, Dataset

from ultron_reasoning import UltronConfig, UltronForCausalLM
from ultron_reasoning.train_loop import Trainer


class TinyDataset(Dataset):
    def __init__(self, n=4, seq=8, vocab=128):
        self.x = torch.randint(0, vocab, (n, seq))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]
        return {"input_ids": x, "labels": x.clone()}


def test_train_step_smoke():
    cfg = UltronConfig(vocab_size=128, hidden_size=128, num_hidden_layers=2,
                       intermediate_size=384, num_attention_heads=4,
                       num_key_value_heads=2, max_position_embeddings=32)
    model = UltronForCausalLM(cfg)
    loader = DataLoader(TinyDataset(vocab=128), batch_size=2)
    trainer = Trainer(model, device="cpu", total_steps=1, grad_accumulation=1, bf16=False)
    state = trainer.fit(loader, max_steps=1)
    assert state.step == 1
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
