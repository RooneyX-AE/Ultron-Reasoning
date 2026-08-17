import torch

from ultron_reasoning import UltronConfig, UltronForCausalLM
from ultron_reasoning.generation import generate


def test_generation_cache():
    config = UltronConfig(vocab_size=128, num_hidden_layers=2, hidden_size=128, intermediate_size=256, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = UltronForCausalLM(config)
    ids = torch.randint(0, 128, (1, 4))
    out = generate(model, ids, max_new_tokens=4, temperature=0)
    assert out.shape == (1, 8)
