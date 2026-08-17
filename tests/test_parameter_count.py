import torch

from ultron_reasoning import UltronConfig, UltronForCausalLM


def test_parameter_budget():
    config = UltronConfig()
    model = UltronForCausalLM(config)
    params = model.num_parameters()
    # The initial architecture is intentionally close to 1B, not an arbitrary
    # round-number target. Keep this test loose until tokenizer/vocabulary is fixed.
    assert 800_000_000 <= params <= 1_200_000_000


def test_forward_shape():
    config = UltronConfig(vocab_size=1024, max_position_embeddings=128)
    model = UltronForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    out = model(input_ids)
    assert out["logits"].shape == (2, 16, config.vocab_size)


def test_loss():
    config = UltronConfig(vocab_size=1024, max_position_embeddings=128)
    model = UltronForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    out = model(input_ids, labels=input_ids)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
