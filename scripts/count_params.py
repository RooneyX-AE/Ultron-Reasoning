from ultron_reasoning import UltronConfig, UltronForCausalLM

config = UltronConfig()
model = UltronForCausalLM(config)
params = model.num_parameters()
print(f"parameters: {params:,}")
print(f"parameters (B): {params / 1e9:.4f}")
print(f"layers: {config.num_hidden_layers}")
print(f"hidden: {config.hidden_size}")
print(f"heads: {config.num_attention_heads}/{config.num_key_value_heads}")
