from dataclasses import dataclass


@dataclass
class UltronConfig:
    vocab_size: int = 65536
    hidden_size: int = 2048
    num_hidden_layers: int = 19
    intermediate_size: int = 6144
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    max_position_embeddings: int = 32768
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    tie_word_embeddings: bool = True
    use_qk_norm: bool = True
    initializer_range: float = 0.02
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.hidden_size // self.num_attention_heads <= 0:
            raise ValueError("invalid attention head dimension")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
