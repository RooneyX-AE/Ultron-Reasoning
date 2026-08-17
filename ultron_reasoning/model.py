import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import UltronConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x_float = x.float()
        x_float = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x_float.to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, None], persistent=False)
        self.register_buffer("sin_cached", freqs.sin()[None, None], persistent=False)

    def forward(self, x, seq_len: int):
        if seq_len > self.max_seq_len_cached:
            device = x.device
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device))
            cos = freqs.cos()[None, None]
            sin = freqs.sin()[None, None]
        else:
            cos = self.cos_cached[..., :seq_len, :].to(x.device)
            sin = self.sin_cached[..., :seq_len, :].to(x.device)
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    cos = torch.cat((cos, cos), dim=-1)
    sin = torch.cat((sin, sin), dim=-1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class GQAAttention(nn.Module):
    def __init__(self, config: UltronConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_groups = self.num_heads // self.num_kv_heads
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.use_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.use_qk_norm else nn.Identity()
        self.rope = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        bsz, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.size(2)
        else:
            past_len = 0

        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = self.rope(hidden_states, past_len + q_len)
        cos = cos[..., past_len:, :]
        sin = sin[..., past_len:, :]
        q, k = apply_rope(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        present = (k, v) if use_cache else None
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        causal = torch.triu(torch.ones(q_len, k.size(2), device=q.device, dtype=torch.bool), diagonal=1 + past_len)
        if attention_mask is not None:
            causal = causal.unsqueeze(0).unsqueeze(0) | ~attention_mask[:, None, None, :].bool()
        else:
            causal = causal[None, None]

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=~causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = attn.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(out), present


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GQAAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        attn_out, present = self.self_attn(
            self.input_layernorm(x), attention_mask, position_ids, past_key_value, use_cache
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, present


class UltronModel(nn.Module):
    def __init__(self, config: Optional[UltronConfig] = None):
        super().__init__()
        self.config = config or UltronConfig()
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.num_hidden_layers)])
        self.norm = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False):
        x = self.embed_tokens(input_ids)
        presents = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values is not None else None
            x, present = layer(x, attention_mask, position_ids, past, use_cache)
            if use_cache:
                presents.append(present)
        x = self.norm(x)
        return x, presents


class UltronForCausalLM(nn.Module):
    def __init__(self, config: Optional[UltronConfig] = None):
        super().__init__()
        self.config = config or UltronConfig()
        self.model = UltronModel(self.config)
        if self.config.tie_word_embeddings:
            self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight
        else:
            self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, position_ids=None, past_key_values=None, use_cache=False):
        hidden_states, presents = self.model(
            input_ids, attention_mask, position_ids, past_key_values, use_cache
        )
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        return {"loss": loss, "logits": logits, "past_key_values": presents}

    def num_parameters(self, trainable_only=False):
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)
