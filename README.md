# Ultron-Reasoning

Ultron-Reasoning is a research implementation of a compact, reasoning-focused decoder-only language model targeting the ~1B parameter class.

## Current architecture

- Dense decoder-only Transformer
- 19 layers
- 2048 hidden dimension
- 6144 SwiGLU intermediate dimension
- 16 query heads / 4 key-value heads (GQA)
- 128-dimensional attention heads
- RoPE positional encoding
- RMSNorm
- optional QK-Norm
- tied token embeddings
- 32K target context
- BF16-oriented training

The architecture is deliberately configurable. The repository is a research implementation, not a claim that these values are already optimal for reasoning.

## Status

Foundation implementation is being built on `feat/ultron-1b-foundation` before training infrastructure and reasoning post-training are added.

## License

MIT. See `LICENSE`.
