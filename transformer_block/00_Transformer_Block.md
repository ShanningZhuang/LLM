# Transformer Block

> Parent: [LLM Algorithms](../00_LLM.md)

## Overview

A Transformer block is the repeating unit of an LLM. A model like LLaMA-2 7B stacks 32 identical blocks, each containing an attention sublayer and a feed-forward sublayer, connected by residual connections and normalization. Understanding how these components compose is key to understanding LLM behavior.

## Block Anatomy (Pre-Norm, LLaMA-style)

```
Input x
    │
    ├───────────────────────┐
    │                       │ (residual)
    ▼                       │
┌──────────────┐            │
│   RMSNorm    │            │
└──────┬───────┘            │
       ▼                    │
┌──────────────┐            │
│  Attention   │            │
│  (causal)    │            │
└──────┬───────┘            │
       ▼                    │
     (+) ◄──────────────────┘  x + Attention(RMSNorm(x))
       │
       ├───────────────────────┐
       │                       │ (residual)
       ▼                       │
┌──────────────┐               │
│   RMSNorm    │               │
└──────┬───────┘               │
       ▼                       │
┌──────────────┐               │
│   FFN        │               │
│  (SwiGLU)    │               │
└──────┬───────┘               │
       ▼                       │
     (+) ◄─────────────────────┘  + FFN(RMSNorm(...))
       │
       ▼
   Output (→ next block)
```

## Topics

| Topic | File | Description |
|-------|------|-------------|
| Residual Connection | [01_Residual_Connection.md](01_Residual_Connection.md) | Skip connections, gradient flow |
| Layer Normalization | [02_Layer_Normalization.md](02_Layer_Normalization.md) | LayerNorm, RMSNorm |
| Pre-Norm vs Post-Norm | [03_Pre_Norm_Post_Norm.md](03_Pre_Norm_Post_Norm.md) | Architecture variants |

## Comparison: Original vs Modern Block

| Aspect | Original Transformer (2017) | Modern LLMs (LLaMA, 2023+) |
|--------|---------------------------|---------------------------|
| Normalization | LayerNorm | RMSNorm |
| Norm placement | Post-Norm | Pre-Norm |
| Activation | ReLU | SiLU/Swish |
| FFN | Standard MLP | SwiGLU |
| Position | Sinusoidal (added) | RoPE (in attention) |
| Attention | Full MHA | GQA |

## Related Sections

- [Attention](../attention/00_Attention.md) — Details on the attention sublayer
- [FFN](../ffn/00_FFN.md) — Details on the feed-forward sublayer
- [Position Encoding](../position_encoding/00_Position_Encoding.md) — How position information enters
