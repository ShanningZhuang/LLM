# LLM Architecture

> Parent: [LLM Algorithms](../00_LLM.md)

## Overview

Modern Large Language Models are built on the **decoder-only Transformer** architecture. This section covers the bird's-eye view: how a prompt goes in and a completion comes out, what architectural choices define each model family, and how scaling laws guide model design.

## The Decoder-Only Architecture at a Glance

```
Input text: "The cat sat"
         │
         ▼
┌─────────────────────────┐
│     Tokenizer           │  "The" "cat" "sat" → [464, 3797, 3332]
├─────────────────────────┤
│   Token Embedding       │  [464, 3797, 3332] → d_model vectors
│   + Position Encoding   │
├─────────────────────────┤
│   Transformer Block ×1  │  ┐
│   ├── Attention (causal)│  │
│   └── FFN               │  │ N layers
│   Transformer Block ×2  │  │ (e.g., 32 for 7B)
│   ...                   │  │
│   Transformer Block ×N  │  ┘
├─────────────────────────┤
│   RMSNorm (final)       │
├─────────────────────────┤
│   LM Head (unembedding) │  → logits over vocabulary
├─────────────────────────┤
│   Softmax + Sampling    │  → next token: "on"
└─────────────────────────┘
```

## Topics

| Topic | File | Description |
|-------|------|-------------|
| Decoder-Only Architecture | [01_Decoder_Only.md](01_Decoder_Only.md) | Full walkthrough of the decoder-only design |
| Model Families | [02_Model_Family.md](02_Model_Family.md) | GPT, LLaMA, Mistral, DeepSeek comparison |
| Scaling Laws | [03_Scaling_Laws.md](03_Scaling_Laws.md) | Chinchilla, compute-optimal training |

## Key Design Dimensions

| Dimension | Typical Values | Example (LLaMA-2 7B) |
|-----------|----------------|----------------------|
| `d_model` | 2048–8192 | 4096 |
| `n_layers` | 24–80 | 32 |
| `n_heads` | 16–64 | 32 |
| `d_ff` | 4× or 8/3× d_model | 11008 (SwiGLU) |
| `vocab_size` | 32K–128K | 32000 |
| Context length | 2K–128K+ | 4096 |

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) | Transformer architecture |
| [GPT-2 (2019)](https://arxiv.org/abs/1810.04805) | Decoder-only at scale |
| [LLaMA (2023)](https://arxiv.org/abs/2302.13971) | Open-source efficient design |
| [Chinchilla (2022)](https://arxiv.org/abs/2203.15556) | Compute-optimal scaling |
