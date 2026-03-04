# Attention Mechanism

> Parent: [LLM Algorithms](../00_LLM.md)

## Overview

Attention is the core mechanism that allows each token to "look at" every other token in the sequence. In decoder-only LLMs, attention is **causal** (each token can only attend to itself and previous tokens). This section covers the full attention pipeline: from basic scaled dot-product to modern efficiency optimizations like GQA and FlashAttention.

## Attention Pipeline

```
Input hidden states: X ∈ ℝ^{seq_len × d_model}
         │
    ┌────▼─────────────────────┐
    │   Linear Projections      │
    │   Q = X·W_Q               │  W_Q ∈ ℝ^{d × d}
    │   K = X·W_K               │  W_K ∈ ℝ^{d × d_kv}
    │   V = X·W_V               │  W_V ∈ ℝ^{d × d_kv}
    └────┬─────────────────────┘
         │
    ┌────▼─────────────────────┐
    │   Split into heads        │
    │   Q: (n_heads, seq, d_k)  │
    │   K: (n_kv_heads, seq, d_k)│  ← fewer heads for GQA
    │   V: (n_kv_heads, seq, d_k)│
    └────┬─────────────────────┘
         │
    ┌────▼─────────────────────┐
    │   Apply RoPE to Q, K      │  (position encoding)
    └────┬─────────────────────┘
         │
    ┌────▼─────────────────────┐
    │   Scaled Dot-Product      │
    │   Attn = softmax(QK^T/√d) │
    │   + Causal Mask            │
    │   Output = Attn · V        │
    └────┬─────────────────────┘
         │
    ┌────▼─────────────────────┐
    │   Concat heads + W_O      │  Project back to d_model
    └──────────────────────────┘
```

## Topics

| Topic | File | Priority |
|-------|------|----------|
| Self-Attention | [01_Self_Attention.md](01_Self_Attention.md) | ★★★★★ |
| Multi-Head Attention | [02_Multi_Head_Attention.md](02_Multi_Head_Attention.md) | ★★★★★ |
| MQA & GQA | [03_MQA_GQA.md](03_MQA_GQA.md) | ★★★★★ |
| Causal Mask | [04_Causal_Mask.md](04_Causal_Mask.md) | ★★★★☆ |
| Efficient Attention | [05_Efficient_Attention.md](05_Efficient_Attention.md) | ★★★★☆ |
| Linear Attention & Hybrid | [../linear_attention/00_Linear_Attention.md](../linear_attention/00_Linear_Attention.md) | ★★★★☆ |

## Attention Complexity

| Aspect | Standard Attention | FlashAttention |
|--------|-------------------|----------------|
| Time | O(n² · d) | O(n² · d) (same) |
| Memory | O(n²) for attention matrix | O(n) (no materialization) |
| IO | Many HBM reads/writes | Tiled, stays in SRAM |
| Wall-clock | Baseline | 2-4× faster |

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) | Scaled dot-product, multi-head attention |
| [Multi-Query Attention (2019)](https://arxiv.org/abs/1911.02150) | Shared KV heads for fast inference |
| [GQA (2023)](https://arxiv.org/abs/2305.13245) | Grouped-query as MHA/MQA middle ground |
| [FlashAttention (2022)](https://arxiv.org/abs/2205.14135) | IO-aware tiled attention |
| [FlashAttention-2 (2023)](https://arxiv.org/abs/2307.08691) | Better parallelism, 2× faster |
