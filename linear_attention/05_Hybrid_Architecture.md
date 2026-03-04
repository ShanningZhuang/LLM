# Hybrid Architectures

> Parent: [Linear Attention](00_Linear_Attention.md)

## Overview

Pure linear attention models and pure softmax Transformers each have trade-offs. **Hybrid architectures** interleave both types of layers in a single model, using softmax attention for precise local/global retrieval and linear attention for efficient long-range state tracking. This is the design behind **Qwen 3.5** and represents the emerging consensus for next-generation LLMs.

---

## Why Hybrid?

### Softmax Attention Strengths

- **Precise retrieval**: softmax concentrates attention on exactly the right tokens
- **In-context learning**: strong at few-shot pattern matching
- **Fine-grained reasoning**: attention patterns can implement precise algorithms

### Softmax Attention Weaknesses

- **O(n) KV cache per layer**: memory grows linearly with context length during inference
- **O(n^2) training cost**: quadratic in sequence length (mitigated by FlashAttention but not eliminated)

### Linear Attention Strengths

- **O(1) per-token inference**: fixed-size state, no KV cache growth
- **Efficient long-range**: naturally handles very long sequences
- **State compression**: entire history compressed into d_v × d_k matrix

### Linear Attention Weaknesses

- **Imprecise retrieval**: compressed state can't match softmax's precision
- **Quality gap on recall-heavy tasks**: when exact token retrieval matters

### The Hybrid Solution

```
Use each where it's strongest:

Linear attention layers    → long-range dependencies, state tracking
                             O(1) memory, bulk of the layers

Softmax attention layers   → precise retrieval, complex reasoning
                             O(n) memory, but only a few layers

Result: Near-Transformer quality with much lower inference cost
```

---

## Interleaving Patterns

### Alternating Pattern

The most common approach: alternate between linear and softmax layers.

```
Layer 0:   [Linear Attention]     ← Gated DeltaNet (O(1) state)
Layer 1:   [Softmax Attention]    ← Sliding Window Attention
Layer 2:   [Linear Attention]
Layer 3:   [Softmax Attention]
Layer 4:   [Linear Attention]
Layer 5:   [Linear Attention]     ← Can have consecutive linear layers
Layer 6:   [Softmax Attention]
...

Each layer still has FFN/MLP after the attention mechanism.
```

### Design Decisions

| Decision | Options | Trade-off |
|----------|---------|-----------|
| Ratio (linear:softmax) | 3:1, 2:1, 1:1 | More linear = faster inference, less precise |
| Softmax type | Full attention, sliding window | Sliding window = local precision, O(w) cache |
| Linear type | Mamba-2, GLA, Gated DeltaNet | Gated DeltaNet = best quality |
| Where to place softmax | Every N layers, early/late, uniform | Architecture-specific tuning |

---

## Qwen 3.5: Hybrid Gated DeltaNet

Qwen 3.5 is a flagship example of the hybrid approach:

### Architecture

```
┌────────────────────────────────────┐
│           Qwen 3.5                  │
│                                    │
│  Embedding                         │
│       │                            │
│  ┌────▼────────────────────────┐   │
│  │  Gated DeltaNet Layer       │   │ ← Linear attention (most layers)
│  │  S_t = α(I-βkk^T)S + βvk^T │   │    O(1) per-token, FLA kernels
│  │  + SwiGLU FFN               │   │
│  ├─────────────────────────────┤   │
│  │  Sliding Window Attention   │   │ ← Softmax attention (some layers)
│  │  Local context window       │   │    FlashAttention, O(w) cache
│  │  + SwiGLU FFN               │   │
│  ├─────────────────────────────┤   │
│  │  Gated DeltaNet Layer       │   │
│  ├─────────────────────────────┤   │
│  │  Gated DeltaNet Layer       │   │
│  ├─────────────────────────────┤   │
│  │  Sliding Window Attention   │   │
│  │  ...                        │   │
│  └─────────────────────────────┘   │
│       │                            │
│  LM Head                           │
└────────────────────────────────────┘

Inference memory:
  Gated DeltaNet layers: O(d²) per layer (fixed state)
  Sliding Window layers: O(w × d) per layer (w = window size)
  vs. pure Transformer:  O(n × d) per layer (grows with context!)
```

### Why Sliding Window for Softmax Layers?

Full attention in the softmax layers would still create O(n) KV caches. Using **sliding window attention** (attending only to the last w tokens) bounds the KV cache at O(w × d) per softmax layer, regardless of total context length.

The linear attention layers handle long-range dependencies through their recurrent state, so the softmax layers only need to provide local precision.

---

## Other Hybrid Models

| Model | Linear Component | Softmax Component | Notes |
|-------|-----------------|-------------------|-------|
| **Qwen 3.5** | Gated DeltaNet | Sliding window attention | Production LLM, 2025 |
| **Jamba** (AI21) | Mamba | Full attention | 1:7 attention:Mamba ratio |
| **Zamba** (Zyphra) | Mamba | Full attention | Shared attention layers |
| **Samba** (Microsoft) | Mamba | Sliding window attention | Research model |
| **StripedHyena** (Together) | Hyena (long conv) | Full attention | Alternating layers |
| **RecurrentGemma** (Google) | Griffin (RG-LRU) | Local attention | Based on linear RNN |

---

## Inference Advantages

### Memory Comparison

```
Pure Transformer (32 layers, d=4096, n=100K tokens):
  KV cache = 32 layers × 2 (K,V) × n × d × 2 bytes
           = 32 × 2 × 100,000 × 4,096 × 2 = ~50 GB per sequence

Hybrid (24 linear + 8 sliding window, w=4096):
  Linear layers:  24 × state_size (constant, ~few MB total)
  Window layers:  8 × 2 × 4,096 × 4,096 × 2 = ~0.5 GB
  Total: ~0.5 GB                                ← 100× smaller!
```

### Throughput

```
Decoding one token:

Pure Transformer:    load all KV cache from HBM → memory-bandwidth bound
                     Cost grows with context length

Hybrid:              Linear layers: O(d²) compute, O(d²) memory
                     Window layers: O(w·d) compute, O(w·d) memory
                     Cost constant regardless of context length!
```

---

## Training

### Kernel Stack

```
Hybrid model training pipeline:

Linear attention layers:
  └── FLA (flash-linear-attention) Triton kernels
      └── Chunk-wise parallel: quadratic within chunks, linear across
      └── Supports: GLA, DeltaNet, Gated DeltaNet, Mamba-2

Softmax attention layers:
  └── FlashAttention-2/3
      └── Tiled attention with online softmax
      └── Sliding window masking (skip tiles outside window)

Both types integrated in the same training loop,
same optimizer, same mixed-precision strategy.
```

### Training Considerations

- **Linear attention layers** may need different learning rates or initialization than softmax layers
- **Warmup**: some hybrid models pretrain as pure Transformer, then distill/convert some layers to linear attention
- **Progressive training**: start with more softmax layers, gradually replace with linear as the model learns

---

## The Broader Trend

```
2017  Pure Transformer          ← "Attention Is All You Need"
      │
2020  Efficient Transformers    ← Sparse attention, Performers, etc.
      │
2023  Pure SSM/Mamba           ← "Can we drop attention entirely?"
      │                           Result: competitive but not dominant
2024  Hybrid models emerge     ← Jamba, StripedHyena, Samba
      │
2025  Production hybrids       ← Qwen 3.5: hybrid is the new default
      │
      ▼
Future: The question is no longer "attention vs. linear"
        but "how to best combine them"
```

The hybrid approach reflects a pragmatic consensus: softmax attention remains unmatched for precise retrieval, but paying its O(n) cost at every layer is wasteful. Using it sparingly (at a few layers with sliding windows) while letting efficient linear attention handle the bulk of computation gives the best quality-efficiency trade-off.

---

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Jamba (Lieber et al., 2024)](https://arxiv.org/abs/2403.19887) | First production hybrid Mamba + attention |
| [Gated DeltaNet (Yang et al., 2025)](https://arxiv.org/abs/2412.06464) | Linear attention component used in Qwen 3.5 |
| Qwen 3.5 Technical Report (2025) | Hybrid Gated DeltaNet + sliding window at scale |
| [StripedHyena (Poli et al., 2023)](https://arxiv.org/abs/2305.13048) | Alternating Hyena + attention layers |
| [RecurrentGemma (De et al., 2024)](https://arxiv.org/abs/2404.07839) | Griffin: RG-LRU + local attention |

## Related

- [Gated DeltaNet](04_Gated_DeltaNet.md) — The linear attention component in Qwen 3.5
- [Mamba](03_Mamba.md) — The linear attention component in Jamba
- [Efficient Attention](../attention/05_Efficient_Attention.md) — FlashAttention for the softmax layers
- [Autoregressive Decoding](../generation/01_Autoregressive_Decoding.md) — KV cache mechanics that hybrids optimize
