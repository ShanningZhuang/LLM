# Feed-Forward Networks

> Parent: [LLM Algorithms](../00_LLM.md)

## Overview

The feed-forward network (FFN) is the other half of each Transformer block, applied after attention. It processes each token **independently** (no cross-token interaction), acting as a learned nonlinear transformation. In modern LLMs, the FFN accounts for roughly **2/3 of all parameters** and is where most "knowledge" is believed to be stored.

## FFN in the Transformer Block

```
                    Attention output
                         │
                    ┌────▼────┐
                    │ RMSNorm │
                    └────┬────┘
                         │
              ┌──────────▼──────────┐
              │                      │
              │    Feed-Forward      │  ← YOU ARE HERE
              │    Network (FFN)     │
              │                      │
              │  Standard:           │
              │  FFN(x) = W₂·σ(W₁·x)│
              │                      │
              │  SwiGLU:             │
              │  FFN(x) = W₂·(Swish(W₁·x) ⊙ W₃·x)│
              │                      │
              └──────────┬──────────┘
                         │
                       (+) residual
                         │
                    Next block
```

## Topics

| Topic | File | Priority |
|-------|------|----------|
| Standard MLP | [01_MLP.md](01_MLP.md) | ★★★★☆ |
| Activation Functions | [02_Activation_Functions.md](02_Activation_Functions.md) | ★★★★☆ |
| Gated FFN (SwiGLU) | [03_Gated_FFN.md](03_Gated_FFN.md) | ★★★★★ |
| Mixture of Experts | [04_MoE.md](04_MoE.md) | ★★★★☆ |

## Parameter Count

| FFN Type | Parameters per Layer | Example (d=4096) |
|----------|---------------------|-------------------|
| Standard (4× expansion) | 2 × d × 4d = 8d² | 134M |
| SwiGLU (8/3× expansion) | 3 × d × (8d/3) = 8d² | 134M |
| MoE (8 experts, top-2) | 8 × 8d² = 64d² | 1.07B |

Note: SwiGLU uses 3 weight matrices but a smaller expansion ratio, keeping total parameters roughly equal to standard FFN.

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) | Standard 2-layer FFN with ReLU |
| [GLU Variants (Shazeer, 2020)](https://arxiv.org/abs/2002.05202) | SwiGLU, GeGLU, ReGLU |
| [Switch Transformer (2021)](https://arxiv.org/abs/2101.03961) | Simplified MoE routing |
| [Mixtral (2024)](https://arxiv.org/abs/2401.04088) | Practical MoE at scale |
