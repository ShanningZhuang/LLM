# Position Encoding

> Parent: [LLM Algorithms](../00_LLM.md)

## Overview

Transformers have no inherent notion of token order — attention treats input as a set, not a sequence. Position encoding injects order information so the model knows which token is first, second, etc. The choice of position encoding directly affects a model's ability to handle long contexts and generalize to unseen sequence lengths.

## Evolution of Position Encoding

```
2017                  2019                  2021                  2023+
 │                     │                     │                     │
 ▼                     ▼                     ▼                     ▼
Sinusoidal         Learned              RoPE                  Context
(fixed)            Embeddings           (rotary)              Extension
                   (GPT-2)              (LLaMA)               (YaRN, PI)
                                             │
                                        ALiBi (2022)
                                        (bias-based)
```

## Topics

| Topic | File | Priority |
|-------|------|----------|
| Absolute Encoding | [01_Absolute_Encoding.md](01_Absolute_Encoding.md) | ★★★☆☆ |
| RoPE | [02_RoPE.md](02_RoPE.md) | ★★★★★ |
| ALiBi | [03_ALiBi.md](03_ALiBi.md) | ★★★☆☆ |
| Context Extension | [04_Context_Extension.md](04_Context_Extension.md) | ★★★★☆ |

## Comparison

| Method | Type | Length Generalization | Used By |
|--------|------|---------------------|---------|
| Sinusoidal | Absolute, additive | Poor | Original Transformer |
| Learned | Absolute, additive | Poor (fixed max) | GPT-2, BERT |
| RoPE | Relative, multiplicative | Moderate (with extension) | LLaMA, Mistral, Qwen |
| ALiBi | Relative, bias | Good | BLOOM, MPT |

## How Position Information Enters the Model

```
Method 1: Additive (Sinusoidal/Learned)        Method 2: Rotary (RoPE)
┌─────────────────────┐                        ┌─────────────────────┐
│ token_embed + pos_embed │                    │ No addition to input │
│ Added BEFORE transformer │                   │ Applied to Q,K in    │
│ blocks                   │                   │ EACH attention layer  │
└─────────────────────┘                        └─────────────────────┘

Method 3: Bias (ALiBi)
┌─────────────────────┐
│ No addition to input │
│ Bias added to QK^T   │
│ in EACH attention     │
│ layer (linear slope)  │
└─────────────────────┘
```

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) | Sinusoidal position encoding |
| [RoFormer (Su et al., 2021)](https://arxiv.org/abs/2104.09864) | Rotary Position Embedding |
| [ALiBi (Press et al., 2022)](https://arxiv.org/abs/2108.12409) | Attention with Linear Biases |
| [YaRN (Peng et al., 2023)](https://arxiv.org/abs/2309.00071) | Yet another RoPE extensioN |
| [PI (Chen et al., 2023)](https://arxiv.org/abs/2306.15595) | Position Interpolation |
