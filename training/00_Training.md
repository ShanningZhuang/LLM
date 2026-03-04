# Training Algorithms

> Parent: [LLM Algorithms](../00_LLM.md)

## Overview

LLM pretraining is conceptually simple — predict the next token — but the engineering and algorithmic details determine whether training converges, diverges, or produces a capable model. This section covers the training objective, optimizer, learning rate schedule, and regularization techniques that make large-scale training work.

## Training Pipeline

```
┌──────────────────────────────────────────────────────┐
│                  Training Loop                        │
│                                                       │
│  for each batch of token sequences:                   │
│                                                       │
│  1. Forward pass                                      │
│     input_ids → model → logits                        │
│                                                       │
│  2. Compute loss                                      │
│     L = CrossEntropy(logits[:-1], input_ids[1:])      │
│     (next-token prediction)                           │
│                                                       │
│  3. Backward pass                                     │
│     loss.backward() → gradients                       │
│                                                       │
│  4. Gradient clipping                                 │
│     clip_grad_norm_(params, max_norm=1.0)             │
│                                                       │
│  5. Optimizer step                                    │
│     AdamW update with learning rate schedule           │
│                                                       │
│  6. Zero gradients                                    │
│     optimizer.zero_grad()                             │
│                                                       │
└──────────────────────────────────────────────────────┘
```

## Topics

| Topic | File | Priority |
|-------|------|----------|
| Pretraining | [01_Pretraining.md](01_Pretraining.md) | ★★★★★ |
| Optimizer (AdamW) | [02_Optimizer.md](02_Optimizer.md) | ★★★★☆ |
| Learning Rate Schedule | [03_Learning_Rate.md](03_Learning_Rate.md) | ★★★★☆ |
| Regularization | [04_Regularization.md](04_Regularization.md) | ★★★☆☆ |

## Typical Hyperparameters

| Hyperparameter | Typical Value | Notes |
|----------------|---------------|-------|
| Optimizer | AdamW | β₁=0.9, β₂=0.95 |
| Peak learning rate | 3e-4 (small) to 1.5e-4 (large) | Scales with model size |
| Weight decay | 0.1 | Applied to non-bias, non-norm params |
| Warmup steps | 2000 | Linear warmup |
| LR schedule | Cosine decay or WSD | Decay to ~10% of peak |
| Gradient clip | 1.0 | Global norm clipping |
| Batch size | 1M–4M tokens | Ramps up during training |
| Precision | BF16 or FP16 | Mixed precision |

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [GPT-3 (Brown et al., 2020)](https://arxiv.org/abs/2005.14165) | Large-scale pretraining recipe |
| [AdamW (Loshchilov & Hutter, 2019)](https://arxiv.org/abs/1711.05101) | Decoupled weight decay |
| [Chinchilla (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556) | Optimal tokens-to-parameters ratio |
| [μP (Yang et al., 2022)](https://arxiv.org/abs/2203.03466) | Hyperparameter transfer across scales |
