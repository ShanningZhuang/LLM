# Pre-Norm vs Post-Norm

> Parent: [Transformer Block](00_Transformer_Block.md)

## Overview

The placement of normalization relative to the sublayer and residual connection defines two major Transformer variants. The original Transformer (2017) used **Post-Norm** -- normalize after the residual addition. GPT-2 (2019) switched to **Pre-Norm** -- normalize before the sublayer -- and nearly all modern LLMs follow suit. This seemingly small architectural choice has a large impact on training stability, learning rate sensitivity, and final model quality.

## The Two Variants

### Post-Norm (Original Transformer)

Apply the sublayer first, add the residual, **then** normalize:

```
x_{l+1} = LayerNorm(x_l + Sublayer(x_l))
```

### Pre-Norm (GPT-2, LLaMA, and most modern LLMs)

Normalize the input **before** passing it to the sublayer:

```
x_{l+1} = x_l + Sublayer(Norm(x_l))
```

Note: the residual stream passes through **unnormalized**. The norm only affects what the sublayer sees.

## Side-by-Side Architecture

```
POST-NORM (Original Transformer)        PRE-NORM (GPT-2, LLaMA)

    x                                       x
    │                                       │
    ├──────────────┐                        ├──────────────┐
    │              │                        │              │
    ▼              │                        ▼              │
┌──────────┐       │                   ┌──────────┐       │
│ Sublayer │       │                   │   Norm   │       │
│ (Attn)   │       │                   └────┬─────┘       │
└────┬─────┘       │                        ▼              │
     ▼             │                   ┌──────────┐       │
   (+) ◄───────────┘                   │ Sublayer │       │
     │                                 │ (Attn)   │       │
     ▼                                 └────┬─────┘       │
┌──────────┐                                ▼              │
│   Norm   │                              (+) ◄───────────┘
└────┬─────┘                                │
     │                                      │
     ├──────────────┐                       ├──────────────┐
     │              │                       │              │
     ▼              │                       ▼              │
┌──────────┐       │                   ┌──────────┐       │
│ Sublayer │       │                   │   Norm   │       │
│ (FFN)    │       │                   └────┬─────┘       │
└────┬─────┘       │                        ▼              │
     ▼             │                   ┌──────────┐       │
   (+) ◄───────────┘                   │ Sublayer │       │
     │                                 │ (FFN)    │       │
     ▼                                 └────┬─────┘       │
┌──────────┐                                ▼              │
│   Norm   │                              (+) ◄───────────┘
└────┬─────┘                                │
     │                                      │
     ▼                                      ▼
   output                                output

Norm is AFTER add                    Norm is BEFORE sublayer
Residual stream passes               Residual stream passes
through Norm each time                through UNNORMALIZED
```

## Why Pre-Norm Trains More Stably

### Gradient Norm Analysis

In **Post-Norm**, the gradient must flow through the normalization layer at every block. LayerNorm involves division by the standard deviation, which can amplify or attenuate gradients unpredictably. For a network with L layers:

```
Post-Norm gradient path:

∂L/∂x_0 = ∂L/∂x_L · ∏(l=1..L) ∂[Norm(x + Sub(x))]/∂x

Each Norm introduces a Jacobian that depends on activation statistics.
At initialization, gradients through the Norm layers can vary wildly.
→ Requires careful learning rate warmup to avoid early divergence.
```

In **Pre-Norm**, the residual stream bypasses normalization entirely. The gradient has a clean identity path:

```
Pre-Norm gradient path:

∂L/∂x_0 = ∂L/∂x_L · ∏(l=1..L) (I + ∂Sub(Norm(x))/∂x)

The identity term I dominates at initialization (when sublayers output ~0).
Gradient norm ≈ ||∂L/∂x_L|| regardless of depth.
→ Stable from the start, no warmup needed.
```

### Gradient Norm at Initialization

```
                Gradient norm (∂L/∂x_l) vs layer depth l
   ||grad||
      │
  1.0 ┤ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■   ← Pre-Norm (flat)
      │
      │
  0.5 ┤
      │                               ○
      │                          ○
      │                     ○         ← Post-Norm (decays)
      │                ○
  0.1 ┤           ○
      │      ○
      │ ○
      └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──── layer
        0  3  6  9  12 15 18 21 24 27 30

Pre-Norm:  gradients remain roughly constant across layers
Post-Norm: gradients decay toward earlier layers without warmup
```

## The Quality Trade-Off

Despite Pre-Norm's training advantages, several studies have found that Post-Norm can achieve **slightly better final quality** when trained successfully:

| Aspect | Post-Norm | Pre-Norm |
|--------|-----------|----------|
| Training stability | Fragile, needs LR warmup | Robust, works out of the box |
| Learning rate sensitivity | High | Low |
| Final quality (converged) | Slightly better (some studies) | Slightly worse |
| Maximum stable depth | ~12 layers without tricks | Hundreds of layers |
| Gradient norm across layers | Decays toward input | Approximately uniform |
| LR warmup required | Yes (critical) | No (or much less) |
| Used in practice | Rarely for LLMs | Nearly universal for LLMs |

The practical consensus: Pre-Norm's stability wins. The tiny quality gap (if any) is not worth the training headaches of Post-Norm.

## Variants and Extensions

### Sandwich-Norm

Apply normalization **both before and after** the sublayer:

```
x_{l+1} = x_l + Norm_2(Sublayer(Norm_1(x_l)))
```

Motivation: the extra Norm after the sublayer bounds the magnitude of what gets added to the residual stream. Used in some CogView and Normformer experiments. Adds overhead and complexity, not widely adopted.

### DeepNorm (for 1000+ layer models)

Scales the residual connection to stabilize very deep Post-Norm Transformers:

```
x_{l+1} = LayerNorm(alpha * x_l + Sublayer(x_l))

where:
  alpha = (2N)^(1/4)    for an N-layer Transformer
  Sublayer weights initialized with scale beta = (8N)^(-1/4)
```

DeepNorm modifies the residual scaling and initialization together so that Post-Norm can work at extreme depths (tested up to 1000 layers). The key insight: by making alpha > 1, the residual dominates early in training, preventing gradient issues.

```
Standard residual:  x + F(x)         → F(x) can dominate
DeepNorm residual:  alpha*x + F(x)   → x dominates (alpha > 1)
                                         F(x) starts small (beta scaling)
```

### QK-Norm

A targeted normalization applied specifically to Q and K vectors before computing attention scores. Prevents attention logits from growing too large in deep or high-dimensional models:

```
attn = softmax(Norm(Q) · Norm(K)^T / sqrt(d_k))
```

Used in Gemma-2 and some other recent architectures.

## Which Models Use Which Variant

| Model | Variant | Norm Type | Notes |
|-------|---------|-----------|-------|
| Original Transformer (2017) | Post-Norm | LayerNorm | LR warmup critical |
| BERT (2018) | Post-Norm | LayerNorm | Following original |
| GPT-2 (2019) | Pre-Norm | LayerNorm | Key switch to Pre-Norm |
| GPT-3 (2020) | Pre-Norm | LayerNorm | Scaled init for residual |
| PaLM (2022) | Pre-Norm | LayerNorm | Parallel attention + FFN |
| LLaMA (2023) | Pre-Norm | RMSNorm | Modern standard |
| Mistral (2023) | Pre-Norm | RMSNorm | Following LLaMA |
| Gemma (2024) | Pre-Norm | RMSNorm | + QK-Norm |
| DeepNet (2022) | Post-Norm + scaling | LayerNorm | DeepNorm, 1000 layers |

## PyTorch: Both Variants

```python
import torch
import torch.nn as nn

class PostNormBlock(nn.Module):
    """Post-Norm Transformer block (original Transformer style)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Post-Norm: Norm AFTER residual addition
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)         # add then norm

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)          # add then norm
        return x


class PreNormBlock(nn.Module):
    """Pre-Norm Transformer block (GPT-2 / LLaMA style)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm: Norm BEFORE sublayer, residual stream untouched
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out                     # add (no norm here)

        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out                      # add (no norm here)
        return x


# Compare gradient norms across a stack of blocks
if __name__ == "__main__":
    torch.manual_seed(42)
    d_model, n_heads, d_ff, n_layers = 256, 4, 512, 12

    post_blocks = nn.Sequential(*[PostNormBlock(d_model, n_heads, d_ff)
                                   for _ in range(n_layers)])
    pre_blocks = nn.Sequential(*[PreNormBlock(d_model, n_heads, d_ff)
                                  for _ in range(n_layers)])

    x = torch.randn(1, 16, d_model, requires_grad=True)

    # Post-Norm gradient
    y = post_blocks(x)
    y.sum().backward()
    print(f"Post-Norm input grad norm: {x.grad.norm().item():.4f}")

    x2 = torch.randn(1, 16, d_model, requires_grad=True)
    y2 = pre_blocks(x2)
    y2.sum().backward()
    print(f"Pre-Norm  input grad norm: {x2.grad.norm().item():.4f}")
```

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) | Original Post-Norm Transformer |
| [GPT-2 (Radford et al., 2019)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | Switched to Pre-Norm |
| [On Layer Normalization in the Transformer Architecture (Xiong et al., 2020)](https://arxiv.org/abs/2002.04745) | Theoretical analysis: Pre-Norm has bounded gradients |
| [Understanding the Difficulty of Training Transformers (Liu et al., 2020)](https://arxiv.org/abs/2004.08249) | Admin (adaptive) initialization for Post-Norm |
| [DeepNet (Wang et al., 2022)](https://arxiv.org/abs/2203.00555) | Scaled residual for stable 1000-layer Post-Norm |
| [NormFormer (Shleifer et al., 2021)](https://arxiv.org/abs/2110.09456) | Sandwich-Norm and additional normalizations |

## Related

- [Residual Connection](01_Residual_Connection.md) -- the skip connection that normalization placement is defined relative to
- [Layer Normalization](02_Layer_Normalization.md) -- the normalization operations themselves (LayerNorm, RMSNorm)
