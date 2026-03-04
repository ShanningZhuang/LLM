# Residual Connection

> Parent: [Transformer Block](00_Transformer_Block.md)

## Overview

A residual (skip) connection adds the input of a sublayer directly to its output:

```
output = x + F(x)
```

where `F` is any sublayer (attention or FFN). Without residuals, training deep Transformers (32+ layers) would be impractical due to vanishing gradients. Every modern LLM uses residual connections around both the attention and feed-forward sublayers.

## Why Residuals Are Essential

### The Vanishing Gradient Problem

In a deep network without residuals, gradients must pass through every layer's nonlinearity during backpropagation. Each layer multiplies the gradient by its Jacobian, and if those Jacobians have spectral norms < 1, the gradient shrinks exponentially:

```
∂L/∂x_0 = ∂L/∂x_N · ∂x_N/∂x_{N-1} · ... · ∂x_1/∂x_0

         = ∂L/∂x_N · ∏(i=0..N-1) J_i

If ||J_i|| < 1 for most layers → gradient vanishes
If ||J_i|| > 1 for most layers → gradient explodes
```

### The Residual Solution

With `x_{l+1} = x_l + F_l(x_l)`, the Jacobian of each layer becomes:

```
∂x_{l+1}/∂x_l = I + ∂F_l/∂x_l
```

The identity matrix `I` guarantees that gradients always have a direct path backward, regardless of what `∂F/∂x` looks like. The gradient from layer N to layer 0 expands to:

```
∂x_N/∂x_0 = ∏(l=0..N-1) (I + ∂F_l/∂x_l)

           = I + Σ(individual terms) + Σ(pairwise terms) + ...
```

The crucial `I` term means the gradient is **at least** the identity, never vanishing to zero.

## Gradient Flow: With vs Without Residuals

```
WITHOUT RESIDUALS                   WITH RESIDUALS

  x_0                                x_0
   │                                  │
   ▼                                  ├──────────────┐
┌──────┐                           ┌──────┐          │
│ F_0  │  gradient must             │ F_0  │          │
└──┬───┘  pass through              └──┬───┘          │
   │      every layer                  ▼              │
   ▼                                 (+) ◄────────────┘  "highway"
┌──────┐                              │
│ F_1  │                              ├──────────────┐
└──┬───┘                           ┌──────┐          │
   │                               │ F_1  │          │
   ▼                               └──┬───┘          │
┌──────┐                              ▼              │
│ F_2  │                             (+) ◄────────────┘
└──┬───┘                              │
   │                                  ├──────────────┐
   ▼                               ┌──────┐          │
  ...                              │ F_2  │          │
   │                               └──┬───┘          │
   ▼                                  ▼              │
  x_N                                (+) ◄────────────┘
                                      │
Gradient:                             ▼
∂L/∂x_0 = ∏ J_i                     x_N
         → vanishes
         (if ||J_i|| < 1)          Gradient:
                                   ∂L/∂x_0 = ∏(I + J_i)
                                            → always includes I^N = I
                                            → gradient has a "highway"
```

## The Residual Stream View

A powerful mental model (introduced by Anthropic and the mechanistic interpretability community) treats the hidden state as a **residual stream** that flows unchanged through the network, with each sublayer **reading from** and **writing to** it:

```
x_0 ──────────────────────────────────────────────────────► x_N
       │  ▲        │  ▲        │  ▲        │  ▲
       │  │        │  │        │  │        │  │
       ▼  │        ▼  │        ▼  │        ▼  │
     ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
     │Attn_0│    │FFN_0 │    │Attn_1│    │FFN_1 │    ...
     └──────┘    └──────┘    └──────┘    └──────┘
```

In this view:
- **The stream** carries accumulated information (starts as token + position embedding)
- **Each sublayer** reads the current stream, computes a delta, and adds it back
- The final output `x_N` is the sum of the original embedding plus all sublayer contributions:

```
x_N = x_0 + Attn_0(x_0') + FFN_0(x_0'') + Attn_1(x_1') + FFN_1(x_1'') + ...
```

This additive structure means sublayers can operate somewhat independently -- a key insight for understanding circuits in LLMs.

## Ensemble Interpretation

Residual networks can be interpreted as **implicit ensembles of shallow networks** (Veit et al., 2016). Expanding the product:

```
x_3 = (I + F_2)(I + F_1)(I + F_0)(x_0)

    = x_0                          ← path through 0 layers
    + F_0(x_0)                     ← path through layer 0 only
    + F_1(x_0)                     ← path through layer 1 only
    + F_2(x_0)                     ← path through layer 2 only
    + F_1(F_0(x_0))                ← path through layers 0,1
    + F_2(F_0(x_0))                ← path through layers 0,2
    + F_2(F_1(x_0))                ← path through layers 1,2
    + F_2(F_1(F_0(x_0)))           ← path through all 3 layers
```

A network with N residual layers has **2^N** implicit paths. Experiments show that:
- Removing a single layer has mild effect (only half the paths are affected)
- Most effective paths are **short** (length ~N/2), behaving like a shallow ensemble
- This explains the robustness of residual networks to layer dropping

## PyTorch Implementation

```python
import torch
import torch.nn as nn

class TransformerBlockWithResidual(nn.Module):
    """Single Transformer block showing residual connections explicitly."""

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
        # Residual 1: around attention
        # x is the "residual stream" — it flows straight through
        residual = x
        x_norm = self.norm1(x)                        # read from stream
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + attn_out                       # write back to stream

        # Residual 2: around FFN
        residual = x
        x_norm = self.norm2(x)                        # read from stream
        ffn_out = self.ffn(x_norm)
        x = residual + ffn_out                        # write back to stream

        return x

# Quick demo: verify gradient flow
if __name__ == "__main__":
    block = TransformerBlockWithResidual(d_model=512, n_heads=8, d_ff=2048)
    x = torch.randn(2, 16, 512, requires_grad=True)
    out = block(x)
    out.sum().backward()
    print(f"Input grad norm: {x.grad.norm().item():.4f}")   # should be healthy
    print(f"Output norm:     {out.norm().item():.4f}")
```

## Practical Details

| Aspect | Detail |
|--------|--------|
| Scaling | Some architectures scale the residual branch by a factor (e.g., DeepNorm uses alpha * x + F(x)) |
| Initialization | Residual branches often initialized to small values so F(x) ≈ 0 at start |
| GPT-2 trick | Scale residual weights by 1/sqrt(N) where N = number of layers |
| ReZero | Replace residual with x + alpha * F(x), where alpha is a learnable scalar initialized to 0 |
| Dropout | Applied to F(x) before adding to x (not to the residual stream itself) |

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Deep Residual Learning (He et al., 2015)](https://arxiv.org/abs/1512.03385) | Original ResNet — identity shortcuts for training very deep CNNs |
| [Residual Networks as Ensembles (Veit et al., 2016)](https://arxiv.org/abs/1605.06431) | Ensemble interpretation of residual networks |
| [ReZero (Bachlechner et al., 2020)](https://arxiv.org/abs/2003.04887) | Learnable residual scaling initialized to zero |
| [DeepNorm (Wang et al., 2022)](https://arxiv.org/abs/2203.00555) | Scaled residual for stable 1000-layer Transformers |

## Related

- [Layer Normalization](02_Layer_Normalization.md) -- normalization applied within the residual block
- [Pre-Norm vs Post-Norm](03_Pre_Norm_Post_Norm.md) -- where normalization sits relative to the residual
