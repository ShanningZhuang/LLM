# Activation Functions

> Parent: [FFN](00_FFN.md)

## Overview

The activation function in a Transformer FFN provides the critical nonlinearity between the two linear layers. Without it, stacking two linear transformations would collapse into a single linear transformation, eliminating the network's ability to learn complex patterns. The choice of activation function has evolved from ReLU (original Transformer, 2017) to GELU (BERT, GPT-2) to SiLU/Swish (LLaMA, Mistral), with each generation offering smoother gradients and better empirical performance.

## The Three Main Activation Functions

### ReLU (Rectified Linear Unit)

```
ReLU(x) = max(0, x)

Gradient:
  dReLU/dx = 1   if x > 0
           = 0   if x < 0
           = undefined at x = 0  (typically set to 0)
```

Used by: Original Transformer (2017), early GPT models.

### GELU (Gaussian Error Linear Unit)

```
GELU(x) = x · Φ(x)

where Φ(x) = CDF of the standard normal distribution
            = 0.5 × (1 + erf(x / √2))

Fast approximation (used in practice):
  GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))

Even faster approximation:
  GELU(x) ≈ x · σ(1.702 × x)    where σ = sigmoid
```

Used by: BERT, GPT-2, GPT-3, OPT, BLOOM.

### SiLU / Swish

```
SiLU(x) = x · σ(x)

where σ(x) = 1 / (1 + e^(-x))  = sigmoid

Gradient:
  dSiLU/dx = σ(x) + x · σ(x) · (1 - σ(x))
           = σ(x) · (1 + x · (1 - σ(x)))
```

Used by: LLaMA, LLaMA-2, LLaMA-3, Mistral, PaLM (typically as the gate activation in SwiGLU).

## ASCII Shape Comparison

```
Output
  ▲
  │              ReLU               GELU              SiLU/Swish
  │
4 ┤          ╱                  ╱                   ╱
  │         ╱                  ╱                   ╱
3 ┤        ╱                 ╱                   ╱
  │       ╱                 ╱                   ╱
2 ┤      ╱                ╱                   ╱
  │     ╱               ╱                   ╱
1 ┤    ╱              ·                   ·
  │   ╱             ·                   ·
0 ┤───────·       ·                 ··
  │       :     ·                ··
-1┤       :   (≈ -0.17 min)   (≈ -0.28 min)
  │       :
  ├───┬───┬───►           ───►             ───►
 -3  -1   1  3    Input   -3  1  3  Input  -3  1  3  Input

Key differences:
  ReLU:   Hard zero for x<0,  sharp corner at x=0
  GELU:   Smooth curve,       small negative dip (~-0.17)
  SiLU:   Smooth curve,       larger negative dip (~-0.28)
```

## Mathematical Properties

| Property | ReLU | GELU | SiLU/Swish |
|----------|------|------|------------|
| Formula | max(0, x) | x * Phi(x) | x * sigma(x) |
| Range | [0, inf) | [~-0.17, inf) | [~-0.28, inf) |
| Monotonic | Yes | No (tiny dip) | No (clear dip) |
| Smooth | No (kink at 0) | Yes (C^inf) | Yes (C^inf) |
| Gradient at 0 | Undefined | 0.5 | 0.5 |
| Gradient for x >> 0 | 1 | ~1 | ~1 |
| Gradient for x << 0 | 0 (dead) | ~0 (near-dead) | ~0 (near-dead) |
| Self-gating | No | Yes (implicit) | Yes (x * gate) |
| Computational cost | Cheapest | ~4x ReLU | ~2x ReLU |

## Why Smooth Activations Are Better for LLMs

### The Dead Neuron Problem (ReLU)

```
With ReLU:
  If W₁[i] · x + b₁[i] < 0 for all x in a batch:
    → neuron i outputs 0
    → gradient through neuron i is 0
    → W₁[i] never updates
    → neuron is permanently dead

In a large LLM (d_ff = 16384):
  Empirically 10-30% of neurons can become permanently dead
  = 1,600 to 4,900 wasted neurons per layer
  = wasted parameters that contribute nothing
```

### Smooth Activations Solve This

```
With GELU/SiLU:
  For x < 0:  output is small but NON-ZERO
                gradient is small but NON-ZERO

  GELU(-1) = -0.159    gradient ≈ -0.083  (still learning!)
  SiLU(-1) = -0.269    gradient ≈ -0.072  (still learning!)
  ReLU(-1) =  0.000    gradient =  0.000  (dead.)

No neuron is ever permanently dead → more efficient use of parameters.
```

### Non-Monotonicity Helps

```
The slight negative dip in GELU and SiLU creates a "soft gating" effect:

  Slightly negative inputs → small negative output → acts as a soft "no"
  Zero-ish inputs → near-zero output → uncertainty
  Positive inputs → passes through → acts as a soft "yes"

This is richer than ReLU's binary 0-or-pass behavior:
  ReLU:  "off" (exactly 0) or "on" (pass through)
  SiLU:  "soft no" → "uncertain" → "soft yes" → "strong yes"
```

### Gradient Flow Comparison

```
Backpropagation through FFN:

  ∂L/∂W₁ = ∂L/∂output · W₂ᵀ · σ'(W₁·x) · xᵀ
                                 ^^^^^^^^
                          This term matters!

ReLU:  σ'(z) = 0 for z < 0  → half the gradient is ZEROED
GELU:  σ'(z) ≠ 0 for z < 0  → gradient flows through ALL neurons
SiLU:  σ'(z) ≠ 0 for z < 0  → gradient flows through ALL neurons

Result: smoother optimization landscape, more stable training at scale.
```

## Computational Cost

| Activation | Operations | Relative Cost | Notes |
|-----------|-----------|---------------|-------|
| ReLU | 1 comparison | 1.0x | Fastest, hardware-optimized |
| GELU (exact) | erf + multiply | ~6x | Rarely used in practice |
| GELU (tanh approx) | tanh + multiply | ~4x | Used by most implementations |
| GELU (sigmoid approx) | sigmoid + multiply | ~2.5x | Fast approximation |
| SiLU/Swish | sigmoid + multiply | ~2x | Simple and fast |

In practice, the activation function is a tiny fraction of total compute (matrix multiplications dominate), so the cost difference is negligible:

```
Time breakdown for one FFN forward pass (d=4096, d_ff=16384, seq_len=2048):

  W₁ matmul:    ████████████████████████████████████ 48%
  W₂ matmul:    ████████████████████████████████████ 48%
  Activation:   █ ~2-4%
  Memory ops:   ▏ <1%

Activation cost is noise compared to the matrix multiplications.
```

## PyTorch Implementation

```python
import torch
import torch.nn.functional as F


x = torch.linspace(-3, 3, 7)
print(f"Input:  {x.tolist()}")

# ReLU
print(f"ReLU:   {F.relu(x).tolist()}")
# [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]

# GELU
print(f"GELU:   {[round(v, 3) for v in F.gelu(x).tolist()]}")
# [-0.004, -0.046, -0.159, 0.0, 0.841, 1.955, 2.996]

# SiLU / Swish
print(f"SiLU:   {[round(v, 3) for v in F.silu(x).tolist()]}")
# [-0.143, -0.238, -0.269, 0.0, 0.731, 1.762, 2.857]


# In an FFN:
class FFN(torch.nn.Module):
    def __init__(self, d_model, d_ff, activation="gelu"):
        super().__init__()
        self.up = torch.nn.Linear(d_model, d_ff, bias=False)
        self.down = torch.nn.Linear(d_ff, d_model, bias=False)
        self.act_fn = {
            "relu": F.relu,
            "gelu": F.gelu,
            "silu": F.silu,
        }[activation]

    def forward(self, x):
        return self.down(self.act_fn(self.up(x)))


# Compare activations
d_model, d_ff = 512, 2048
x = torch.randn(1, 32, d_model)

for act in ["relu", "gelu", "silu"]:
    ffn = FFN(d_model, d_ff, activation=act)
    out = ffn(x)
    print(f"{act:>4s}: output mean={out.mean():.4f}, std={out.std():.4f}")
```

## Which Model Uses Which Activation

| Model Family | Activation | FFN Type | Year |
|-------------|-----------|----------|------|
| Transformer (original) | ReLU | Standard | 2017 |
| BERT | GELU | Standard | 2018 |
| GPT-2 | GELU | Standard | 2019 |
| GPT-3 | GELU | Standard | 2020 |
| T5 | ReLU | Standard | 2019 |
| OPT | ReLU | Standard | 2022 |
| BLOOM | GELU | Standard | 2022 |
| PaLM | SiLU (SwiGLU) | Gated | 2022 |
| LLaMA / LLaMA-2 / LLaMA-3 | SiLU (SwiGLU) | Gated | 2023-24 |
| Mistral / Mixtral | SiLU (SwiGLU) | Gated | 2023-24 |
| Gemma | GELU (GeGLU) | Gated | 2024 |
| DeepSeek-V2/V3 | SiLU (SwiGLU) | Gated | 2024 |
| Qwen-2 | SiLU (SwiGLU) | Gated | 2024 |
| Phi-3 | SiLU (SwiGLU) | Gated | 2024 |

**Clear trend**: pre-2022 models use standard FFN with ReLU or GELU. Post-2022 models almost universally use gated variants (SwiGLU or GeGLU) with SiLU/Swish as the gate activation.

## Historical Evolution

```
2017 ── ReLU ──────── Original Transformer
           │            Simple, fast, but dead neurons
           │
2018 ── GELU ──────── BERT, GPT-2
           │            Smooth, no dead neurons
           │            Became the standard for 3 years
           │
2020 ── SiLU/Swish ── Proposed for Transformers (Shazeer)
           │            Similar to GELU but simpler formula
           │            Combined with gating → SwiGLU
           │
2022+ ─ SwiGLU ────── PaLM, LLaMA, Mistral, DeepSeek, ...
                        SiLU as gate activation in GLU variant
                        Now the de facto standard
```

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Rectified Linear Units (Nair & Hinton, 2010)](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf) | ReLU for deep networks |
| [GELU (Hendrycks & Gimpel, 2016)](https://arxiv.org/abs/1606.08415) | Gaussian Error Linear Unit |
| [Swish (Ramachandran et al., 2017)](https://arxiv.org/abs/1710.05941) | Swish/SiLU discovered via search |
| [GLU Variants (Shazeer, 2020)](https://arxiv.org/abs/2002.05202) | Compared SwiGLU, GeGLU, ReGLU in Transformers |

## Related

- [Standard MLP](01_MLP.md) -- where the activation function sits in the FFN
- [Gated FFN (SwiGLU)](03_Gated_FFN.md) -- gated variants that use SiLU/GELU as gate activations
