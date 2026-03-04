# Standard MLP (Multi-Layer Perceptron)

> Parent: [FFN](00_FFN.md)

## Overview

The standard feed-forward network (FFN) in a Transformer is a simple 2-layer MLP applied **independently** to each token position. Despite its simplicity, the FFN accounts for roughly 2/3 of all parameters in a Transformer and is believed to store the majority of factual knowledge. The original Transformer (Vaswani et al., 2017) introduced this design, and it remains the foundation for all modern FFN variants.

## The MLP Formula

The standard 2-layer FFN computes:

```
FFN(x) = W₂ · σ(W₁ · x + b₁) + b₂
```

Where:
- `x ∈ ℝ^d` -- input vector (one token's hidden state)
- `W₁ ∈ ℝ^{d_ff × d}` -- up-projection (expands dimensionality)
- `b₁ ∈ ℝ^{d_ff}` -- first bias (omitted in modern models like LLaMA)
- `σ` -- nonlinear activation function (ReLU, GELU, SiLU, etc.)
- `W₂ ∈ ℝ^{d × d_ff}` -- down-projection (compresses back)
- `b₂ ∈ ℝ^d` -- second bias (omitted in modern models)

Modern LLMs simplify this to:

```
FFN(x) = W₂ · σ(W₁ · x)
```

No bias terms -- fewer parameters, works just as well with proper normalization (RMSNorm before the FFN handles centering).

## Two-Layer Structure

```
                         x ∈ ℝ^d
                         │
                         │  input (d_model = 4096 in LLaMA-2 7B)
                         │
                    ┌────▼────┐
                    │   W₁    │  Up-projection
                    │ d → d_ff│  (4096 → 16384 in original)
                    └────┬────┘  (4096 → 11008 in LLaMA-2 7B)
                         │
                         │  hidden ∈ ℝ^{d_ff}
                         │
                    ┌────▼────┐
                    │   σ()   │  Activation function
                    │ (ReLU,  │  (nonlinearity is crucial --
                    │  GELU,  │   without it, two linear layers
                    │  SiLU)  │   collapse into one)
                    └────┬────┘
                         │
                         │  activated ∈ ℝ^{d_ff}
                         │
                    ┌────▼────┐
                    │   W₂    │  Down-projection
                    │ d_ff → d│  (back to d_model)
                    └────┬────┘
                         │
                         │  output ∈ ℝ^d
                         ▼
```

## The Expansion Ratio

The intermediate dimension `d_ff` is larger than `d_model`. This expansion is critical for capacity:

```
Original Transformer:  d_ff = 4 × d_model

Example:
  d_model = 512   → d_ff = 2048
  d_model = 4096  → d_ff = 16384
```

### Why Expand?

The expansion creates a **bottleneck architecture**:

```
Narrow           Wide              Narrow
(d=4096)  →  (d_ff=16384)  →  (d=4096)

│ compressed │    expanded    │ compressed │
│ repr.      │    space for   │ repr.      │
│            │    pattern     │            │
│            │    matching    │            │
```

1. **More capacity**: the wider hidden layer can represent more complex functions
2. **Sparse activation**: with ReLU, only a fraction of the d_ff neurons activate for any given input -- the network learns to use different neurons for different patterns
3. **Information bottleneck**: compressing back to d forces the network to select the most useful features

## The FFN as Key-Value Memory

Geva et al. (2021) proposed an influential interpretation: each FFN layer acts as a **key-value memory**.

```
W₁ (up-projection):         W₂ (down-projection):
┌─────────────────┐         ┌─────────────────┐
│  key₁ (row 1)   │         │ val₁  val₂  ... │
│  key₂ (row 2)   │         │  │     │         │
│  key₃ (row 3)   │         │  ▼     ▼         │
│  ...             │         │ col₁  col₂  ... │
│  key_{d_ff}      │         │                  │
└─────────────────┘         └─────────────────┘

key_i = i-th row of W₁  ∈ ℝ^d      (pattern detector)
val_i = i-th column of W₂ ∈ ℝ^d    (output to add to residual stream)

FFN(x) = Σᵢ σ(key_i · x) · val_i

For each neuron i:
  1. key_i · x  = how well input x matches pattern i  (memory lookup)
  2. σ(...)     = activation gate (0 if no match)
  3. val_i      = what to write to the residual stream (memory value)
```

This interpretation explains:
- **Knowledge storage**: factual associations are stored as key-value pairs
- **Sparse activation**: only neurons whose keys match the input fire
- **Knowledge editing**: you can modify specific facts by editing specific rows/columns

## No Cross-Token Interaction

Unlike attention, the FFN processes each position **completely independently**:

```
Position 1:  x₁ → FFN(x₁) → y₁     ─┐
Position 2:  x₂ → FFN(x₂) → y₂      │  Same weights W₁, W₂
Position 3:  x₃ → FFN(x₃) → y₃      │  but NO interaction
Position 4:  x₄ → FFN(x₄) → y₄     ─┘  between positions

This is why:
- FFN can be computed as a single batched matrix multiply over all positions
- Attention handles cross-token mixing; FFN handles per-token transformation
- The two sublayers have complementary roles in the Transformer
```

## Parameter Count

For one FFN layer (standard, no bias):

```
Parameters = d × d_ff  (W₁)  +  d_ff × d  (W₂)
           = 2 × d × d_ff

With d_ff = 4d:
           = 2 × d × 4d = 8d²
```

| Component | Parameters (d=4096, d_ff=16384) | Share of Block |
|-----------|--------------------------------|----------------|
| W₁ (up-projection) | 4096 x 16384 = 67.1M | ~33% |
| W₂ (down-projection) | 16384 x 4096 = 67.1M | ~33% |
| FFN total | 134.2M | ~66% |
| Attention (Q,K,V,O) | 4 x 4096 x 4096 = 67.1M | ~33% |

The FFN is **two-thirds of each Transformer block's parameters**. For the whole model:

```
LLaMA-2 7B (32 layers):
  FFN per layer:  3 × 4096 × 11008 = 135.3M  (SwiGLU, 3 matrices)
  FFN all layers: 32 × 135.3M = 4,328M ≈ 4.3B
  Total model:    ~6.6B
  FFN fraction:   4.3B / 6.6B ≈ 65.5%
```

## Expansion Ratios Across Models

| Model | d_model | d_ff | Ratio (d_ff/d) | FFN Type | Bias |
|-------|---------|------|----------------|----------|------|
| Transformer (original) | 512 | 2048 | 4.0x | Standard | Yes |
| GPT-2 | 768 | 3072 | 4.0x | Standard | Yes |
| GPT-3 175B | 12288 | 49152 | 4.0x | Standard | Yes |
| BERT-base | 768 | 3072 | 4.0x | Standard | Yes |
| LLaMA-2 7B | 4096 | 11008 | 2.69x | SwiGLU | No |
| LLaMA-2 70B | 8192 | 28672 | 3.5x | SwiGLU | No |
| Mistral 7B | 4096 | 14336 | 3.5x | SwiGLU | No |
| DeepSeek-V2 | 5120 | 12288 | 2.4x | SwiGLU | No |

Note: SwiGLU models use smaller d_ff ratios because they have 3 weight matrices instead of 2 (see [Gated FFN](03_Gated_FFN.md)).

## Modern Variations

### No Bias Terms (LLaMA, Mistral, most modern LLMs)

```
Original:  FFN(x) = W₂ · σ(W₁ · x + b₁) + b₂    ← with bias
Modern:    FFN(x) = W₂ · σ(W₁ · x)                 ← no bias

Why remove bias:
- RMSNorm before FFN handles any needed centering
- Fewer parameters without measurable quality loss
- Simpler implementation, slightly faster
- Better compatibility with certain parallelism strategies
```

### Different Expansion Ratios

```
Original Transformer:   d_ff = 4 × d_model     (expansion ratio 4)
LLaMA-2 (SwiGLU):      d_ff ≈ 2.69 × d_model  (keeps same total params as 4x standard)
Mistral (SwiGLU):       d_ff = 3.5 × d_model   (slightly more capacity)
```

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardFFN(nn.Module):
    """Standard 2-layer FFN as in the original Transformer."""

    def __init__(self, d_model: int, d_ff: int, bias: bool = True):
        super().__init__()
        self.up_proj   = nn.Linear(d_model, d_ff, bias=bias)     # W₁
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)     # W₂

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = self.up_proj(x)          # (batch, seq_len, d_ff)
        x = F.relu(x)                # activation
        x = self.down_proj(x)        # (batch, seq_len, d_model)
        return x


class ModernFFN(nn.Module):
    """Modern FFN: no bias, GELU activation (GPT-2 / BERT style)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))


# Quick demo
if __name__ == "__main__":
    d_model, d_ff = 4096, 16384  # original 4x expansion
    ffn = StandardFFN(d_model, d_ff, bias=False)

    x = torch.randn(2, 128, d_model)  # (batch=2, seq_len=128, d=4096)
    out = ffn(x)
    print(f"Input shape:  {x.shape}")        # [2, 128, 4096]
    print(f"Output shape: {out.shape}")       # [2, 128, 4096]

    n_params = sum(p.numel() for p in ffn.parameters())
    print(f"Parameters:   {n_params:,}")      # 134,217,728 (= 2 × 4096 × 16384)
    print(f"              = 2 × {d_model} × {d_ff} = {2*d_model*d_ff:,}")
```

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) | Standard 2-layer FFN with ReLU, d_ff = 4d |
| [Transformer Feed-Forward Layers Are Key-Value Memories (Geva et al., 2021)](https://arxiv.org/abs/2012.14913) | Key-value memory interpretation of FFN |
| [Knowledge Neurons in Pretrained Transformers (Dai et al., 2022)](https://arxiv.org/abs/2104.08696) | Identified individual neurons storing facts |

## Related

- [Activation Functions](02_Activation_Functions.md) -- the nonlinearity between the two layers
- [Gated FFN (SwiGLU)](03_Gated_FFN.md) -- modern gated variant that replaces standard MLP
- [Mixture of Experts](04_MoE.md) -- replacing a single FFN with multiple expert FFNs
