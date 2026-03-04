# Gated FFN (SwiGLU)

> Parent: [FFN](00_FFN.md)

## Overview

Gated feed-forward networks replace the standard 2-layer MLP with a **gating mechanism** that controls information flow. The key idea: instead of blindly passing all information through an activation function, a separate "gate" projection learns **what to let through**. The most successful variant, **SwiGLU** (Shazeer, 2020), is now the de facto standard FFN in modern LLMs -- used by LLaMA, PaLM, Mistral, DeepSeek, and nearly every post-2022 model.

## From Standard FFN to Gated FFN

### Standard FFN (Recap)

```
FFN(x) = W₂ · σ(W₁ · x)

Two weight matrices:  W₁ (up-projection)
                      W₂ (down-projection)
```

### Gated Linear Unit (GLU) - General Form

```
GLU(x) = (W₁ · x ⊙ σ(W₃ · x)) · W₂

Three weight matrices:  W₁ (value projection)
                        W₃ (gate projection)
                        W₂ (down-projection)

⊙ = element-wise multiplication (Hadamard product)
σ = activation function applied to the gate
```

The critical difference: the input is projected **twice** -- once to produce a "value" and once to produce a "gate". The gate controls how much of each value dimension passes through.

## Gating Mechanism Diagram

```
                           x ∈ ℝ^d
                           │
                    ┌──────┴──────┐
                    │             │
               ┌────▼────┐  ┌────▼────┐
               │   W₁    │  │   W₃    │
               │ (value) │  │ (gate)  │
               │ d → d_ff│  │ d → d_ff│
               └────┬────┘  └────┬────┘
                    │            │
                    │       ┌────▼────┐
                    │       │   σ()   │  activation on gate
                    │       │ (SiLU,  │
                    │       │  GELU)  │
                    │       └────┬────┘
                    │            │
                    │   value    │   gate
                    │  ∈ ℝ^d_ff │  ∈ ℝ^d_ff
                    │            │
                    └─────┬──────┘
                          │
                     ┌────▼────┐
                     │    ⊙    │  element-wise multiply
                     │ (gate)  │  gate controls what passes
                     └────┬────┘
                          │
                          │  gated ∈ ℝ^d_ff
                          │
                     ┌────▼────┐
                     │   W₂    │  down-projection
                     │ d_ff → d│
                     └────┬────┘
                          │
                          │  output ∈ ℝ^d
                          ▼
```

## GLU Variants

Different activation functions applied to the gate produce different variants:

| Variant | Gate Activation σ | Formula | Used By |
|---------|------------------|---------|---------|
| GLU | Sigmoid | (W₁x ⊙ sigmoid(W₃x)) W₂ | Original (Dauphin et al., 2017) |
| ReGLU | ReLU | (W₁x ⊙ ReLU(W₃x)) W₂ | Research |
| GeGLU | GELU | (W₁x ⊙ GELU(W₃x)) W₂ | Gemma |
| **SwiGLU** | **SiLU/Swish** | **(Swish(W₁x) ⊙ W₃x) W₂** | **LLaMA, PaLM, Mistral** |

Note: in SwiGLU, the convention is slightly different -- the activation is applied to W₁ (not W₃), and W₃ produces the unactivated "gate". The effect is equivalent; what matters is that one branch is activated and the other is not.

## SwiGLU in Detail

### Formula

```
SwiGLU(x) = (Swish(W_gate · x) ⊙ W_up · x) · W_down

Where:
  Swish(z) = z · σ(z) = z · sigmoid(z) = SiLU(z)

Expanding:
  gate    = W_gate · x              ∈ ℝ^d_ff
  up      = W_up · x                ∈ ℝ^d_ff
  hidden  = SiLU(gate) ⊙ up         ∈ ℝ^d_ff    ← gated activation
  output  = W_down · hidden          ∈ ℝ^d

Naming in LLaMA codebase:
  W_gate  = gate_proj    (applies SiLU activation)
  W_up    = up_proj      (no activation, provides values)
  W_down  = down_proj    (compresses back to d_model)
```

### Why Gating Helps

```
Standard FFN:
  Each d_ff neuron:  output_i = σ(w₁ᵢ · x)
  → The activation function is the ONLY control on information flow
  → Binary-ish: either passes information or blocks it

Gated FFN:
  Each d_ff neuron:  output_i = σ(w_gate_i · x) × (w_up_i · x)
  → Two separate projections collaborate:
    - gate decides IF information passes  (σ ∈ [0, 1] for sigmoid-like)
    - value decides WHAT information passes
  → Richer representation: the gate can learn to suppress or amplify
    independently of the value

Analogy:
  Standard FFN = dimmer switch (one knob controls both on/off and brightness)
  Gated FFN    = separate on/off switch + brightness dial
```

### Gradient Properties

```
Standard FFN gradient through a neuron:
  ∂/∂x [σ(w₁·x)] = σ'(w₁·x) · w₁
  → single chain rule path

SwiGLU gradient through a neuron:
  ∂/∂x [SiLU(w_gate·x) · (w_up·x)]
  = SiLU'(w_gate·x) · w_gate · (w_up·x)     ← gradient through gate
  + SiLU(w_gate·x) · w_up                     ← gradient through value
  → TWO gradient paths → better gradient flow
```

## Parameter Count: The 3-Matrix Tradeoff

Gated FFN uses 3 weight matrices instead of 2, increasing parameters per d_ff dimension:

```
Standard FFN:  2 matrices × d × d_ff  = 2 · d · d_ff
Gated FFN:     3 matrices × d × d_ff  = 3 · d · d_ff
```

To keep the **same total parameter count**, reduce d_ff:

```
Standard:  2 · d · d_ff_std  = 2 · d · 4d  = 8d²

Gated (equal params):
  3 · d · d_ff_gated = 8d²
  d_ff_gated = 8d/3 ≈ 2.67d

So: d_ff = (2/3) × 4d = 8d/3
```

### Real-World d_ff Values

| Model | d_model | d_ff | Ratio | Param Equiv |
|-------|---------|------|-------|-------------|
| LLaMA-2 7B | 4096 | 11008 | 2.69x | 3 x 4096 x 11008 = 135M/layer |
| LLaMA-2 13B | 5120 | 13824 | 2.70x | 3 x 5120 x 13824 = 212M/layer |
| LLaMA-2 70B | 8192 | 28672 | 3.50x | 3 x 8192 x 28672 = 704M/layer |
| Mistral 7B | 4096 | 14336 | 3.50x | 3 x 4096 x 14336 = 176M/layer |
| LLaMA-3 8B | 4096 | 14336 | 3.50x | 3 x 4096 x 14336 = 176M/layer |

Note: LLaMA-2 7B uses d_ff=11008 (close to 8/3 x 4096 = 10922.67, rounded to nearest multiple of 256 for hardware efficiency). Newer models like Mistral and LLaMA-3 use d_ff=14336 (= 3.5 x 4096), trading more parameters for more capacity.

## Comparison: Standard FFN vs SwiGLU vs GeGLU

| Aspect | Standard FFN | SwiGLU | GeGLU |
|--------|-------------|--------|-------|
| Weight matrices | 2 (W₁, W₂) | 3 (gate, up, down) | 3 (gate, up, down) |
| Gate activation | N/A | SiLU/Swish | GELU |
| d_ff (iso-param) | 4d | 8d/3 ≈ 2.67d | 8d/3 ≈ 2.67d |
| Params per layer | 8d² | 8d² | 8d² |
| FLOPs per layer | 16d²T | 24d²T (3 matmuls) | 24d²T (3 matmuls) |
| Training quality | Baseline | Best (most models) | Very good |
| Inference speed | Fastest | ~1.1-1.2x standard | ~1.1-1.2x standard |
| Used by | GPT-2/3, BERT | LLaMA, PaLM, Mistral | Gemma |

Note: FLOPs are higher for gated variants (3 matrix multiplies vs 2), but quality improvement outweighs the extra cost. T = sequence length.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU FFN as used in LLaMA, PaLM, Mistral.

    SwiGLU(x) = W_down · (SiLU(W_gate · x) ⊙ W_up · x)

    Three projections:
      gate_proj: d_model → d_ff  (activation applied here)
      up_proj:   d_model → d_ff  (no activation, provides values)
      down_proj: d_ff → d_model  (compresses back)
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        gate = F.silu(self.gate_proj(x))   # (batch, seq_len, d_ff)
        up   = self.up_proj(x)             # (batch, seq_len, d_ff)
        return self.down_proj(gate * up)   # (batch, seq_len, d_model)


class GeGLU(nn.Module):
    """GeGLU: same structure as SwiGLU but uses GELU as gate activation."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(x))
        up   = self.up_proj(x)
        return self.down_proj(gate * up)


# Comparison demo
if __name__ == "__main__":
    d_model = 4096

    # Standard FFN: d_ff = 4 * d_model
    d_ff_std = 4 * d_model  # 16384
    std_ffn = nn.Sequential(
        nn.Linear(d_model, d_ff_std, bias=False),
        nn.SiLU(),
        nn.Linear(d_ff_std, d_model, bias=False),
    )

    # SwiGLU: d_ff = 8/3 * d_model (rounded)
    d_ff_swiglu = 11008  # LLaMA-2 7B value
    swiglu = SwiGLU(d_model, d_ff_swiglu)

    # Parameter counts
    std_params = sum(p.numel() for p in std_ffn.parameters())
    swiglu_params = sum(p.numel() for p in swiglu.parameters())

    print(f"Standard FFN:  {std_params:>12,} params  (d_ff={d_ff_std})")
    print(f"SwiGLU FFN:    {swiglu_params:>12,} params  (d_ff={d_ff_swiglu})")
    print(f"Ratio:         {swiglu_params/std_params:.3f}")
    # Standard FFN:   134,217,728 params  (d_ff=16384)
    # SwiGLU FFN:     135,266,304 params  (d_ff=11008)
    # Ratio:          1.008  (nearly identical total params)

    # Forward pass
    x = torch.randn(2, 128, d_model)
    out_std = std_ffn(x)
    out_swiglu = swiglu(x)
    print(f"\nStandard output shape: {out_std.shape}")    # [2, 128, 4096]
    print(f"SwiGLU output shape:   {out_swiglu.shape}")   # [2, 128, 4096]
```

## Why SwiGLU Outperforms Standard FFN

Shazeer (2020) showed consistent improvements across different model sizes and tasks:

```
Perplexity on C4 (lower is better):

  Model size    Standard (ReLU)    SwiGLU     Improvement
  ──────────    ───────────────    ──────     ───────────
  Small          25.2               24.0       -4.8%
  Medium         21.3               20.2       -5.2%
  Large          18.7               17.8       -4.8%

  (Iso-parameter comparison: same total parameter count)

The improvement is consistent and comes "for free" --
same parameter count, slightly more FLOPs, better quality.
```

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Language Modeling with Gated Convolutional Networks (Dauphin et al., 2017)](https://arxiv.org/abs/1612.08083) | Original Gated Linear Unit (GLU) |
| [GLU Variants Improve Transformer (Shazeer, 2020)](https://arxiv.org/abs/2002.05202) | SwiGLU, GeGLU, ReGLU -- systematic comparison |
| [PaLM (Chowdhery et al., 2022)](https://arxiv.org/abs/2204.02311) | SwiGLU at scale (540B parameters) |
| [LLaMA (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971) | SwiGLU as standard recipe for efficient LLMs |

## Related

- [Standard MLP](01_MLP.md) -- the 2-layer FFN that SwiGLU replaces
- [Activation Functions](02_Activation_Functions.md) -- SiLU/Swish and GELU used as gate activations
- [Mixture of Experts](04_MoE.md) -- each expert is typically a SwiGLU FFN
