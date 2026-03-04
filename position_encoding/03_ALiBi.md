# ALiBi (Attention with Linear Biases)

> Parent: [Position Encoding](00_Position_Encoding.md)

## Overview

ALiBi (Attention with Linear Biases), introduced by Press et al. (2022), takes a radically simple approach to position encoding: **no position embedding at all**. Instead, it adds a fixed linear bias to the attention scores based on the distance between query and key positions. Each attention head uses a different slope, allowing different heads to attend to different ranges -- some focus locally, others globally.

ALiBi's primary advantage is excellent **length generalization**: a model trained on 1K tokens can inference on 8K+ tokens with minimal quality loss, without any fine-tuning or architectural changes.

## Key Idea

```
Standard attention:     softmax(Q K^T / sqrt(d))

ALiBi attention:        softmax(Q K^T / sqrt(d)  +  m * bias_matrix)
                                                     │
                                                     └─ linear penalty
                                                        for distance
```

There is **no position embedding** added to the input. Position information comes entirely from the bias term added directly to the attention logits.

## The Bias Matrix

The bias matrix penalizes attention scores linearly based on distance. For a causal (decoder) model, the bias matrix looks like:

```
Bias matrix for a sequence of length 6:

         key position j
         0    1    2    3    4    5
    0 [  0   -    -    -    -    -  ]     "-" = masked (causal)
q   1 [ -1    0   -    -    -    -  ]
u   2 [ -2   -1   0    -    -    -  ]
e   3 [ -3   -2  -1    0    -    -  ]
r   4 [ -4   -3  -2   -1    0    -  ]
y   5 [ -5   -4  -3   -2   -1    0  ]
i
       bias(i, j) = -(i - j)     for i >= j  (causal positions)

Each entry = -(query_pos - key_pos) = negative distance
```

This raw bias is then multiplied by a head-specific slope `m`:

```
Full ALiBi bias for head h:

    attention_logits = Q K^T / sqrt(d)  +  m_h * bias_matrix

    where m_h is the slope for head h
```

## Head-Specific Slopes

Different heads get different slopes, creating a spectrum from local to global attention:

```
slopes m_h = 2^{-8/n * h}    for h = 1, 2, ..., n_heads

Example with 8 heads:
    m_1 = 2^{-8/8 * 1} = 2^{-1}   = 0.5       (mild penalty → global)
    m_2 = 2^{-8/8 * 2} = 2^{-2}   = 0.25
    m_3 = 2^{-8/8 * 3} = 2^{-3}   = 0.125
    m_4 = 2^{-8/8 * 4} = 2^{-4}   = 0.0625
    m_5 = 2^{-8/8 * 5} = 2^{-5}   = 0.03125
    m_6 = 2^{-8/8 * 6} = 2^{-6}   = 0.015625
    m_7 = 2^{-8/8 * 7} = 2^{-7}   = 0.0078125
    m_8 = 2^{-8/8 * 8} = 2^{-8}   = 0.00390625 (gentle penalty → more global)
```

### Visualizing Slopes Across Heads

```
   Attention penalty as a function of distance, per head:

   Penalty
   (negative
   bias)
    ▲
    │
 -5 │●                              head 1 (m=0.5): steep slope
    │ ●                              → attends mostly to nearby tokens
    │  ●
-2.5│   ●   ○                       head 4 (m=0.0625): moderate slope
    │    ●   ○                       → attends to medium range
    │     ●   ○
  0 │──────●───○────□──────────────► distance |i - j|
    │       ●   ○    □               head 8 (m=0.004): gentle slope
    │        ●   ○    □              → attends broadly (near-global)
    │         ●   ○    □
    │          ●   ○    □
    0    2    4    6    8   10

    ● = head 1 (steep)
    ○ = head 4 (moderate)
    □ = head 8 (gentle)
```

The slopes are **fixed** (not learned). This geometric series was chosen by the authors through experiments and provides good coverage from local to global attention patterns.

## Why ALiBi Generalizes to Longer Sequences

The key to ALiBi's length generalization is that it never assigns meaning to absolute positions. The bias only depends on **relative distance**, and the linear function extends naturally:

```
Training (seq_len = 1024):
    max distance seen = 1023
    max bias = m * (-1023)

Inference (seq_len = 4096):
    max distance = 4095
    max bias = m * (-4095)    ← extrapolation is just linear extension!

    The model has learned "farther away = less attention"
    This principle holds regardless of absolute sequence length.
```

Compare with RoPE, where rotation angles at unseen positions create out-of-distribution frequency patterns that require correction (PI, YaRN).

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import math


def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Compute ALiBi slopes for each attention head.

    Slopes follow a geometric series: 2^{-8/n * h} for h = 1..n_heads
    For n_heads not a power of 2, uses interpolation.

    Returns:
        slopes: (n_heads,) tensor of per-head slopes
    """
    def _get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = _get_slopes_power_of_2(n_heads)
    else:
        # For non-power-of-2: interpolate between closest powers of 2
        closest_power = 2 ** math.floor(math.log2(n_heads))
        slopes = _get_slopes_power_of_2(closest_power)
        extra = _get_slopes_power_of_2(2 * closest_power)
        slopes += extra[0::2][: n_heads - closest_power]

    return torch.tensor(slopes, dtype=torch.float32)


def build_alibi_bias(
    n_heads: int,
    seq_len: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Build the full ALiBi bias tensor.

    Returns:
        bias: (1, n_heads, seq_len, seq_len) tensor
              Lower-triangular (causal) with linear distance penalties.
    """
    # Distance matrix: bias[i][j] = -(i - j) for i >= j
    positions = torch.arange(seq_len, device=device)
    distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq_len, seq_len)

    # Causal mask: only keep i >= j (lower triangular)
    causal_mask = distance >= 0
    distance = distance.float() * causal_mask.float()
    distance = -distance.abs()                                   # negative distances

    # Slopes per head
    slopes = get_alibi_slopes(n_heads).to(device)                # (n_heads,)

    # Broadcast: (n_heads, 1, 1) * (seq_len, seq_len) -> (n_heads, seq_len, seq_len)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * distance.unsqueeze(0)

    # Set future positions to -inf (causal masking)
    causal_penalty = (~causal_mask).float() * (-1e9)
    alibi = alibi + causal_penalty.unsqueeze(0)

    return alibi.unsqueeze(0)  # (1, n_heads, seq_len, seq_len)


class ALiBiAttention(nn.Module):
    """Multi-head attention with ALiBi position encoding."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        # Precompute ALiBi bias (can be recomputed for longer sequences)
        alibi = build_alibi_bias(n_heads, max_seq_len)
        self.register_buffer("alibi", alibi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores with ALiBi bias (NO position embedding on input)
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn + self.alibi[:, :, :seq_len, :seq_len]   # add ALiBi bias
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(out)


# --- Demo ---
if __name__ == "__main__":
    d_model, n_heads, seq_len, batch = 512, 8, 64, 2

    model = ALiBiAttention(d_model, n_heads, max_seq_len=128)
    x = torch.randn(batch, seq_len, d_model)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    # Show slopes
    slopes = get_alibi_slopes(n_heads)
    print(f"\nALiBi slopes for {n_heads} heads:")
    for i, s in enumerate(slopes):
        print(f"  Head {i}: m = {s:.6f}")

    # Show bias matrix for head 0
    bias = build_alibi_bias(n_heads, 6)
    print(f"\nBias matrix for head 0 (m={slopes[0]:.4f}), seq_len=6:")
    print(bias[0, 0].int())
```

## ALiBi vs RoPE: Detailed Comparison

| Aspect | ALiBi | RoPE |
|--------|-------|------|
| Position mechanism | Bias added to attention scores | Rotation applied to Q, K |
| Where applied | After QK^T computation | Before QK^T computation |
| Input modification | None | None |
| Parameters | Zero (slopes are fixed) | Zero (frequencies are fixed) |
| Length generalization | Excellent (linear extrapolation) | Moderate (needs extension methods) |
| Quality at training length | Slightly lower than RoPE | Higher |
| Relative position | Yes (bias depends on distance) | Yes (dot product depends on distance) |
| Distance decay | Linear (by construction) | Natural decay (from rotation) |
| Computational cost | One addition to attention matrix | Element-wise ops on Q, K |
| KV cache compatible | Yes (bias only depends on positions) | Yes (rotations pre-applied) |
| Long context extension | Not needed (already generalizes) | PI, NTK, YaRN required |
| Adoption trend (2024+) | Declining | Dominant |

### Why RoPE Won Despite ALiBi's Advantages

ALiBi's length generalization is superior, but RoPE produces higher quality at the training context length. Since long-context extension methods (PI, NTK, YaRN) largely solved RoPE's extrapolation problem, the quality advantage of RoPE prevailed. Most new models (2023+) use RoPE with context extension rather than ALiBi.

## Models Using ALiBi

| Model | Context Length | Notes |
|-------|---------------|-------|
| BLOOM (BigScience, 2022) | 2048 | 176B parameter multilingual model |
| MPT (MosaicML, 2023) | 2048-65536 | Trained at 2K, extended to 65K |
| Falcon (TII, 2023) | 2048 | 40B and 180B models |

## Where ALiBi Fits in the Transformer

```
Input tokens
     │
     ▼
┌─────────────┐
│ Token Embed  │     No position embedding! (unlike absolute encoding)
└──────┬──────┘      No rotation! (unlike RoPE)
       │
       ▼
  ╔═══════════════════════════════════════╗
  ║  Transformer Block                    ║
  ║                                       ║
  ║  Q, K, V = linear projections         ║
  ║                                       ║
  ║  scores = Q K^T / sqrt(d)             ║
  ║           │                           ║
  ║           ▼                           ║
  ║  ┌─────────────────────┐              ║
  ║  │  + ALiBi bias       │ ← added HERE ║
  ║  │  (m_h * distance)   │              ║
  ║  └─────────┬───────────┘              ║
  ║            │                          ║
  ║            ▼                          ║
  ║       softmax(scores)                 ║
  ║            │                          ║
  ║            ▼                          ║
  ║       attn_weights @ V                ║
  ║                                       ║
  ╚═══════════════════════════════════════╝
```

## Limitations

| Limitation | Detail |
|------------|--------|
| Lower quality | Slightly worse perplexity than RoPE at training length |
| Linear bias assumption | Real attention patterns may not decay linearly with distance |
| Less adopted | Fewer models, less ecosystem support, less research on extensions |
| Fixed slopes | Cannot adapt slopes during training (by design, but limits flexibility) |
| Bidirectional limitation | Original paper focuses on causal models; adapting for encoders needs care |

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [ALiBi (Press et al., 2022)](https://arxiv.org/abs/2108.12409) | Attention with Linear Biases -- length generalization |
| [BLOOM (BigScience, 2022)](https://arxiv.org/abs/2211.05100) | Large-scale model using ALiBi |
| [MPT (MosaicML, 2023)](https://www.mosaicml.com/blog/mpt-7b) | Demonstrated ALiBi's long-context extrapolation |

## Related

- [Absolute Encoding](01_Absolute_Encoding.md) -- the additive approach that ALiBi replaced
- [RoPE](02_RoPE.md) -- the rotary approach that became more popular than ALiBi
- [Context Extension](04_Context_Extension.md) -- extension techniques for RoPE (not needed for ALiBi)
