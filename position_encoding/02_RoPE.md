# RoPE (Rotary Position Embedding)

> Parent: [Position Encoding](00_Position_Encoding.md)

## Overview

RoPE (Rotary Position Embedding), introduced by Su et al. (2021), encodes position information by **rotating** query and key vectors in the attention mechanism. Unlike absolute encoding which adds a position vector at the input, RoPE applies a position-dependent rotation to Q and K inside every attention layer. The key property: the dot product between a rotated query at position `m` and a rotated key at position `n` depends only on the relative distance `(m - n)`, making RoPE a relative position encoding despite being applied to individual positions.

RoPE is the dominant position encoding in modern LLMs, used by LLaMA, Mistral, Qwen, PaLM, Gemma, Phi, CodeLlama, DeepSeek, and many others.

## Key Insight: Position as Rotation

Instead of adding a position vector to the input, RoPE rotates the query and key vectors by an angle proportional to their position:

```
Absolute encoding:    q = (Wx + p)         ← position added to input
RoPE:                 q = R(pos) * Wx      ← input projected, then rotated

Where R(pos) is a rotation that depends on position.
```

The critical insight is that when we compute the attention score `q_m^T * k_n`, the rotation matrices interact such that:

```
q_m^T * k_n = (R(m) * q)^T * (R(n) * k)
            = q^T * R(m)^T * R(n) * k
            = q^T * R(m - n) * k           ← depends only on relative position!
```

This works because R(m)^T * R(n) = R(n - m) -- composing rotations.

## Mathematical Formulation

### Step 1: Group Dimensions into Pairs

Given a d-dimensional query or key vector, group the dimensions into d/2 pairs:

```
q = [q_0, q_1, q_2, q_3, q_4, q_5, ..., q_{d-2}, q_{d-1}]
     ╰──pair 0──╯  ╰──pair 1──╯  ╰──pair 2──╯      ╰──pair d/2-1──╯
```

### Step 2: Define Rotation Frequencies

Each pair `i` has a base frequency:

```
theta_i = 10000^{-2i/d}

pair 0:  theta_0 = 10000^{0}     = 1.0        (highest frequency)
pair 1:  theta_1 = 10000^{-2/d}  ~ 0.9998
pair 2:  theta_2 = 10000^{-4/d}  ~ 0.9996
...
pair d/2-1: theta_{d/2-1} = 10000^{-1} = 0.0001  (lowest frequency)
```

### Step 3: Rotate Each Pair by Position-Dependent Angle

At position `m`, rotate each 2D pair by angle `m * theta_i`:

```
┌ q'_{2i}   ┐     ┌ cos(m*theta_i)  -sin(m*theta_i) ┐   ┌ q_{2i}   ┐
│            │  =  │                                   │ * │          │
└ q'_{2i+1} ┘     └ sin(m*theta_i)   cos(m*theta_i)  ┘   └ q_{2i+1} ┘
```

### The Full Rotation Matrix

The complete rotation is a block-diagonal matrix of 2x2 rotations:

```
         ┌                                                              ┐
         │ cos(m*θ₀)  -sin(m*θ₀)   0          0        ...   0    0    │
         │ sin(m*θ₀)   cos(m*θ₀)   0          0        ...   0    0    │
         │ 0           0        cos(m*θ₁)  -sin(m*θ₁)  ...   0    0    │
R(m)  =  │ 0           0        sin(m*θ₁)   cos(m*θ₁)  ...   0    0    │
         │ ...         ...       ...         ...        ...  ...  ...   │
         │ 0           0           0          0     cos(m*θ_{d/2-1}) -sin│
         │ 0           0           0          0     sin(m*θ_{d/2-1})  cos│
         └                                                              ┘
```

### Visual: Rotation of a 2D Vector Pair

```
              y (q_{2i+1})
              ▲
              │        ╱ q' (rotated by m*theta_i)
              │      ╱
              │    ╱  ⟋ angle = m * theta_i
              │  ╱  ╱
              │╱╱
              ┼──────────────► x (q_{2i})
             ╱│
           ╱  │    q (original)
         ╱    │     = (q_{2i}, q_{2i+1})
       •      │
              │

    Different pairs rotate at different speeds:
    pair 0: fast rotation (theta_0 = 1.0)
    pair 1: slightly slower
    ...
    pair d/2-1: very slow rotation (theta_{d/2-1} = 0.0001)

    → Low-index pairs encode fine-grained (local) position
    → High-index pairs encode coarse (global) position
```

## Complex Number Interpretation

RoPE has an elegant formulation using complex numbers. Treat each 2D pair as a complex number:

```
z_i = q_{2i} + j * q_{2i+1}       (where j is the imaginary unit)
```

Then rotation by angle `m * theta_i` is simply multiplication by a complex exponential:

```
z'_i = z_i * e^{j * m * theta_i}
```

The dot product between rotated q at position m and rotated k at position n:

```
Re(z_q * conj(z_k)) = Re(z_q * e^{j*m*theta} * conj(z_k * e^{j*n*theta}))
                     = Re(z_q * conj(z_k) * e^{j*(m-n)*theta})
```

This depends only on `(m - n)` -- the relative position.

## Why RoPE Dominates Modern LLMs

| Property | Benefit |
|----------|---------|
| Relative position | Attention scores depend on `(m - n)`, not absolute positions |
| Natural distance decay | Dot product between distant tokens decays, acting as soft local bias |
| Per-layer application | Applied in every attention layer, position signal never degrades |
| No extra parameters | Rotation angles are fixed, no learned position parameters |
| Efficient computation | No matrix multiply needed -- just element-wise multiply and add |
| Compatible with KV cache | Rotations are pre-applied to cached K, no recomputation needed |
| Extendable | Can be extended to longer contexts via PI, NTK, YaRN |

### Distance Decay Property

The dot product between rotated vectors naturally decays with distance, providing an inductive bias toward local attention:

```
                    Attention score contribution from RoPE
                    ▲
                    │
               1.0  │●
                    │ ●
                    │  ●
                    │   ●●
                    │     ●●●
                    │        ●●●●●
                    │             ●●●●●●●●●●
                    │                       ●●●●●●●●●●●●●●●
                    └──────────────────────────────────────────► |m - n|
                    0    5    10    20         50        100

    Close tokens → strong interaction
    Distant tokens → weak interaction (but not zero)
```

## Efficient Implementation

The naive approach would multiply by the rotation matrix, but this is wasteful. Instead, use the identity:

```
[x cos(θ) - y sin(θ)]   =   [x]   [cos(θ)]       [-y]   [sin(θ)]
[x sin(θ) + y cos(θ)]       [y] ⊙ [cos(θ)]   +   [ x] ⊙ [sin(θ)]

where ⊙ is element-wise multiply
```

This replaces matrix multiplication with two element-wise multiplications and one addition.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import math


def precompute_rope_frequencies(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos and sin tables for RoPE.

    Args:
        dim: dimension of each head (must be even)
        max_seq_len: maximum sequence length to precompute
        base: base for frequency computation (default 10000)

    Returns:
        cos_table: (max_seq_len, dim) cosine values
        sin_table: (max_seq_len, dim) sine values
    """
    # Frequencies for each pair: theta_i = base^{-2i/dim}
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # (dim/2,)

    # Position indices
    positions = torch.arange(max_seq_len).float()                      # (max_seq_len,)

    # Outer product: angles for each (position, pair)
    angles = torch.outer(positions, freqs)                             # (max_seq_len, dim/2)

    # Duplicate for both elements of each pair
    angles = angles.repeat(1, 2)                                       # (max_seq_len, dim)

    return torch.cos(angles), torch.sin(angles)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rearrange pairs: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]"""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    Args:
        q: query tensor, shape (batch, n_heads, seq_len, head_dim)
        k: key tensor,   shape (batch, n_kv_heads, seq_len, head_dim)
        cos: precomputed cosines, shape (seq_len, head_dim)
        sin: precomputed sines,   shape (seq_len, head_dim)

    Returns:
        Rotated (q, k) with same shapes as input
    """
    # Reshape cos/sin for broadcasting: (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply rotation using the efficient formulation
    q_rotated = q * cos + rotate_half(q) * sin
    k_rotated = k * cos + rotate_half(k) * sin

    return q_rotated, k_rotated


class RoPEAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 4096):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        # Precompute RoPE tables
        cos, sin = precompute_rope_frequencies(self.head_dim, max_seq_len)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.wv(x).view(batch, seq_len, self.n_heads, self.head_dim)

        # Transpose to (batch, n_heads, seq_len, head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Apply RoPE to Q and K (NOT to V)
        q, k = apply_rope(q, k, self.cos[:seq_len], self.sin[:seq_len])

        # Standard scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(out)


# --- Demo ---
if __name__ == "__main__":
    d_model, n_heads, seq_len, batch = 512, 8, 64, 2

    model = RoPEAttention(d_model, n_heads)
    x = torch.randn(batch, seq_len, d_model)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    # Verify relative position property
    head_dim = d_model // n_heads
    cos, sin = precompute_rope_frequencies(head_dim, 100)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    # q at pos 10, k at pos 7 (distance = 3)
    q1, k1 = apply_rope(q, k, cos[10:11], sin[10:11])
    _, k1b = apply_rope(q, k, cos[7:8], sin[7:8])
    score1 = (q1 @ k1b.transpose(-2, -1)).item()

    # q at pos 20, k at pos 17 (distance = 3 again)
    q2, k2 = apply_rope(q, k, cos[20:21], sin[20:21])
    _, k2b = apply_rope(q, k, cos[17:18], sin[17:18])
    score2 = (q2 @ k2b.transpose(-2, -1)).item()

    print(f"\nRelative position test (same distance=3, different absolute positions):")
    print(f"  Score at (10, 7):  {score1:.6f}")
    print(f"  Score at (20, 17): {score2:.6f}")
    print(f"  Match: {abs(score1 - score2) < 1e-5}")
```

## Where RoPE Is Applied in the Transformer

```
Input tokens
     │
     ▼
┌─────────────┐
│ Token Embed  │     No position embedding added here!
└──────┬──────┘
       │
       ▼
  ╔═══════════════════════════════════════╗
  ║  Transformer Block (repeated N times) ║
  ║                                       ║
  ║  ┌───────────┐                        ║
  ║  │ RMSNorm   │                        ║
  ║  └─────┬─────┘                        ║
  ║        │                              ║
  ║        ├──► W_Q ──► Q ──► RoPE(Q) ───╮║
  ║        ├──► W_K ──► K ──► RoPE(K) ───┤║  ← RoPE applied HERE
  ║        └──► W_V ──► V ────────────────┤║    (to Q and K only)
  ║                                       ▼║
  ║                               Attention ║
  ║                               score =   ║
  ║                               RoPE(Q) * ║
  ║                               RoPE(K)^T ║
  ║        ┌─────────────────────────┘      ║
  ║        │                                ║
  ║  ┌─────┴─────┐                          ║
  ║  │ W_O proj  │                          ║
  ║  └─────┬─────┘                          ║
  ║        │ + residual                     ║
  ║  ┌─────┴─────┐                          ║
  ║  │ RMSNorm   │                          ║
  ║  └─────┬─────┘                          ║
  ║  ┌─────┴─────┐                          ║
  ║  │    FFN    │                          ║
  ║  └─────┬─────┘                          ║
  ║        │ + residual                     ║
  ╚════════╪══════════════════════════════╝
           │
           ▼
        Output
```

Key points:
- RoPE is applied to Q and K **after** the linear projection, **before** the dot product
- V is **not** rotated -- only Q and K need position information for the attention scores
- Applied in **every** attention layer, so position signal is fresh at every layer
- Compatible with GQA/MQA: just apply RoPE to whatever Q and K heads exist

## Models Using RoPE

| Model | Variant | Notes |
|-------|---------|-------|
| LLaMA (1, 2, 3) | Standard RoPE | base=10000 |
| Mistral | Standard RoPE | With sliding window attention |
| Qwen (1, 1.5, 2, 2.5) | Standard RoPE | Various base frequencies |
| PaLM / Gemini | Standard RoPE | Google's models |
| Gemma | Standard RoPE | Google's open models |
| Phi (1, 2, 3) | Standard RoPE | Microsoft |
| CodeLlama | Extended RoPE | base=1000000 for 100K context |
| DeepSeek | Standard RoPE | With YaRN for long context |
| Yi | Standard RoPE | 200K context via NTK extension |

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [RoFormer (Su et al., 2021)](https://arxiv.org/abs/2104.09864) | Original RoPE formulation |
| [LLaMA (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971) | Popularized RoPE in large-scale LLMs |

## Related

- [Absolute Encoding](01_Absolute_Encoding.md) -- the older additive approach that RoPE replaced
- [ALiBi](03_ALiBi.md) -- alternative relative position method using attention biases
- [Context Extension](04_Context_Extension.md) -- techniques to extend RoPE beyond training length
