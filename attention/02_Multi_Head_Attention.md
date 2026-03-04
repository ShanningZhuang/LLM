# Multi-Head Attention

> Parent: [Attention](00_Attention.md)

## Overview

Multi-Head Attention (MHA) runs multiple self-attention operations in parallel, each with its own learned Q, K, V projections. Different heads can learn to attend to different types of relationships: some capture local syntax, others capture long-range dependencies, some focus on positional patterns, etc. The key insight is that splitting d_model into h smaller heads uses the **same total parameter count** as a single large attention -- you get diversity of attention patterns for free.

## Why Multiple Heads?

A single attention head computes one set of attention weights -- one "view" of how tokens relate. This is limiting:

```
Single head (d_model = 512):
  - One Q, K, V projection each of size 512
  - One attention pattern per token pair
  - Must capture ALL relationships in one matrix

Multi-head (h=8, d_k=64 each):
  - 8 separate Q, K, V projections each of size 64
  - 8 different attention patterns
  - Each head can specialize
```

**Empirical observations of head specialization:**

| Head | Typical learned pattern |
|------|------------------------|
| Head 1 | Attends to previous token (bigram) |
| Head 2 | Attends to sentence start |
| Head 3 | Attends to syntactic parent (e.g., verb for its object) |
| Head 4 | Attends to same-entity mentions (coreference) |
| Head 5 | Broad/uniform attention (captures global context) |
| Head 6 | Attends to punctuation/delimiters |
| Head 7 | Positional pattern (fixed offset) |
| Head 8 | Rare/specialized pattern |

## The MHA Formula

```
MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) · W_O

where head_i = Attention(Q · W_Q^i, K · W_K^i, V · W_V^i)
```

**Dimensions:**

```
d_model = 512, h = 8 heads

d_k = d_v = d_model / h = 512 / 8 = 64 per head

W_Q^i ∈ ℝ^{d_model × d_k}    →  512 × 64
W_K^i ∈ ℝ^{d_model × d_k}    →  512 × 64
W_V^i ∈ ℝ^{d_model × d_v}    →  512 × 64

W_O   ∈ ℝ^{(h·d_v) × d_model} →  512 × 512
```

## Head Splitting and Concatenation

```
Input X ∈ ℝ^{n × d_model}
         │
    ┌────▼─────────────────────────────────────────────┐
    │  Project: Q = X·W_Q, K = X·W_K, V = X·W_V       │
    │  Q, K, V each ∈ ℝ^{n × d_model}                 │
    └────┬─────────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────────┐
    │  Reshape: split d_model into h heads             │
    │                                                   │
    │  Q: (n, d_model) → (n, h, d_k) → (h, n, d_k)   │
    │  K: (n, d_model) → (n, h, d_k) → (h, n, d_k)   │
    │  V: (n, d_model) → (n, h, d_v) → (h, n, d_v)   │
    └────┬─────────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────────┐
    │  Parallel attention per head                      │
    │                                                   │
    │  head_1: Attn(Q_1, K_1, V_1) → (n, d_v)         │
    │  head_2: Attn(Q_2, K_2, V_2) → (n, d_v)         │
    │  ...                                              │
    │  head_h: Attn(Q_h, K_h, V_h) → (n, d_v)         │
    └────┬─────────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────────┐
    │  Concatenate: (h, n, d_v) → (n, h·d_v)          │
    │              = (n, d_model)                       │
    └────┬─────────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────────┐
    │  Output projection: × W_O                        │
    │  (n, d_model) · (d_model, d_model) → (n, d_model)│
    └──────────────────────────────────────────────────┘
```

## Implementation Detail: Reshape, Don't Create Separate Matrices

In practice, we don't create h separate weight matrices. Instead, we use one large projection and reshape:

```
Conceptual (h separate matrices):          Practical (one matrix + reshape):

W_Q^1 ∈ ℝ^{d × d_k}                      W_Q ∈ ℝ^{d × d_model}
W_Q^2 ∈ ℝ^{d × d_k}                      Q = X · W_Q           → (n, d_model)
...                                         Q = Q.view(n, h, d_k) → (n, h, d_k)
W_Q^h ∈ ℝ^{d × d_k}

These are mathematically equivalent: stacking h small matrices
side-by-side gives one big matrix.
```

## Parameter Count: Same as Single-Head

This is a crucial insight -- MHA does NOT add parameters compared to a single attention with the same d_model:

| Component | Single-Head (d_k = d_model) | Multi-Head (h heads, d_k = d_model/h) |
|-----------|---------------------------|--------------------------------------|
| W_Q | d_model x d_model | h x (d_model x d_k) = d_model x d_model |
| W_K | d_model x d_model | h x (d_model x d_k) = d_model x d_model |
| W_V | d_model x d_model | h x (d_model x d_v) = d_model x d_model |
| W_O | d_model x d_model | d_model x d_model |
| **Total** | **4 x d_model^2** | **4 x d_model^2** |

**Same total parameters.** MHA is a reparameterization that enables diverse attention patterns, not an increase in model capacity.

### Example: LLaMA-2 7B

```
d_model = 4096, h = 32 heads, d_k = d_v = 128

W_Q: 4096 × 4096 = 16.8M params
W_K: 4096 × 4096 = 16.8M params
W_V: 4096 × 4096 = 16.8M params
W_O: 4096 × 4096 = 16.8M params
─────────────────────────────────
Total per layer:     67.1M params
× 32 layers:          2.1B params  (≈30% of total 7B)
```

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention as described in 'Attention Is All You Need'."""

    def __init__(self, d_model: int, n_heads: int, bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head

        # Single large projection matrices (more efficient than h separate ones)
        self.W_Q = nn.Linear(d_model, d_model, bias=bias)
        self.W_K = nn.Linear(d_model, d_model, bias=bias)
        self.W_V = nn.Linear(d_model, d_model, bias=bias)
        self.W_O = nn.Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,           # (batch, seq_len, d_model)
        mask: torch.Tensor = None,  # (batch, 1, seq_len, seq_len) or broadcastable
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Step 1: Project to Q, K, V
        Q = self.W_Q(x)  # (batch, seq_len, d_model)
        K = self.W_K(x)
        V = self.W_V(x)

        # Step 2: Reshape into (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # Now: (batch, n_heads, seq_len, d_k)

        # Step 3: Scaled dot-product attention (per head, batched)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (batch, n_heads, seq_len, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        # attn_weights: (batch, n_heads, seq_len, seq_len)

        attn_output = torch.matmul(attn_weights, V)
        # attn_output: (batch, n_heads, seq_len, d_k)

        # Step 4: Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        # attn_output: (batch, seq_len, d_model)

        # Step 5: Output projection
        output = self.W_O(attn_output)
        # output: (batch, seq_len, d_model)

        return output


# --- Usage Example ---
batch_size, seq_len, d_model, n_heads = 2, 128, 512, 8

x = torch.randn(batch_size, seq_len, d_model)
mha = MultiHeadAttention(d_model=512, n_heads=8)
output = mha(x)
print(output.shape)  # torch.Size([2, 128, 512])

# Verify parameter count
total_params = sum(p.numel() for p in mha.parameters())
print(f"Total params: {total_params:,}")        # 1,048,576
print(f"Expected: 4 × {d_model}² = {4 * d_model**2:,}")  # 1,048,576
```

## Head Specialization: What Different Heads Learn

Research (Clark et al., 2019; Voita et al., 2019) has shown that attention heads in trained models specialize into distinct roles:

```
Layer 1 (early):
  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
  │ Head 0   │  │ Head 1   │  │ Head 2   │  │ Head 3   │
  │ Previous │  │ Next     │  │ Self     │  │ Position │
  │ token    │  │ token    │  │ (diag)   │  │ offset=2 │
  └─────────┘  └─────────┘  └─────────┘  └─────────┘

Layer 12 (middle):
  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
  │ Head 0   │  │ Head 1   │  │ Head 2   │  │ Head 3   │
  │ Subject  │  │ Verb→Obj │  │ Coref   │  │ Separator│
  │ →verb    │  │          │  │ links   │  │ tokens   │
  └─────────┘  └─────────┘  └─────────┘  └─────────┘

Layer 24 (late):
  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
  │ Head 0   │  │ Head 1   │  │ Head 2   │  │ Head 3   │
  │ Semantic │  │ Broad    │  │ Rare     │  │ Domain-  │
  │ similar  │  │ context  │  │ pattern  │  │ specific │
  └─────────┘  └─────────┘  └─────────┘  └─────────┘
```

**Key findings:**
- Early layers: positional/syntactic heads (attend to adjacent tokens)
- Middle layers: syntactic/semantic heads (grammatical dependencies)
- Late layers: task-specific/semantic heads (high-level reasoning)
- Many heads are "redundant" -- can be pruned with minimal quality loss

## Common Configurations

| Model | d_model | n_heads | d_k | Total attn params/layer |
|-------|---------|---------|-----|------------------------|
| GPT-2 Small | 768 | 12 | 64 | 2.4M |
| GPT-2 Large | 1280 | 20 | 64 | 6.6M |
| LLaMA-2 7B | 4096 | 32 | 128 | 67.1M |
| LLaMA-2 13B | 5120 | 40 | 128 | 104.9M |
| LLaMA-2 70B | 8192 | 64 | 128 | 268.4M |
| GPT-3 175B | 12288 | 96 | 128 | 603.9M |

Note: d_k = d_model / n_heads is almost always 64 or 128 across all model sizes. The scaling is done by increasing d_model and n_heads, not d_k.

## Key Takeaways

1. MHA splits the representation into h heads -- each head learns different attention patterns
2. The total parameter count is the same as single-head attention with the same d_model
3. Implementation: one big projection + reshape is equivalent to h separate small projections
4. Heads specialize: early heads capture positional patterns, later heads capture semantics
5. d_k is typically 64 or 128 regardless of model size

## Related

- [Self-Attention](01_Self_Attention.md) - The single-head attention that MHA is built on
- [MQA & GQA](03_MQA_GQA.md) - Variants that share K, V heads to reduce KV cache size
- [Efficient Attention](05_Efficient_Attention.md) - Making the O(n^2) computation faster
