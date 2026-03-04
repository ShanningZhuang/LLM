# Multi-Query Attention (MQA) & Grouped-Query Attention (GQA)

> Parent: [Attention](00_Attention.md)

## Overview

During autoregressive inference, the KV cache stores the keys and values of all previous tokens for every layer and every head. This cache grows linearly with the number of attention heads and becomes a major memory bottleneck -- especially for long sequences and large batch sizes. Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) reduce KV cache size by sharing key/value heads across multiple query heads, trading a small quality loss for dramatically lower memory usage and faster inference.

## The Problem: KV Cache Memory

During generation, each new token only needs to compute its own Q, but needs K, V from ALL previous tokens. We cache these to avoid recomputation:

```
Standard MHA KV cache per layer:

  K cache: (batch, n_heads, seq_len, d_k)
  V cache: (batch, n_heads, seq_len, d_k)

Example: LLaMA-2 7B, batch=1, seq=4096
  n_heads=32, d_k=128, n_layers=32, dtype=fp16 (2 bytes)

  KV cache = 2 × 32 × 4096 × 128 × 32 × 2 bytes
           = 2 × 32 × 4096 × 128 × 32 × 2
           = 2,147,483,648 bytes
           = 2 GB   ← just for KV cache!

With batch=32: 64 GB of KV cache alone.
```

## MHA vs MQA vs GQA

```
Multi-Head Attention (MHA):          Multi-Query Attention (MQA):
Every head has its own K, V          ALL heads share ONE K, V

Q heads:  Q1  Q2  Q3  Q4  Q5  Q6    Q heads:  Q1  Q2  Q3  Q4  Q5  Q6
          │   │   │   │   │   │               │   │   │   │   │   │
K heads:  K1  K2  K3  K4  K5  K6    K heads:  └───┴───┴───┼───┴───┘
          │   │   │   │   │   │                           │
V heads:  V1  V2  V3  V4  V5  V6    V heads:             K1 (shared)
                                                          │
                                                          V1 (shared)

Grouped-Query Attention (GQA):
Groups of heads share K, V

Q heads:  Q1  Q2  Q3  Q4  Q5  Q6
          │   │   │   │   │   │
          └───┤   └───┤   └───┤
              │       │       │
K heads:      K1      K2      K3    (n_kv_heads = 3, group_size = 2)
              │       │       │
V heads:      V1      V2      V3
```

### Summary

| Method | Q heads | KV heads | KV cache reduction |
|--------|---------|----------|-------------------|
| MHA | n_heads | n_heads | 1x (baseline) |
| GQA | n_heads | n_kv_heads | n_heads / n_kv_heads |
| MQA | n_heads | 1 | n_heads |

## KV Cache Size Comparison

Formula: `KV_cache = 2 × n_kv_heads × d_k × seq_len × n_layers × bytes_per_param`

| Config | n_kv_heads | KV Cache (seq=4096, fp16) | Relative |
|--------|-----------|---------------------------|----------|
| MHA (32 heads) | 32 | 2 x 32 x 128 x 4096 x 32 x 2 = 2.0 GB | 1.0x |
| GQA (8 KV heads) | 8 | 2 x 8 x 128 x 4096 x 32 x 2 = 0.5 GB | 0.25x |
| GQA (4 KV heads) | 4 | 2 x 4 x 128 x 4096 x 32 x 2 = 0.25 GB | 0.125x |
| MQA (1 KV head) | 1 | 2 x 1 x 128 x 4096 x 32 x 2 = 64 MB | 0.03x |

**4x less KV cache with GQA-8, 32x less with MQA.** This directly translates to:
- Larger batch sizes (more users served simultaneously)
- Longer context lengths fitting in GPU memory
- Faster inference (less memory bandwidth for loading KV cache)

## Quality Comparison

From the GQA paper (Ainslie et al., 2023):

```
Quality (higher is better):

MHA  ████████████████████████████  (baseline, best quality)
GQA  ███████████████████████████   (very close to MHA)
MQA  ██████████████████████████    (slight degradation)

Inference speed (higher is better):

MHA  ██████████                    (baseline, slowest)
GQA  █████████████████████         (much faster)
MQA  ████████████████████████████  (fastest)
```

| Method | Quality | Inference Speed | Memory |
|--------|---------|-----------------|--------|
| MHA | Best | Baseline | Highest |
| GQA | Near-MHA (~0.1-0.3% loss) | 1.5-2x faster | n_heads/n_kv_heads reduction |
| MQA | Slight degradation | 2-3x faster | n_heads x reduction |

**GQA hits the sweet spot:** nearly MHA quality with significant speed/memory gains.

## Example: LLaMA-2 Model Family

LLaMA-2 chose different strategies at different scales:

| Model | n_heads | n_kv_heads | Method | KV cache (4K, fp16) |
|-------|---------|-----------|--------|-------------------|
| LLaMA-2 7B | 32 | 32 | MHA | 2.0 GB |
| LLaMA-2 13B | 40 | 40 | MHA | 3.1 GB |
| LLaMA-2 70B | 64 | 8 | **GQA** | 0.5 GB |

Why only 70B uses GQA:
- At 70B scale, the model is large enough that the quality impact of GQA is negligible
- The KV cache savings are critical: 64 KV heads would need 4 GB, 8 KV heads need 0.5 GB
- Smaller models (7B, 13B) can afford MHA because their KV cache is manageable

Note: LLaMA-3 uses GQA at all sizes (8B: 8 KV heads for 32 Q heads).

## How GQA Works in Detail

```
Example: n_heads = 8, n_kv_heads = 2 (group_size = 4)

Query heads:     Q0  Q1  Q2  Q3 │ Q4  Q5  Q6  Q7
                  │   │   │   │ │  │   │   │   │
                  └───┴───┴───┘ │  └───┴───┴───┘
                        │       │        │
KV heads:              K0, V0   │      K1, V1
                                │
                        Group 0 │ Group 1

For Q0, Q1, Q2, Q3:
  scores = Q_i · K_0^T / sqrt(d_k)     ← all use K_0
  output = softmax(scores) · V_0        ← all use V_0

For Q4, Q5, Q6, Q7:
  scores = Q_i · K_1^T / sqrt(d_k)     ← all use K_1
  output = softmax(scores) · V_1        ← all use V_1
```

The query heads within a group still produce different attention patterns (different Q projections), but they all attend using the same K and V. This means the "what to attend to" (K) and "what information to retrieve" (V) are shared within a group, but "what to look for" (Q) remains unique per head.

## Converting MHA to GQA (Uptraining)

You can convert an already-trained MHA model to GQA without training from scratch:

```
Step 1: Group the existing KV heads

  Original: K0, K1, K2, K3, K4, K5, K6, K7  (8 KV heads)
  Target:   G0 = {K0,K1,K2,K3}, G1 = {K4,K5,K6,K7}  (2 groups)

Step 2: Mean-pool within each group

  K_new_0 = mean(K0, K1, K2, K3)   ← average the weight matrices
  K_new_1 = mean(K4, K5, K6, K7)

  (Same for V heads)

Step 3: Uptrain with small additional compute

  - Train for ~5% of original pretraining tokens
  - The model quickly adapts to the shared KV structure
  - Quality recovers to near-MHA levels
```

This is much cheaper than training a GQA model from scratch, which is why LLaMA-2 70B was first trained as MHA then converted.

## Which Models Use What

| Model | Method | n_heads | n_kv_heads | Notes |
|-------|--------|---------|-----------|-------|
| GPT-2 | MHA | 12-20 | 12-20 | Original multi-head |
| GPT-3 | MHA | 96 | 96 | Standard MHA |
| GPT-J (6B) | MQA | 16 | 1 | Early MQA adoption |
| PaLM (540B) | MQA | 48 | 1 | MQA for efficiency at scale |
| LLaMA-2 7B | MHA | 32 | 32 | MHA at small scale |
| LLaMA-2 70B | GQA | 64 | 8 | GQA for large scale |
| LLaMA-3 8B | GQA | 32 | 8 | GQA at all scales |
| LLaMA-3 70B | GQA | 64 | 8 | GQA standard |
| Mistral 7B | GQA | 32 | 8 | GQA at 7B |
| Falcon 40B | MQA | 64 | 1 | Aggressive MQA |
| Gemma 7B | MHA | 16 | 16 | Full MHA, fewer heads |

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).
    Supports MHA (n_kv_heads == n_heads), MQA (n_kv_heads == 1),
    and GQA (1 < n_kv_heads < n_heads).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int = None,  # None defaults to MHA
        bias: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads  # default: MHA
        self.n_groups = n_heads // self.n_kv_heads  # queries per KV head
        self.d_k = d_model // n_heads

        assert n_heads % self.n_kv_heads == 0, \
            "n_heads must be divisible by n_kv_heads"

        # Q projection: full size (all query heads)
        self.W_Q = nn.Linear(d_model, n_heads * self.d_k, bias=bias)
        # K, V projections: reduced size (only n_kv_heads)
        self.W_K = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=bias)
        self.W_V = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=bias)
        # Output projection: full size
        self.W_O = nn.Linear(n_heads * self.d_k, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,           # (batch, seq_len, d_model)
        mask: torch.Tensor = None,  # (batch, 1, seq_len, seq_len)
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Project
        Q = self.W_Q(x)  # (batch, seq_len, n_heads * d_k)
        K = self.W_K(x)  # (batch, seq_len, n_kv_heads * d_k)
        V = self.W_V(x)  # (batch, seq_len, n_kv_heads * d_k)

        # Reshape into heads
        Q = Q.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        # Q: (batch, n_heads, seq_len, d_k)
        # K, V: (batch, n_kv_heads, seq_len, d_k)

        # Expand K, V to match Q's head count by repeating
        # Each KV head is shared by n_groups query heads
        if self.n_groups > 1:
            K = K.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1)
            K = K.reshape(batch, self.n_heads, seq_len, self.d_k)
            V = V.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1)
            V = V.reshape(batch, self.n_heads, seq_len, self.d_k)
        # Now K, V: (batch, n_heads, seq_len, d_k) -- same shape as Q

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, self.n_heads * self.d_k)

        return self.W_O(attn_output)


# --- Usage Examples ---
d_model = 4096

# MHA: all 32 heads have their own KV
mha = GroupedQueryAttention(d_model, n_heads=32, n_kv_heads=32)

# GQA: 32 query heads, 8 KV heads (LLaMA-2 70B style)
gqa = GroupedQueryAttention(d_model, n_heads=32, n_kv_heads=8)

# MQA: 32 query heads, 1 KV head
mqa = GroupedQueryAttention(d_model, n_heads=32, n_kv_heads=1)

# Compare parameter counts
for name, module in [("MHA", mha), ("GQA-8", gqa), ("MQA", mqa)]:
    total = sum(p.numel() for p in module.parameters())
    kv_params = sum(p.numel() for p in [module.W_K.weight, module.W_V.weight])
    print(f"{name:6s}: total={total:>12,}  KV_params={kv_params:>10,}")

# Output:
# MHA   : total=  67,108,864  KV_params=33,554,432
# GQA-8 : total=  41,943,040  KV_params= 8,388,608
# MQA   : total=  35,651,584  KV_params= 1,048,576
```

## Key Takeaways

1. KV cache is the memory bottleneck during inference -- it scales with n_kv_heads x seq_len x n_layers
2. MQA shares one KV head across all query heads: maximum compression, some quality loss
3. GQA groups query heads to share KV heads: near-MHA quality, significant memory savings
4. GQA is now the industry standard for large models (LLaMA-3, Mistral, etc.)
5. Existing MHA models can be converted to GQA by mean-pooling KV heads and uptraining

## Related

- [Multi-Head Attention](02_Multi_Head_Attention.md) - Standard MHA that GQA/MQA modify
- [Efficient Attention](05_Efficient_Attention.md) - FlashAttention (complementary: reduces compute, while GQA reduces memory)
- [Self-Attention](01_Self_Attention.md) - Foundation: the scaled dot-product used inside each head
