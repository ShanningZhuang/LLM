# Self-Attention

> Parent: [Attention](00_Attention.md)

## Overview

Self-attention is the mechanism that allows each token in a sequence to "look at" every other token (including itself) and compute a weighted combination of their representations. It is the core building block of the Transformer architecture. The "self" in self-attention means the queries, keys, and values all come from the same input sequence (as opposed to cross-attention, where queries come from one sequence and keys/values from another).

## The Q, K, V Framework

Every token produces three vectors by projecting its hidden state through learned weight matrices:

```
Input X ∈ ℝ^{seq_len × d_model}

Q = X · W_Q    (Queries)    W_Q ∈ ℝ^{d_model × d_k}
K = X · W_K    (Keys)       W_K ∈ ℝ^{d_model × d_k}
V = X · W_V    (Values)     W_V ∈ ℝ^{d_model × d_v}
```

**Intuition:**
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What information do I provide if selected?"

The dot product Q · K^T measures how well a query matches each key (relevance score). The result is used to weight the values.

## Scaled Dot-Product Attention

The full formula:

```
Attention(Q, K, V) = softmax( Q · K^T / sqrt(d_k) ) · V
```

### Step-by-Step Computation

```
   X (input)
   │
   ├──→ × W_Q ──→ Q ──────────────┐
   │                                │
   ├──→ × W_K ──→ K ──→ K^T ──┐   │
   │                            │   │
   └──→ × W_V ──→ V ──┐       │   │
                       │       │   │
                       │   ┌───▼───▼────┐
                       │   │  Q · K^T    │  Dot product: (n × d_k) · (d_k × n) = (n × n)
                       │   └─────┬──────┘
                       │         │
                       │   ┌─────▼──────┐
                       │   │  ÷ sqrt(d_k)│  Scale down
                       │   └─────┬──────┘
                       │         │
                       │   ┌─────▼──────┐
                       │   │  + mask     │  (optional: causal mask)
                       │   └─────┬──────┘
                       │         │
                       │   ┌─────▼──────┐
                       │   │  softmax    │  Row-wise → attention weights (n × n)
                       │   └─────┬──────┘
                       │         │
                       │   ┌─────▼──────┐
                       └──→│ weights · V │  Weighted sum: (n × n) · (n × d_v) = (n × d_v)
                           └─────┬──────┘
                                 │
                              Output ∈ ℝ^{n × d_v}
```

## Why Scale by sqrt(d_k)?

Without scaling, the dot products Q · K^T grow in magnitude with the dimension d_k.

**Mathematical argument:** If elements of Q and K are independent with mean 0 and variance 1, then each element of Q · K^T is a sum of d_k products:

```
q · k = sum_{i=1}^{d_k} q_i * k_i

E[q · k] = 0
Var[q · k] = d_k     ← grows with dimension!
```

When d_k is large (e.g., 128), the dot products become large in magnitude. Large values push softmax into regions with extremely small gradients (the "saturated" regime), making training difficult.

Dividing by sqrt(d_k) normalizes the variance back to 1:

```
Var[q · k / sqrt(d_k)] = d_k / d_k = 1
```

| d_k | Var(q·k) without scaling | Var(q·k) with scaling |
|-----|--------------------------|----------------------|
| 64  | 64                       | 1                    |
| 128 | 128                      | 1                    |
| 256 | 256                      | 1                    |

## Worked Example: 3 Tokens

Consider a sequence of 3 tokens with d_k = 4 (tiny for illustration):

```
Input tokens: ["The", "cat", "sat"]

After projection through W_Q, W_K, W_V:

Q = [[1.0, 0.5, 0.2, 0.1],    ← "The"
     [0.3, 1.2, 0.8, 0.4],    ← "cat"
     [0.1, 0.3, 1.0, 0.9]]    ← "sat"

K = [[0.9, 0.4, 0.3, 0.2],
     [0.2, 1.1, 0.7, 0.5],
     [0.4, 0.2, 0.8, 1.0]]

V = [[0.5, 0.1, 0.8, 0.3],
     [0.2, 0.9, 0.4, 0.7],
     [0.7, 0.3, 0.6, 0.5]]
```

**Step 1: Q · K^T** (3x4 times 4x3 = 3x3)

```
        K_The  K_cat  K_sat
Q_The [ 1.17,  1.00,  0.72 ]
Q_cat [ 1.06,  1.84,  1.24 ]
Q_sat [ 0.39,  1.17,  1.62 ]
```

**Step 2: Scale by sqrt(d_k) = sqrt(4) = 2**

```
        K_The  K_cat  K_sat
Q_The [ 0.585, 0.500, 0.360 ]
Q_cat [ 0.530, 0.920, 0.620 ]
Q_sat [ 0.195, 0.585, 0.810 ]
```

**Step 3: Softmax (row-wise)**

```
              The    cat    sat
"The" attn: [0.378, 0.347, 0.275]   ← "The" attends fairly evenly
"cat" attn: [0.280, 0.413, 0.307]   ← "cat" attends most to "cat"
"sat" attn: [0.226, 0.334, 0.418]   ← "sat" attends most to itself (value 0.810 was highest)
```

**Step 4: Weighted sum of V**

```
Output for "cat" = 0.280 * V_The + 0.413 * V_cat + 0.307 * V_sat

= 0.280 * [0.5, 0.1, 0.8, 0.3]
+ 0.413 * [0.2, 0.9, 0.4, 0.7]
+ 0.307 * [0.7, 0.3, 0.6, 0.5]

= [0.140, 0.028, 0.224, 0.084]
+ [0.083, 0.372, 0.165, 0.289]
+ [0.215, 0.092, 0.184, 0.154]

= [0.438, 0.492, 0.573, 0.527]    ← new representation for "cat"
```

The new representation for "cat" is a blend of all three value vectors, weighted by how relevant each key was to the "cat" query.

## Attention Weight Interpretation

The attention weight matrix (after softmax) tells us which tokens "attend to" which:

```
Attention weights (read row-by-row):

              The    cat    sat
   "The"  → [ 0.38   0.35   0.27 ]   Row sums to 1.0
   "cat"  → [ 0.28   0.41   0.31 ]   Row sums to 1.0
   "sat"  → [ 0.23   0.33   0.42 ]   Row sums to 1.0
              ▲
              │
         Column = source token
         Row = query token (the one being updated)
```

**Reading the matrix:**
- Row i: "Where does token i get its information from?"
- Column j: "How much does token i attend to token j?"
- Diagonal: self-attention (token attending to itself)
- High off-diagonal: strong cross-token dependencies

## Complexity Analysis

| Operation | Shape | FLOPs | Memory |
|-----------|-------|-------|--------|
| Q = X·W_Q | (n,d)·(d,d_k) | O(n·d·d_k) | O(n·d_k) |
| K = X·W_K | (n,d)·(d,d_k) | O(n·d·d_k) | O(n·d_k) |
| V = X·W_V | (n,d)·(d,d_v) | O(n·d·d_v) | O(n·d_v) |
| Q·K^T | (n,d_k)·(d_k,n) | **O(n^2·d_k)** | **O(n^2)** |
| softmax | (n,n) | O(n^2) | O(n^2) |
| attn·V | (n,n)·(n,d_v) | **O(n^2·d_v)** | O(n·d_v) |

**The bottleneck is Q·K^T:** O(n^2) in both time and memory, where n = sequence length.

For a 4K context:  4096^2 = 16.7M entries per attention matrix.
For a 128K context: 131072^2 = 17.2B entries per attention matrix.

This quadratic scaling is what motivates efficient attention methods like FlashAttention and linear attention.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    Q: torch.Tensor,    # (batch, seq_len, d_k)
    K: torch.Tensor,    # (batch, seq_len, d_k)
    V: torch.Tensor,    # (batch, seq_len, d_v)
    mask: torch.Tensor = None,  # (batch, seq_len, seq_len) or broadcastable
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes scaled dot-product attention.

    Returns:
        output: (batch, seq_len, d_v)
        attention_weights: (batch, seq_len, seq_len)
    """
    d_k = Q.size(-1)

    # Step 1: Q · K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores shape: (batch, seq_len, seq_len)

    # Step 2: Apply mask (e.g., causal mask)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 3: Softmax over the key dimension (last dim)
    attention_weights = F.softmax(scores, dim=-1)

    # Step 4: Weighted sum of values
    output = torch.matmul(attention_weights, V)
    # output shape: (batch, seq_len, d_v)

    return output, attention_weights


class SelfAttention(nn.Module):
    """Single-head self-attention layer."""

    def __init__(self, d_model: int, d_k: int = None, d_v: int = None):
        super().__init__()
        d_k = d_k or d_model
        d_v = d_v or d_model

        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)
        self.W_O = nn.Linear(d_v, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) or None
        Returns:
            output: (batch, seq_len, d_model)
        """
        Q = self.W_Q(x)   # (batch, seq_len, d_k)
        K = self.W_K(x)   # (batch, seq_len, d_k)
        V = self.W_V(x)   # (batch, seq_len, d_v)

        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        output = self.W_O(attn_output)  # (batch, seq_len, d_model)
        return output


# --- Usage Example ---
batch_size, seq_len, d_model = 2, 10, 512
x = torch.randn(batch_size, seq_len, d_model)

attn = SelfAttention(d_model=512, d_k=64, d_v=64)
output = attn(x)
print(output.shape)  # torch.Size([2, 10, 512])
```

## Self-Attention vs Other Attention Types

| Type | Q source | K, V source | Use case |
|------|----------|-------------|----------|
| Self-Attention | Same input X | Same input X | Decoder-only LLMs (GPT) |
| Cross-Attention | Decoder hidden states | Encoder output | Encoder-decoder (T5, BART) |
| Masked Self-Attention | Same input X (masked) | Same input X (masked) | Causal / autoregressive |

## Key Takeaways

1. Self-attention lets every token interact with every other token through Q, K, V projections
2. The scaling factor 1/sqrt(d_k) prevents softmax saturation as dimensions grow
3. Attention weights form an n x n matrix -- this is the O(n^2) bottleneck
4. The output for each token is a weighted average of all value vectors, where weights reflect relevance

## Related

- [Multi-Head Attention](02_Multi_Head_Attention.md) - Running multiple attention heads in parallel
- [Causal Mask](04_Causal_Mask.md) - Masking future tokens for autoregressive generation
- [Efficient Attention](05_Efficient_Attention.md) - Overcoming the O(n^2) bottleneck
