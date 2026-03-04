# Causal Mask

> Parent: [Attention](00_Attention.md)

## Overview

In autoregressive language models, each token must only attend to itself and the tokens that came before it -- never to future tokens. The causal mask (also called the autoregressive mask or look-ahead mask) enforces this constraint by setting future positions to negative infinity before the softmax, ensuring they receive zero attention weight. This is what makes a decoder-only model generate text left-to-right, one token at a time.

## Why Causal Masking?

An autoregressive model predicts the next token given all previous tokens:

```
P(x_1, x_2, ..., x_n) = P(x_1) · P(x_2|x_1) · P(x_3|x_1,x_2) · ... · P(x_n|x_1,...,x_{n-1})
```

During training, we process the entire sequence at once (teacher forcing). Without masking, token x_3 could "see" x_4, x_5, ... -- it would be cheating. The causal mask prevents this:

```
Without mask (bidirectional):         With causal mask (autoregressive):

Token 1 sees: [1, 2, 3, 4, 5]       Token 1 sees: [1]
Token 2 sees: [1, 2, 3, 4, 5]       Token 2 sees: [1, 2]
Token 3 sees: [1, 2, 3, 4, 5]       Token 3 sees: [1, 2, 3]
Token 4 sees: [1, 2, 3, 4, 5]       Token 4 sees: [1, 2, 3, 4]
Token 5 sees: [1, 2, 3, 4, 5]       Token 5 sees: [1, 2, 3, 4, 5]
```

## How the Mask Works

The mask is applied to the attention scores (after Q*K^T, before softmax):

```
Step 1: Compute raw attention scores

  scores = Q · K^T / sqrt(d_k)

         K_1    K_2    K_3    K_4    K_5
  Q_1 [  0.8    1.2    0.3    0.5    0.1  ]
  Q_2 [  0.4    1.5    0.7    0.9    0.3  ]
  Q_3 [  0.2    0.6    1.8    0.4    0.8  ]
  Q_4 [  0.3    0.1    0.5    1.6    0.7  ]
  Q_5 [  0.1    0.4    0.3    0.6    1.9  ]


Step 2: Create causal mask (upper triangle = -inf)

         K_1    K_2    K_3    K_4    K_5
  Q_1 [  0     -inf   -inf   -inf   -inf  ]
  Q_2 [  0      0     -inf   -inf   -inf  ]
  Q_3 [  0      0      0     -inf   -inf  ]
  Q_4 [  0      0      0      0     -inf  ]
  Q_5 [  0      0      0      0      0    ]


Step 3: Add mask to scores

         K_1    K_2    K_3    K_4    K_5
  Q_1 [  0.8   -inf   -inf   -inf   -inf  ]
  Q_2 [  0.4    1.5   -inf   -inf   -inf  ]
  Q_3 [  0.2    0.6    1.8   -inf   -inf  ]
  Q_4 [  0.3    0.1    0.5    1.6   -inf  ]
  Q_5 [  0.1    0.4    0.3    0.6    1.9  ]


Step 4: Softmax (row-wise) — -inf becomes 0

         K_1    K_2    K_3    K_4    K_5
  Q_1 [ 1.000  0.000  0.000  0.000  0.000 ]  ← token 1 only sees itself
  Q_2 [ 0.250  0.750  0.000  0.000  0.000 ]  ← token 2 sees 1 and 2
  Q_3 [ 0.125  0.187  0.621  0.000  0.000 ]  ← token 3 sees 1, 2, 3
  Q_4 [ 0.128  0.105  0.157  0.470  0.000 ]  ← token 4 sees 1, 2, 3, 4
  Q_5 [ 0.082  0.111  0.101  0.136  0.500 ]  ← token 5 sees all
```

## ASCII Diagram: Mask Application

```
Before masking:                     After masking:
(full attention matrix)             (causal / lower triangular)

  ┌───┬───┬───┬───┬───┐            ┌───┬───┬───┬───┬───┐
  │ ■ │ ■ │ ■ │ ■ │ ■ │            │ ■ │   │   │   │   │
  ├───┼───┼───┼───┼───┤            ├───┼───┼───┼───┼───┤
  │ ■ │ ■ │ ■ │ ■ │ ■ │            │ ■ │ ■ │   │   │   │
  ├───┼───┼───┼───┼───┤     →      ├───┼───┼───┼───┼───┤
  │ ■ │ ■ │ ■ │ ■ │ ■ │            │ ■ │ ■ │ ■ │   │   │
  ├───┼───┼───┼───┼───┤            ├───┼───┼───┼───┼───┤
  │ ■ │ ■ │ ■ │ ■ │ ■ │            │ ■ │ ■ │ ■ │ ■ │   │
  ├───┼───┼───┼───┼───┤            ├───┼───┼───┼───┼───┤
  │ ■ │ ■ │ ■ │ ■ │ ■ │            │ ■ │ ■ │ ■ │ ■ │ ■ │
  └───┴───┴───┴───┴───┘            └───┴───┴───┴───┴───┘

  ■ = attended (nonzero weight)     Upper triangle masked to 0
  All n² entries computed           Only n(n+1)/2 entries matter
```

## Types of Masks

| Mask Type | Purpose | Where Used |
|-----------|---------|------------|
| **Causal mask** | Prevent attending to future tokens | Decoder-only models (GPT, LLaMA) |
| **Padding mask** | Ignore padding tokens in batched input | Any model with variable-length inputs |
| **Cross-attention mask** | Control which encoder tokens the decoder attends to | Encoder-decoder models (T5, BART) |
| **Sliding window mask** | Limit attention to a local window | Mistral, Longformer |

### Causal Mask vs Padding Mask

```
Causal mask (structural):              Padding mask (data-dependent):

Sequence: "The cat sat"                Batch: ["The cat", "Hi _ _"]
                                        (_ = padding)
  The cat sat                            The cat  Hi  _   _
  ┌───┬───┬───┐                          ┌───┬───┬───┬───┬───┐
  │ 1 │ 0 │ 0 │  The                    │ 1 │ 1 │ 1 │ 0 │ 0 │  The
  │ 1 │ 1 │ 0 │  cat                    │ 1 │ 1 │ 1 │ 0 │ 0 │  cat
  │ 1 │ 1 │ 1 │  sat                    │ 1 │ 1 │ 1 │ 0 │ 0 │  Hi
  └───┴───┴───┘                          └───┴───┴───┴───┴───┘

Same for all sequences                  Different for each sequence
in the batch                            (depends on actual lengths)
```

When both masks are needed, they are combined (element-wise AND or addition):

```
Combined = causal_mask AND padding_mask

  The cat  Hi  _   _
  ┌───┬───┬───┬───┬───┐
  │ 1 │ 0 │ 0 │ 0 │ 0 │  The    (causal: only see self; padding: ignore _ _)
  │ 1 │ 1 │ 0 │ 0 │ 0 │  cat
  │ 1 │ 1 │ 1 │ 0 │ 0 │  Hi
  │ 0 │ 0 │ 0 │ 0 │ 0 │  _      (padding tokens attend to nothing)
  │ 0 │ 0 │ 0 │ 0 │ 0 │  _
  └───┴───┴───┴───┴───┘
```

## Efficient Implementation

In practice, we don't need to compute the upper triangle at all. FlashAttention exploits this by only processing the lower-triangular tiles:

```
Standard (wasteful):                  Efficient (FlashAttention-aware):

  Compute full n×n scores             Only compute lower triangle
  Apply mask (-inf)                    Skip upper triangle entirely
  Softmax zeros them out               ~50% compute saved for
  Wasted ~50% computation              the masking step
```

However, the naive implementation still computes the full matrix and masks it, because GPU parallelism makes it efficient to compute and discard:

```python
# Naive: compute all, then mask
scores = Q @ K.T / sqrt(d_k)      # full n×n
scores = scores + causal_mask       # add -inf to upper triangle
weights = softmax(scores)           # zeros in upper triangle
```

## PyTorch Implementation

```python
import torch
import torch.nn.functional as F


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Creates a causal (lower-triangular) mask.

    Returns:
        mask: (seq_len, seq_len) with 0 in lower triangle, -inf in upper triangle
    """
    # torch.triu returns upper triangular part
    # diagonal=1 means start one above the main diagonal
    mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=device),
        diagonal=1
    )
    return mask
    # Result:
    #   [[ 0,   -inf, -inf, -inf],
    #    [ 0,    0,   -inf, -inf],
    #    [ 0,    0,    0,   -inf],
    #    [ 0,    0,    0,    0  ]]


def causal_attention(
    Q: torch.Tensor,  # (batch, n_heads, seq_len, d_k)
    K: torch.Tensor,  # (batch, n_heads, seq_len, d_k)
    V: torch.Tensor,  # (batch, n_heads, seq_len, d_k)
) -> torch.Tensor:
    """Scaled dot-product attention with causal mask."""
    d_k = Q.size(-1)
    seq_len = Q.size(-2)

    # Attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    # scores: (batch, n_heads, seq_len, seq_len)

    # Apply causal mask (additive: -inf makes softmax output 0)
    causal_mask = create_causal_mask(seq_len, device=Q.device)
    scores = scores + causal_mask  # broadcasts over batch and heads

    # Softmax and weighted sum
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)

    return output


# --- Alternative: Using torch.nn.functional.scaled_dot_product_attention ---
# PyTorch 2.0+ has built-in support with is_causal flag:

def causal_attention_native(Q, K, V):
    """Uses PyTorch's built-in SDPA with causal mask (uses FlashAttention when available)."""
    return F.scaled_dot_product_attention(
        Q, K, V,
        is_causal=True,  # automatically applies causal mask
        # This uses FlashAttention or memory-efficient attention under the hood
    )


# --- Demo ---
batch, n_heads, seq_len, d_k = 2, 8, 16, 64

Q = torch.randn(batch, n_heads, seq_len, d_k)
K = torch.randn(batch, n_heads, seq_len, d_k)
V = torch.randn(batch, n_heads, seq_len, d_k)

output = causal_attention(Q, K, V)
print(output.shape)  # torch.Size([2, 8, 16, 64])

# Verify: first token only attends to itself
mask = create_causal_mask(5)
print(mask)
# tensor([[0., -inf, -inf, -inf, -inf],
#         [0., 0., -inf, -inf, -inf],
#         [0., 0., 0., -inf, -inf],
#         [0., 0., 0., 0., -inf],
#         [0., 0., 0., 0., 0.]])
```

## Prefill vs Decode: When the Mask Matters

During inference, the autoregressive generation has two phases with different masking needs:

```
Phase 1: PREFILL (process the prompt)
──────────────────────────────────────
Input: "The cat sat on"  (4 tokens processed in parallel)

  Need causal mask:
    The  cat  sat  on
    ┌───┬───┬───┬───┐
    │ ■ │   │   │   │  The
    │ ■ │ ■ │   │   │  cat
    │ ■ │ ■ │ ■ │   │  sat
    │ ■ │ ■ │ ■ │ ■ │  on
    └───┴───┴───┴───┘

  Multiple query tokens → mask is essential to prevent
  "sat" from seeing "on" during prefill.

Phase 2: DECODE (generate one token at a time)
──────────────────────────────────────
Generate: "the" (1 new token, attends to all cached K,V)

  Q = just the new token's query
  K, V = cached keys/values from all previous tokens

  Q has shape (1, d_k)     ← single query
  K has shape (5, d_k)     ← all previous tokens

  scores = Q · K^T  → shape (1, 5)

    The  cat  sat  on  the
    ┌───┬───┬───┬───┬───┐
    │ ■ │ ■ │ ■ │ ■ │ ■ │  "the" (new token)
    └───┴───┴───┴───┴───┘

  No mask needed! The new token naturally only sees
  previous tokens (they're the only ones in the KV cache).
```

| Phase | Mask needed? | Why |
|-------|-------------|-----|
| Prefill | Yes | Multiple tokens processed in parallel; must prevent forward-looking |
| Decode | No | Only one query token; KV cache contains only past tokens |

## Connection to Autoregressive Property

The causal mask is what enables **parallel training** of an autoregressive model:

```
Without causal mask:
  Must train token-by-token (sequential, slow)
  Feed x_1 → predict x_2
  Feed x_1, x_2 → predict x_3
  ...

With causal mask:
  Process entire sequence at once (parallel, fast)
  Each position i predicts x_{i+1}
  Mask ensures position i cannot see x_{i+1}, ..., x_n
  Loss computed simultaneously for all positions

  This is why Transformer training is much faster than RNN training!
```

The loss function sums next-token prediction loss over all positions:

```
L = -1/n * sum_{i=1}^{n-1} log P(x_{i+1} | x_1, ..., x_i)

Each term only depends on tokens up to position i (guaranteed by causal mask).
```

## Key Takeaways

1. The causal mask prevents tokens from attending to future positions, enforcing left-to-right generation
2. Implementation: add a matrix of -inf values (upper triangle) to attention scores before softmax
3. Three common mask types: causal (structural), padding (data-dependent), cross-attention (encoder-decoder)
4. During decode phase, no explicit mask is needed since the KV cache only contains past tokens
5. The causal mask enables parallel training of autoregressive models (process full sequence at once)

## Related

- [Self-Attention](01_Self_Attention.md) - The attention mechanism that the mask modifies
- [Efficient Attention](05_Efficient_Attention.md) - FlashAttention exploits the triangular structure
- [Multi-Head Attention](02_Multi_Head_Attention.md) - Mask is applied identically to all heads
