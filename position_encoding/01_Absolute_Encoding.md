# Absolute Position Encoding

> Parent: [Position Encoding](00_Position_Encoding.md)

## Overview

Absolute position encoding assigns a fixed vector to each position in the sequence and adds it to the token embedding. The original Transformer (Vaswani et al., 2017) used sinusoidal functions to generate these vectors deterministically. Later models like GPT-2 and BERT switched to learned position embeddings. Both approaches are **additive** -- the position signal is summed with the token embedding before entering the Transformer blocks.

Modern LLMs have largely moved away from absolute encoding because it cannot generalize to sequence lengths beyond those seen during training.

## How Absolute Encoding Enters the Model

```
Token IDs:        [  23,    891,    7,    1042  ]
                    │       │      │       │
                    ▼       ▼      ▼       ▼
              ┌──────────────────────────────────┐
              │     Token Embedding Table         │
              │     nn.Embedding(vocab, d_model)  │
              └──────────────────────────────────┘
                    │       │      │       │
                    ▼       ▼      ▼       ▼
              e_0 [0.3,..] e_1   e_2     e_3        ← token embeddings

Position IDs:  [   0,      1,     2,      3   ]
                    │       │      │       │
                    ▼       ▼      ▼       ▼
              ┌──────────────────────────────────┐
              │     Position Encoding             │
              │     (sinusoidal or learned)        │
              └──────────────────────────────────┘
                    │       │      │       │
                    ▼       ▼      ▼       ▼
              p_0 [0.1,..] p_1   p_2     p_3        ← position embeddings

                    │       │      │       │
                    ▼       ▼      ▼       ▼
              ┌──────────────────────────────────┐
              │         Element-wise Addition      │
              │   input_i = e_i + p_i              │
              └──────────────────────────────────┘
                    │       │      │       │
                    ▼       ▼      ▼       ▼
              x_0          x_1   x_2     x_3        ← fed into Transformer
```

Both sinusoidal and learned encoding are applied **once** at the input, not at every layer. The position signal must survive through all Transformer blocks via the residual stream.

## Sinusoidal Encoding (Vaswani et al., 2017)

### Formulas

For position `pos` and dimension index `i` (where `d_model` is the embedding dimension):

```
PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
```

Each dimension uses a sinusoid of a different frequency. Low dimensions oscillate rapidly (high frequency), while high dimensions oscillate slowly (low frequency):

```
dim 0,1:   frequency = 1 / 10000^{0/d}     = 1.0       (fast oscillation)
dim 2,3:   frequency = 1 / 10000^{2/d}     ~ 0.9998    (slightly slower)
...
dim d-2,d-1: frequency = 1 / 10000^{1}     = 0.0001    (very slow)
```

### Properties

| Property | Explanation |
|----------|-------------|
| Bounded | Values always in [-1, 1] due to sin/cos |
| Unique | Each position gets a distinct vector (like a binary counter but smooth) |
| Deterministic | No parameters to learn, computed from formulas |
| Relative position via linear transform | PE(pos+k) can be written as a linear function of PE(pos) for any fixed k |
| No max length | Defined for any pos >= 0, though quality degrades for unseen positions |

### Relative Position as Linear Transformation

A key design insight: for any fixed offset `k`, there exists a matrix `M_k` such that:

```
PE(pos + k) = M_k * PE(pos)

where M_k is a block-diagonal matrix of 2x2 rotation matrices:

M_k = diag(R(k*w_0), R(k*w_1), ..., R(k*w_{d/2-1}))

     ┌ cos(k*w_i)  -sin(k*w_i) ┐
R =  │                          │
     └ sin(k*w_i)   cos(k*w_i) ┘
```

This means the model can, in principle, learn to compute relative positions from the absolute sinusoidal encoding. However, in practice the model must learn this transformation through the attention weights, which is inefficient compared to methods that encode relative position directly.

## Learned Position Embeddings (GPT-2, BERT)

Instead of using fixed formulas, learned position embeddings treat each position as a learnable vector:

```
position_embeddings = nn.Embedding(max_seq_len, d_model)
```

The model learns the position vectors during training, just like it learns token embeddings. This is simpler to implement and slightly more flexible, but introduces a hard ceiling on sequence length.

| Property | Detail |
|----------|--------|
| Parameterized | `max_seq_len * d_model` additional parameters |
| Max length | Fixed at training time (512 for BERT, 1024 for GPT-2) |
| Extrapolation | Cannot handle positions beyond max_seq_len at all |
| Quality | Slightly better than sinusoidal for in-distribution lengths |

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import math


class SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal position encoding from 'Attention Is All You Need'."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)                # (max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)      # (max_len, 1)

        # Compute the div term: 10000^{2i/d_model}
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )                                                   # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)       # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)       # odd dimensions

        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer("pe", pe.unsqueeze(0))         # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: token embeddings, shape (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1)]


class LearnedPositionEncoding(nn.Module):
    """Learned position embeddings (GPT-2, BERT style)."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: token embeddings, shape (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)  # (seq_len,)
        return x + self.pos_embedding(positions)             # broadcast over batch


# --- Demo ---
if __name__ == "__main__":
    batch, seq_len, d_model = 2, 32, 64

    token_embed = torch.randn(batch, seq_len, d_model)

    # Sinusoidal
    sin_enc = SinusoidalPositionEncoding(d_model)
    out_sin = sin_enc(token_embed)
    print(f"Sinusoidal: input {token_embed.shape} -> output {out_sin.shape}")
    print(f"  PE values at pos=0, dims 0..3: {sin_enc.pe[0, 0, :4].tolist()}")

    # Learned
    learn_enc = LearnedPositionEncoding(d_model)
    out_learn = learn_enc(token_embed)
    print(f"Learned:    input {token_embed.shape} -> output {out_learn.shape}")
    print(f"  Params: {sum(p.numel() for p in learn_enc.parameters())}")
```

## Sinusoidal vs Learned: Comparison

| Aspect | Sinusoidal | Learned |
|--------|-----------|---------|
| Parameters | 0 | max_len * d_model |
| Max sequence length | Theoretically unlimited | Fixed at max_len |
| Extrapolation | Degrades gracefully | Fails completely |
| Quality (in-distribution) | Good | Slightly better |
| Implementation | Compute once, cache | Standard nn.Embedding |
| Used by | Original Transformer, some encoder models | GPT-2, BERT, GPT-3 |
| Deterministic | Yes | No (random init, then learned) |

In the original Transformer paper, the authors found no significant difference between sinusoidal and learned embeddings for in-distribution lengths. They chose sinusoidal for the theoretical ability to extrapolate, though in practice this extrapolation is limited.

## Why Modern LLMs Moved Away from Absolute Encoding

### Problem 1: No Length Generalization

Absolute encodings assign meaning to specific positions. A model trained with max_len=512 has no way to handle position 513:
- Learned: the embedding table simply has no entry for position 513
- Sinusoidal: values are defined but the model has never seen them during training

### Problem 2: Position Signal Degrades Through Layers

Since absolute encoding is added once at the input, the position information must survive through many layers of attention and FFN transformations. In deep models (32-96 layers), this signal can degrade.

### Problem 3: Attention Cannot Easily Extract Relative Position

Many language tasks depend on relative position ("the word 3 tokens ago") rather than absolute position ("the word at position 47"). With absolute encoding, the model must learn to subtract positions through the attention mechanism, which is indirect and wasteful.

```
Absolute:  "What is at position 5?"  → natural
           "What is 3 positions back?" → must learn subtraction

Relative:  "What is at position 5?"  → less natural
           "What is 3 positions back?" → natural
```

### The Shift to Relative Methods

| Generation | Method | Problem Solved |
|------------|--------|---------------|
| 2017 | Sinusoidal (absolute) | Basic position awareness |
| 2019 | Learned (absolute) | Slightly better in-distribution |
| 2021 | RoPE (relative, rotary) | Direct relative position in attention |
| 2022 | ALiBi (relative, bias) | Length generalization without embedding |
| 2023+ | Extended RoPE (YaRN, PI) | Long context from short training |

The dominant modern approach is RoPE, which encodes position by rotating query and key vectors inside attention, naturally capturing relative position without any additive embedding at the input.

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) | Sinusoidal position encoding |
| [BERT (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805) | Learned position embeddings (512 max) |
| [GPT-2 (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | Learned position embeddings (1024 max) |

## Related

- [RoPE](02_RoPE.md) -- the dominant modern approach using rotary position encoding
- [ALiBi](03_ALiBi.md) -- bias-based alternative to position embeddings
- [Context Extension](04_Context_Extension.md) -- techniques to extend beyond training length
