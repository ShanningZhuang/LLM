# Token Embedding

> Parent: [Embedding](00_Embedding.md)

## Overview

The token embedding layer is the first neural network component in an LLM. It converts each discrete token ID into a continuous dense vector of dimension d_model. This is implemented as a simple **lookup table** -- no matrix multiplication, no activation function, just indexing into a learned matrix. Despite its simplicity, the embedding layer accounts for a significant fraction of total parameters, especially when the vocabulary is large.

## Embedding as a Lookup Table

```
Vocabulary size V = 32,000
Embedding dimension d = 4,096

Embedding matrix E ∈ ℝ^{V × d_model}    (32,000 × 4,096 = 131M parameters)

┌──────────────────────────────────────────┐
│  E[0]     = [0.012, -0.034, ..., 0.021]  │  ← token 0 ("<unk>")
│  E[1]     = [0.005,  0.017, ..., -0.008] │  ← token 1 ("<s>")
│  E[2]     = [-0.011, 0.042, ..., 0.015]  │  ← token 2 ("the")
│  ...                                      │
│  E[15496] = [0.023, -0.019, ..., 0.031]  │  ← token 15496 ("Hello")
│  ...                                      │
│  E[31999] = [0.008,  0.011, ..., -0.027] │  ← token 31999
└──────────────────────────────────────────┘

Input:   token_id = 15496
Output:  E[15496] = [0.023, -0.019, ..., 0.031]    (d_model-dimensional vector)
```

### The Lookup Process

```
Input token IDs:  [15496,   995,    13]
                  "Hello"  "world"  "."
                     │        │       │
                     ▼        ▼       ▼
              ┌──────────────────────────────┐
              │     Embedding Matrix E        │
              │     (V × d_model)             │
              │                               │
              │  Row 15496: [0.02, -0.01, ...]│──► x_1 ∈ ℝ^{d_model}
              │  Row   995: [0.03,  0.04, ...]│──► x_2 ∈ ℝ^{d_model}
              │  Row    13: [-0.01, 0.02, ...]│──► x_3 ∈ ℝ^{d_model}
              └──────────────────────────────┘
                                │
                                ▼
              Output: X ∈ ℝ^{3 × d_model}   (one vector per token)
              This X is then passed to the first Transformer block
              (after adding position encoding)
```

## No Computation -- Just Indexing

The embedding operation is **not** a matrix multiplication. It is a table lookup:

```
# Conceptually equivalent:
output = E[token_id]       # O(1) index operation

# NOT this:
output = one_hot @ E       # O(V * d) matrix multiplication -- wasteful
```

This distinction matters for efficiency: the embedding layer has O(batch_size x seq_len) lookup cost, not O(batch_size x seq_len x V) as a full matrix multiply would require.

## Relationship to One-Hot Encoding

Mathematically, the embedding lookup **is equivalent** to multiplying a one-hot vector by the embedding matrix:

```
one_hot(token_id=2) = [0, 0, 1, 0, ..., 0]    ∈ ℝ^V
                                    ↑
                              position 2 is 1

Embedding lookup:
x = one_hot(2)^T · E
  = [0, 0, 1, 0, ..., 0] · E
  = E[2]                              ← selects row 2
```

But in practice, we never construct the one-hot vector. The embedding layer is a **learned dense projection** of the one-hot representation, implemented as direct indexing for efficiency.

```
One-hot (V-dim, sparse) ──── Embedding ────► Dense vector (d_model-dim)

[0, 0, ..., 1, ..., 0]  ──── E[i] ────►  [0.023, -0.019, ..., 0.031]
       V dims                                    d_model dims
     (32,000)                                      (4,096)
```

## Embedding Dimension Choices

The embedding dimension d_model is one of the most important hyperparameters. It determines the width of the entire model (all layers use the same d_model).

| Model | d_model | Vocab Size | Embedding Params | Total Params | Embed % |
|-------|---------|-----------|-----------------|-------------|---------|
| GPT-2 Small | 768 | 50,257 | 39M | 124M | 31% |
| GPT-2 XL | 1,600 | 50,257 | 80M | 1.5B | 5.3% |
| BERT-base | 768 | 30,522 | 23M | 110M | 21% |
| LLaMA-7B | 4,096 | 32,000 | 131M | 6.7B | 2.0% |
| LLaMA-13B | 5,120 | 32,000 | 164M | 13B | 1.3% |
| LLaMA-70B | 8,192 | 32,000 | 262M | 70B | 0.4% |
| LLaMA-3 8B | 4,096 | 128,256 | 525M | 8B | 6.6% |
| GPT-3 175B | 12,288 | 50,257 | 617M | 175B | 0.4% |

Key observations:
- **Small models**: embedding is a large fraction of total parameters (20-30%)
- **Large models**: embedding becomes negligible (<1%)
- **Large vocabularies**: embedding cost grows proportionally (LLaMA-3's 128K vocab is 4x the params of LLaMA-2's 32K vocab at the same d_model)

### Dimension vs. Capacity

```
Small d_model (e.g., 768)            Large d_model (e.g., 8192)
├── Fewer parameters                  ├── More parameters
├── Faster computation                ├── Slower computation
├── Less capacity per token           ├── More capacity per token
├── Tokens may collide in space       ├── Tokens well-separated
└── Good for small models             └── Required for large models
```

A rule of thumb: to represent V tokens well in d dimensions, you need d >> log(V). For V = 32,000, log_2(V) ~ 15, so d = 4,096 provides ample capacity.

## How Embeddings Are Learned

Embeddings are learned end-to-end during pretraining through backpropagation:

```
Forward pass:
  token_id → E[token_id] → Transformer blocks → LM Head → loss

Backward pass:
  loss → gradients flow back through LM Head → Transformer → embedding

  gradient w.r.t. E[token_id] = d(loss) / d(E[token_id])

  Only the rows corresponding to tokens in the current batch receive gradients.
  All other rows have zero gradient for this step.
```

### Sparse Gradient Updates

```
Batch tokens: [42, 1337, 42, 7, 1337, 42]

Gradient updates this step:
  E[7]    ← updated (appeared 1x)
  E[42]   ← updated (appeared 3x, gradients accumulated)
  E[1337] ← updated (appeared 2x, gradients accumulated)
  E[other] ← NOT updated (zero gradient)
```

This means:
- **Frequent tokens** (e.g., "the", "is") get many gradient updates and converge quickly
- **Rare tokens** (e.g., unusual proper nouns) get few updates and may not learn good representations
- With large vocabularies, many tokens are rare, leading to **undertrained embeddings**

## Initialization Strategies

Embeddings must be initialized before training. Common strategies:

### Normal Initialization

```python
# Most common: sample from N(0, sigma^2)
nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
```

The standard deviation 0.02 is a convention from GPT-2. Some models scale it:

```
sigma = 1 / sqrt(d_model)
```

For d_model = 4096: sigma = 1/64 = 0.0156

### Xavier (Glorot) Initialization

```python
# Maintains variance across layers
nn.init.xavier_uniform_(embedding.weight)
# or
nn.init.xavier_normal_(embedding.weight)
```

Xavier initialization sets:

```
sigma = sqrt(2 / (fan_in + fan_out)) = sqrt(2 / (V + d_model))
```

For V = 32,000 and d_model = 4,096: sigma = sqrt(2/36096) = 0.0074

### Initialization Comparison

| Strategy | Formula | sigma (V=32K, d=4096) | Used By |
|----------|---------|----------------------|---------|
| Normal(0, 0.02) | Fixed | 0.02 | GPT-2, GPT-3 |
| Normal(0, 1/sqrt(d)) | 1/sqrt(d_model) | 0.0156 | Many LLMs |
| Xavier Normal | sqrt(2/(V+d)) | 0.0074 | BERT |
| Truncated Normal | N(0, 0.02) clipped to [-0.04, 0.04] | 0.02 | Some variants |

In practice, the exact initialization matters less for large models trained for many steps, as the embeddings move far from their initial values.

## PyTorch Implementation

```python
import torch
import torch.nn as nn

# --- Basic embedding layer ---
vocab_size = 32000
d_model = 4096

embedding = nn.Embedding(vocab_size, d_model)
print(f"Embedding shape: {embedding.weight.shape}")
# Embedding shape: torch.Size([32000, 4096])
print(f"Parameters: {embedding.weight.numel():,}")
# Parameters: 131,072,000

# --- Forward pass: lookup ---
token_ids = torch.tensor([15496, 995, 13])  # "Hello", "world", "."
vectors = embedding(token_ids)
print(f"Input shape:  {token_ids.shape}")    # torch.Size([3])
print(f"Output shape: {vectors.shape}")      # torch.Size([3, 4096])

# --- Batched forward pass ---
batch = torch.tensor([
    [15496, 995, 13,  0],     # "Hello world ." <pad>
    [464, 3797, 3332, 13],    # "The cat sat ."
])
vectors = embedding(batch)
print(f"Batch shape:  {batch.shape}")        # torch.Size([2, 4])
print(f"Output shape: {vectors.shape}")      # torch.Size([2, 4, 4096])

# --- Custom initialization ---
embedding = nn.Embedding(vocab_size, d_model)
nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

# --- With padding token ---
embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
# E[0] is initialized to zeros and NOT updated during training

# --- Embedding in a model ---
class SimpleEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = d_model ** 0.5  # optional: scale by sqrt(d_model)

    def forward(self, token_ids):
        # token_ids: (batch_size, seq_len)
        x = self.embedding(token_ids)   # (batch_size, seq_len, d_model)
        x = x * self.scale              # scale (used in original Transformer)
        return x
```

### Scaling by sqrt(d_model)

The original Transformer paper multiplies embeddings by sqrt(d_model):

```
x = E[token_id] * sqrt(d_model)
```

Why? With standard initialization, embedding vectors have norm ~ sqrt(d_model) * sigma, which can be much smaller than positional encoding values. Scaling ensures the two are on the same scale before addition. Most modern LLMs (GPT, LLaMA) do **not** use this scaling, relying instead on learned magnitudes and normalization layers.

## Embedding Geometry

Trained embeddings form a meaningful geometric space:

```
                 king ●
                     /|
                    / |
    (gender axis) /  |  (royalty axis)
                 /   |
          queen ●    |
                     |
                man ●
                   /
                  /
           woman ●

    king - man + woman ≈ queen
```

This structure is an emergent property of training, not explicitly designed. Semantically similar tokens end up close together in embedding space.

## Summary

| Aspect | Detail |
|--------|--------|
| What | Lookup table mapping token IDs to dense vectors |
| Shape | E ∈ ℝ^{V x d_model} |
| Operation | E[token_id] -- pure indexing, O(1) per token |
| Parameters | V x d_model (e.g., 131M for V=32K, d=4096) |
| Learning | End-to-end via backpropagation, sparse gradient updates |
| Initialization | Usually Normal(0, 0.02) or Normal(0, 1/sqrt(d)) |
| Equivalence | Same as one_hot(id) @ E, but implemented as index |

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Word2Vec (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) | Dense word embeddings via prediction |
| [GloVe (Pennington et al., 2014)](https://nlp.stanford.edu/pubs/glove.pdf) | Embeddings from co-occurrence statistics |
| [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) | Embedding layer in the Transformer |

## Related

- [Tokenization](01_Tokenization.md) - How text is converted to token IDs before embedding
- [Output Head](03_Output_Head.md) - The reverse operation: vectors back to vocabulary logits
- [Position Encoding](../position_encoding/00_Position_Encoding.md) - Added to embeddings before the Transformer
