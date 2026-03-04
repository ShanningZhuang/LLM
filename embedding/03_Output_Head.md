# Output Head (LM Head)

> Parent: [Embedding](00_Embedding.md)

## Overview

The output head (also called the **LM head** or **unembedding layer**) is the final component of an LLM. It takes the hidden state from the last Transformer block and projects it back to a distribution over the vocabulary. This is conceptually the reverse of the embedding layer: while the embedding maps token IDs to vectors, the LM head maps vectors back to token logits. A key design decision is **weight tying** -- sharing the same matrix for both operations.

## LM Head: Architecture

```
Last Transformer block output: h ∈ ℝ^{d_model}
                                    │
                               ┌────▼────┐
                               │ RMSNorm │   Final layer normalization
                               └────┬────┘
                                    │
                               ┌────▼──────────────────┐
                               │ LM Head (Linear)       │
                               │ logits = h · W^T       │   W ∈ ℝ^{V × d_model}
                               │ No bias term           │
                               └────┬──────────────────┘
                                    │
                                    ▼
                          logits ∈ ℝ^V   (one score per vocabulary token)
                                    │
                               ┌────▼──────────────────┐
                               │ Softmax                │
                               │ P(token_i) = exp(z_i)  │
                               │              / sum(exp) │
                               └────┬──────────────────┘
                                    │
                                    ▼
                         P ∈ ℝ^V   (probability distribution over vocabulary)
                                    │
                               ┌────▼──────────────────┐
                               │ Sample / Argmax        │   Decoding strategy
                               └───────────────────────┘
                                    │
                                    ▼
                              Next token ID
```

## Linear Projection

The LM head is a single linear layer (no activation function):

```
logits = h · W^T + b

Where:
  h ∈ ℝ^{d_model}         hidden state of the last layer for one position
  W ∈ ℝ^{V × d_model}     weight matrix (same shape as embedding matrix)
  b ∈ ℝ^V                 bias (usually OMITTED in modern LLMs)
  logits ∈ ℝ^V            unnormalized scores over vocabulary
```

### No Bias in Modern LLMs

Most modern LLMs omit the bias term:

```
logits = h · W^T       (no bias)
```

Reasons:
1. **Fewer parameters**: bias adds V parameters (32K-128K), negligible but unnecessary
2. **Weight tying compatibility**: the embedding matrix has no bias, so tying is cleaner without one
3. **No empirical benefit**: studies show bias does not improve language modeling performance

## Softmax: Logits to Probabilities

The softmax function converts raw logits to a valid probability distribution:

```
P(token_i) = exp(z_i) / sum_{j=1}^{V} exp(z_j)

Where:
  z_i = logit for token i
  V   = vocabulary size
```

Properties:
- All probabilities are positive: P(token_i) > 0
- They sum to 1: sum_{i=1}^{V} P(token_i) = 1
- Preserves ranking: if z_i > z_j, then P(token_i) > P(token_j)
- Amplifies differences: large logits get disproportionately more probability

### Numerical Stability

In practice, softmax is computed with a shift for numerical stability:

```
z_max = max(z_1, ..., z_V)
P(token_i) = exp(z_i - z_max) / sum_{j=1}^{V} exp(z_j - z_max)
```

This prevents overflow from exp(large_number) without changing the result.

## Temperature Scaling

Temperature controls the "sharpness" of the output distribution:

```
P(token_i) = exp(z_i / T) / sum_{j=1}^{V} exp(z_j / T)

Where T = temperature (default: 1.0)
```

### Effect of Temperature

```
Logits: [2.0, 1.0, 0.5, 0.1]

T = 0.1 (very sharp):     [0.9991, 0.0009, 0.0000, 0.0000]  ← nearly argmax
T = 0.5 (focused):        [0.8360, 0.1224, 0.0303, 0.0113]  ← strong preference
T = 1.0 (default):        [0.4672, 0.1721, 0.1043, 0.0700]  ← moderate spread
T = 2.0 (creative):       [0.3165, 0.2095, 0.1728, 0.1500]  ← flatter
T = inf (uniform):        [0.2500, 0.2500, 0.2500, 0.2500]  ← uniform random

Temperature scale:
  T → 0   : deterministic (argmax)
  T = 1   : standard softmax
  T → inf : uniform distribution
```

Temperature is applied **at inference time only**, not during training.

## Weight Tying

Weight tying (also called **weight sharing**) reuses the embedding matrix E as the LM head weight matrix W:

```
W_out = E^T

So:  logits = h · E        (using the SAME matrix as the embedding layer)
```

### Without vs. With Weight Tying

```
WITHOUT weight tying:                    WITH weight tying:
┌─────────────────┐                      ┌─────────────────┐
│ Embedding: E     │  V × d params       │ Embedding: E     │  V × d params
│ (input layer)    │                      │ (input layer)    │
└─────────────────┘                      └────────┬────────┘
                                                  │ shared
┌─────────────────┐                      ┌────────▼────────┐
│ LM Head: W       │  V × d params       │ LM Head: E^T     │  0 extra params
│ (output layer)   │                      │ (output layer)   │
└─────────────────┘                      └─────────────────┘

Total: 2 × V × d                        Total: 1 × V × d
```

### Why Weight Tying Works

The geometric argument:

```
Embedding:  token_id → E[token_id] = vector in d_model space
LM Head:    h → h · E^T = similarity of h with every token's embedding

With weight tying, the LM head computes:
  logit(token_i) = h · E[i]^T = dot_product(h, E[i])

This means:
  - Tokens whose embeddings are SIMILAR to h get HIGH logits
  - Tokens whose embeddings are DISSIMILAR to h get LOW logits
```

This is sensible because:

1. **Similar tokens should have similar embeddings** (semantic clustering)
2. **Similar tokens should have similar output probabilities** (if "happy" is likely, "glad" should also be somewhat likely)
3. **Weight tying enforces both constraints with a single matrix**, creating a consistent semantic space used for both input representation and output prediction

```
Embedding space with weight tying:

      Token "happy" ●─── embedding vector
                    /
                   / cos_sim ≈ 0.85
                  /
    Token "glad" ●─── nearby embedding

    If hidden state h points toward "happy"'s embedding,
    both "happy" AND "glad" get high logits.

    This is exactly the behavior we want!
```

### Parameter Savings

```
Model: V = 32,000, d_model = 4,096

Without tying:
  Embedding:  32,000 × 4,096 = 131,072,000 params
  LM Head:    32,000 × 4,096 = 131,072,000 params
  Total:      262,144,000 params

With tying:
  Shared:     32,000 × 4,096 = 131,072,000 params
  Total:      131,072,000 params

  Savings:    131M parameters (50% of embedding + LM head)
```

For larger vocabularies the savings are even more significant:

| Model Config | V | d_model | Without Tying | With Tying | Savings |
|-------------|------|---------|--------------|------------|---------|
| LLaMA-7B size | 32K | 4,096 | 262M | 131M | 131M (2.0% of 6.7B) |
| LLaMA-3 size | 128K | 4,096 | 1,050M | 525M | 525M (6.6% of 8B) |
| GPT-3 size | 50K | 12,288 | 1,234M | 617M | 617M (0.4% of 175B) |
| Gemma size | 256K | 3,072 | 1,573M | 786M | 786M (11% of 7B) |

## Which Models Use Weight Tying?

| Model | Weight Tying | Vocab Size | Notes |
|-------|-------------|-----------|-------|
| GPT-2 | Yes | 50,257 | Shared embedding and LM head |
| GPT-3 | No | 50,257 | Separate matrices |
| BERT | Yes | 30,522 | Shared with MLM head |
| T5 | Yes | 32,100 | Shared across encoder and decoder |
| LLaMA / LLaMA-2 | No | 32,000 | Separate matrices |
| LLaMA-3 | Yes | 128,256 | Tying saves 525M params with large vocab |
| Mistral-7B | No | 32,000 | Separate matrices |
| Gemma | Yes | 256,000 | Essential with 256K vocab |
| Qwen-2 | Yes | 151,936 | Large vocab benefits from tying |
| DeepSeek-V2 | No | 100,015 | Separate matrices |

Trend: models with **large vocabularies** (>100K) tend to use weight tying because the parameter savings become substantial. Models with smaller vocabularies often keep separate matrices for added expressiveness.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LMHead(nn.Module):
    """LM head without weight tying."""
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)           # final normalization
        self.linear = nn.Linear(d_model, vocab_size, bias=False)  # no bias

    def forward(self, h):
        # h: (batch_size, seq_len, d_model)
        h = self.norm(h)
        logits = self.linear(h)  # (batch_size, seq_len, vocab_size)
        return logits


class LMHeadWithTying(nn.Module):
    """LM head with weight tying -- shares embedding matrix."""
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        # No separate linear layer -- we reuse the embedding weight

    def forward(self, h):
        # h: (batch_size, seq_len, d_model)
        h = self.norm(h)
        # Weight tying: use embedding matrix as LM head
        logits = F.linear(h, self.embedding.weight)  # h @ E^T
        # Equivalent to: logits = h @ self.embedding.weight.T
        return logits

    def embed(self, token_ids):
        return self.embedding(token_ids)


# --- Full model skeleton showing weight tying ---
class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, tie_weights=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model) for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        if tie_weights:
            self.lm_head.weight = self.embedding.weight  # SHARED parameter

    def forward(self, token_ids):
        # token_ids: (batch_size, seq_len)
        x = self.embedding(token_ids)          # (B, S, d_model)
        for layer in self.layers:
            x = layer(x)                       # (B, S, d_model)
        x = self.norm(x)                       # (B, S, d_model)
        logits = self.lm_head(x)               # (B, S, vocab_size)
        return logits


# --- Temperature scaling at inference ---
def sample_with_temperature(logits, temperature=1.0):
    """Apply temperature and sample from the distribution."""
    # logits: (vocab_size,)
    if temperature == 0:
        return logits.argmax()                 # greedy decoding

    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# --- Usage ---
model = SimpleLLM(vocab_size=32000, d_model=4096, n_layers=32, tie_weights=True)

# Verify weight tying
print(model.embedding.weight is model.lm_head.weight)  # True
print(f"Embedding params: {model.embedding.weight.numel():,}")
# Embedding params: 131,072,000
# LM head reuses the same 131M params (not counted twice)
```

## Training: Cross-Entropy Loss

During training, the LM head logits are used to compute cross-entropy loss against the target token:

```
loss = -log P(target_token)
     = -log( exp(z_target) / sum_j exp(z_j) )
     = -z_target + log( sum_j exp(z_j) )

This is computed efficiently by PyTorch:
  loss = F.cross_entropy(logits, target_ids)
```

```python
# Training step
logits = model(input_ids)           # (B, S, V)
# Shift: predict token t+1 from position t
shift_logits = logits[:, :-1, :]    # (B, S-1, V)
shift_labels = input_ids[:, 1:]     # (B, S-1)

loss = F.cross_entropy(
    shift_logits.reshape(-1, vocab_size),
    shift_labels.reshape(-1)
)
```

## LM Head Initialization

When weight tying is **not** used, the LM head needs its own initialization:

```python
# Common initialization for LM head
nn.init.normal_(lm_head.weight, mean=0.0, std=0.02)

# Some models scale initialization by depth
# (e.g., GPT-2 scales residual layer outputs by 1/sqrt(N))
nn.init.normal_(lm_head.weight, mean=0.0, std=0.02 / (2 * n_layers) ** 0.5)
```

When weight tying **is** used, the LM head inherits the embedding initialization and they are trained jointly.

## Summary

```
Input                                          Output
  │                                              │
  ▼                                              ▼
token_id ──► E[token_id] ──►  ...  ──► h ──► h·E^T ──► softmax ──► P(next)
             ▲ embedding          Transformer    ▲ LM head
             │                    blocks         │
             └──────────── weight tying ─────────┘
                         (same matrix E)
```

| Component | Operation | Parameters | Shape |
|-----------|-----------|-----------|-------|
| LM Head | logits = h @ W^T | V x d_model | ℝ^{d_model} -> ℝ^V |
| Softmax | P = softmax(logits / T) | 0 | ℝ^V -> ℝ^V |
| Weight Tying | W = E | 0 (shared) | -- |
| Temperature | Scale logits before softmax | 0 (hyperparameter) | scalar |

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Using the Output Embedding to Improve LMs (Press & Wolf, 2017)](https://arxiv.org/abs/1608.05859) | Weight tying between embedding and LM head |
| [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) | Transformer with shared embedding weights |
| [LLaMA (2023)](https://arxiv.org/abs/2302.13971) | Separate (untied) embedding and LM head |
| [Gemma (2024)](https://arxiv.org/abs/2403.08295) | Weight tying with 256K vocabulary |

## Related

- [Token Embedding](02_Token_Embedding.md) - The input embedding layer that the LM head may share weights with
- [Tokenization](01_Tokenization.md) - How text is split into tokens before embedding
- [Sampling Strategies](../generation/02_Sampling.md) - Temperature, top-k, top-p applied after the LM head
