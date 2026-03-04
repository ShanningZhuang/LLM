# Decoder-Only Transformer Architecture

> Parent: [Architecture](00_Architecture.md)

## Overview

The decoder-only Transformer is the dominant architecture behind modern LLMs (GPT, LLaMA, Mistral, DeepSeek). Unlike encoder-decoder models (T5) or encoder-only models (BERT), the decoder-only design uses a single stack of causally-masked Transformer blocks to model the probability of the next token given all previous tokens.

## Learning Objectives

- [x] Full forward pass: input text to output token
- [x] Why decoder-only won over other architectures
- [x] Parameter counting formulas for each component
- [x] PyTorch pseudocode for a simplified decoder-only model

---

## End-to-End Forward Pass

### Step-by-Step Walkthrough

```
Input: "The cat sat on"

Step 1: Tokenization
  "The cat sat on" → [464, 3797, 3332, 319]

Step 2: Token Embedding + Position Encoding
  [464, 3797, 3332, 319] → x ∈ R^(4 x d_model)
  Each token ID maps to a d_model-dimensional vector.
  Position information is added (learned, RoPE, or sinusoidal).

Step 3: N Transformer Blocks (the core)
  For each block i = 1..N:
    x = x + Attention(RMSNorm(x))      # pre-norm + residual
    x = x + FFN(RMSNorm(x))            # pre-norm + residual

Step 4: Final Normalization
  x = RMSNorm(x)

Step 5: LM Head (unembedding)
  logits = x @ W_embed^T   ∈ R^(4 x vocab_size)
  (Often weight-tied with the token embedding matrix)

Step 6: Sampling
  probs = softmax(logits[-1] / temperature)
  next_token = sample(probs) → "the"
```

### Full Architecture Diagram

```
Input: "The cat sat on"
         │
         ▼
┌─────────────────────────────────────────────────┐
│                  Tokenizer                       │
│  "The" "cat" "sat" "on" → [464, 3797, 3332, 319]│
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│          Token Embedding Table                   │
│     W_embed ∈ R^(vocab_size x d_model)          │
│     [464]  → [0.12, -0.34, ..., 0.56]          │
│     [3797] → [0.78, 0.11, ..., -0.23]          │
│     ...                                          │
│          + Position Encoding (RoPE / learned)    │
│                                                  │
│     x ∈ R^(seq_len x d_model)                   │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│          Transformer Block 1                     │
│  ┌─────────────────────────────────────────┐    │
│  │ RMSNorm → Multi-Head Causal Attention   │    │
│  │          Q, K, V projections            │    │
│  │          Causal mask (lower triangular) │    │
│  │          + Residual connection          │    │
│  ├─────────────────────────────────────────┤    │
│  │ RMSNorm → Feed-Forward Network (SwiGLU)│    │
│  │          + Residual connection          │    │
│  └─────────────────────────────────────────┘    │
├─────────────────────────────────────────────────┤
│          Transformer Block 2                     │
│          ...                                     │
├─────────────────────────────────────────────────┤
│          ...                                     │
├─────────────────────────────────────────────────┤
│          Transformer Block N (e.g., 32)          │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│              Final RMSNorm                       │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│          LM Head (Linear, no bias)               │
│     logits = x @ W_embed^T                       │
│     logits ∈ R^(seq_len x vocab_size)            │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│     Softmax + Sampling (last position only)      │
│     next_token = sample(softmax(logits[-1] / T)) │
│                                                  │
│     → "the"                                      │
└─────────────────────────────────────────────────┘
```

---

## Why Decoder-Only Won

### Architecture Comparison

```
Encoder-Only (BERT):
  Input → [Encoder x N] → Hidden states
  Bidirectional attention (sees all tokens)
  Good for: classification, NER, sentence embeddings
  Bad for: generation (not autoregressive)

Encoder-Decoder (T5):
  Input → [Encoder x N] → Memory
  Memory + Target → [Decoder x N] → Output
  Cross-attention from decoder to encoder
  Good for: translation, summarization
  Bad for: scaling efficiency (2 stacks = more complexity)

Decoder-Only (GPT):
  Input → [Decoder x N] → Next token
  Causal attention (sees only past tokens)
  Good for: everything (generation, classification, reasoning)
  Why it won: simplest architecture that scales
```

### Why Decoder-Only Dominates

| Factor | Encoder-Only | Encoder-Decoder | Decoder-Only |
|--------|-------------|-----------------|--------------|
| Generation | Cannot generate natively | Yes | Yes |
| Scaling simplicity | One stack | Two stacks + cross-attn | One stack |
| Pre-training objective | MLM (15% tokens) | Span corruption | Next-token (100% tokens) |
| Training signal density | Low (15%) | Medium | High (100%) |
| In-context learning | No | Limited | Excellent |
| Task flexibility | Classification only | Seq2seq tasks | Universal |
| KV cache efficiency | N/A | Two separate caches | Single cache |
| Representative models | BERT, RoBERTa | T5, BART | GPT, LLaMA, Mistral |

Key reasons decoder-only won:

1. **Next-token prediction uses 100% of tokens** -- BERT's MLM only trains on the 15% masked tokens per pass, wasting 85% of compute
2. **Emergent in-context learning** -- decoder-only models learn to follow instructions and few-shot examples purely from scale
3. **Simpler to scale** -- one parameter set, one forward pass, one KV cache
4. **Unified interface** -- every task becomes "complete this text," no task-specific heads needed

---

## Parameter Counting

### Formulas

For a decoder-only Transformer with:
- `V` = vocab_size, `d` = d_model, `N` = n_layers
- `h` = n_heads, `d_h` = head_dim = d / h
- `d_ff` = FFN intermediate size
- `n_kv` = number of KV heads (for GQA)

```
Component                    Parameters
───────────────────────────────────────────────────────────
Token embedding              V * d
Position embedding           seq_len * d        (if learned; 0 for RoPE)

Per Transformer Block:
  Attention:
    Q projection             d * d              = d * (h * d_h)
    K projection             d * (n_kv * d_h)   (fewer if GQA)
    V projection             d * (n_kv * d_h)   (fewer if GQA)
    Output projection        d * d
  Attention total (MHA)      4 * d^2            (when n_kv = h)
  Attention total (GQA)      d^2 * (1 + 1 + 2*n_kv/h)

  FFN (standard):
    Up projection            d * d_ff
    Down projection          d_ff * d
  FFN total (standard)       2 * d * d_ff       (d_ff = 4d typically)

  FFN (SwiGLU):
    Gate projection          d * d_ff
    Up projection            d * d_ff
    Down projection          d_ff * d
  FFN total (SwiGLU)         3 * d * d_ff       (d_ff = 8d/3 typically)

  RMSNorm (x2 per block)    2 * d              (just scale vectors)

Per block total (MHA+SwiGLU) = 4*d^2 + 3*d*d_ff + 2*d

All N blocks                 N * (4*d^2 + 3*d*d_ff + 2*d)

Final RMSNorm                d
LM Head                      V * d  (often tied with embedding → 0 extra)
───────────────────────────────────────────────────────────
Total (weight-tied)          V*d + N*(4*d^2 + 3*d*d_ff + 2*d) + d
```

### Example: LLaMA-2 7B Parameter Count

```
LLaMA-2 7B Configuration:
  V = 32,000     d = 4,096     N = 32
  h = 32         d_h = 128     n_kv = 32 (full MHA)
  d_ff = 11,008  (SwiGLU: 8/3 * 4096 ≈ 10922, rounded to 11008)

Token embedding:
  32,000 * 4,096 = 131,072,000         ~131M

Per Transformer Block:
  Attention (Q, K, V, O):
    4 * 4,096 * 4,096 = 67,108,864     ~67M
  FFN (SwiGLU: gate + up + down):
    3 * 4,096 * 11,008 = 135,266,304   ~135M
  RMSNorm (x2):
    2 * 4,096 = 8,192                  ~8K
  Block total:                          ~202M

All 32 blocks:
  32 * 202,383,360 = 6,476,267,520     ~6,476M

Final RMSNorm:    4,096
LM Head:          tied with embedding → 0

Total: 131M + 6,476M + ~0 ≈ 6,607M ≈ 6.6B

(Published as "7B" -- close match, difference from
 bias terms and rounding conventions)
```

### Parameter Distribution

```
LLaMA-2 7B Parameter Breakdown:

Token Embedding:  ██ 2.0%                    131M
Attention (all):  ████████████████ 32.5%     2,147M
FFN (all):        ████████████████████████   4,328M
                  ████████ 65.5%
Norms + other:    ▏ <0.1%

Key insight: ~2/3 of parameters are in the FFN layers!
```

---

## Architecture Comparison Table

| Property | BERT (Encoder-Only) | T5 (Encoder-Decoder) | GPT / LLaMA (Decoder-Only) |
|----------|--------------------|-----------------------|---------------------------|
| Attention type | Bidirectional | Bidirectional (enc) + Causal (dec) | Causal only |
| Pre-training task | Masked LM (MLM) | Span corruption | Next-token prediction |
| Training signal | 15% of tokens | ~15% of tokens | 100% of tokens |
| Generation | Not native | Native | Native |
| Cross-attention | No | Yes (dec→enc) | No |
| In-context learning | No | Limited | Strong |
| Typical use | Classification, NER | Translation, summarization | Everything |
| Scaling record | 340M (BERT-large) | 11B (T5-XXL) | 1.8T (GPT-4, est.) |
| KV cache | N/A | Two caches | One cache |
| Position encoding | Learned absolute | Relative bias | RoPE / learned |

---

## PyTorch Pseudocode

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape

        # Project to Q, K, V
        q = self.W_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        # In practice, use F.scaled_dot_product_attention with is_causal=True
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        # Combine heads
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class SwiGLU_FFN(nn.Module):
    """SwiGLU: gate * swish(gate_proj(x)) * up_proj(x)"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn      = CausalSelfAttention(d_model, n_heads)
        self.ffn_norm  = RMSNorm(d_model)
        self.ffn       = SwiGLU_FFN(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))    # pre-norm + residual
        x = x + self.ffn(self.ffn_norm(x))      # pre-norm + residual
        return x


class DecoderOnlyLM(nn.Module):
    """Simplified decoder-only language model (LLaMA-style)."""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        # RoPE would be applied inside attention; omitted for clarity
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        # Weight-tied LM head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight  # weight tying

    def forward(self, token_ids):
        # token_ids: (batch, seq_len)
        x = self.tok_embed(token_ids)              # (B, T, d_model)

        for block in self.blocks:
            x = block(x)                            # (B, T, d_model)

        x = self.final_norm(x)                      # (B, T, d_model)
        logits = self.lm_head(x)                    # (B, T, vocab_size)
        return logits


# Instantiate a LLaMA-2 7B-like model
model = DecoderOnlyLM(
    vocab_size  = 32000,
    d_model     = 4096,
    n_layers    = 32,
    n_heads     = 32,
    d_ff        = 11008,
    max_seq_len = 4096,
)

# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total:,}")  # ~6.6B
```

---

## Causal Masking: The Key Constraint

The causal (autoregressive) mask is what makes decoder-only models generative. Each token can only attend to itself and previous tokens.

```
Attention mask for sequence of length 4:

       Token 1  Token 2  Token 3  Token 4
Token 1  [1       0        0        0]
Token 2  [1       1        0        0]
Token 3  [1       1        1        0]
Token 4  [1       1        1        1]

1 = can attend, 0 = masked (-inf before softmax)

This ensures:
- P(token_1) depends on nothing (just learned bias)
- P(token_2) depends on token_1
- P(token_3) depends on token_1, token_2
- P(token_4) depends on token_1, token_2, token_3

Joint probability factorization:
P(x_1, x_2, ..., x_T) = P(x_1) * P(x_2|x_1) * ... * P(x_T|x_1,...,x_{T-1})
```

---

## Pre-Norm vs Post-Norm

Modern decoder-only models all use **pre-norm** (normalize before sublayer):

```
Post-Norm (original Transformer, 2017):
  x = LayerNorm(x + Attention(x))
  x = LayerNorm(x + FFN(x))
  Problem: gradient flow is difficult at large depth

Pre-Norm (GPT-2 onwards, all modern LLMs):
  x = x + Attention(RMSNorm(x))
  x = x + FFN(RMSNorm(x))
  Benefit: residual stream is clean, gradients flow directly
```

```
Pre-Norm residual stream:

Input ──────────────────────────────────────────── Output
  │            │              │             │
  └──► Norm    └──► Norm      └──► Norm     └──► Norm
       │            │              │              │
       Attn         FFN            Attn           FFN
       │            │              │              │
  ◄────┘       ◄────┘         ◄────┘         ◄────┘
  (+add)       (+add)         (+add)         (+add)

The residual stream carries information directly from
input to output -- sublayers read from and write to it.
```

---

## Key Papers

| Paper | Year | Key Contribution |
|-------|------|------------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Original Transformer (encoder-decoder) |
| [GPT (Radford et al.)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | 2018 | First decoder-only Transformer for LM |
| [BERT (Devlin et al.)](https://arxiv.org/abs/1810.04805) | 2018 | Encoder-only with MLM |
| [GPT-2 (Radford et al.)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 2019 | Pre-norm, scaled decoder-only |
| [LLaMA (Touvron et al.)](https://arxiv.org/abs/2302.13971) | 2023 | RMSNorm + SwiGLU + RoPE recipe |

---

## Related

- [Embedding](../embedding/00_Embedding.md) -- token and position embeddings
- [Transformer Block](../transformer_block/00_Transformer_Block.md) -- attention + FFN internals
- [Attention](../attention/00_Attention.md) -- MHA, GQA, MQA, causal masking
- [FFN](../ffn/00_FFN.md) -- SwiGLU, standard FFN, MoE
- [Position Encoding](../position_encoding/00_Position_Encoding.md) -- RoPE, ALiBi, learned
