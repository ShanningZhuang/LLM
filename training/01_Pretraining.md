# Pretraining

> Parent: [Training](00_Training.md)

## Overview

Pretraining is the stage where a language model learns the statistical structure of language by predicting the next token across trillions of tokens. The objective is conceptually simple -- causal language modeling (CLM) -- but the engineering details around data preparation, mixed precision, and training stability determine whether a model becomes capable or collapses.

## Next-Token Prediction (Causal Language Modeling)

Given a sequence of tokens `[t_1, t_2, ..., t_n]`, the model learns to predict the next token at every position:

```
Input:   [t_1,   t_2,   t_3,   ..., t_n  ]
Target:  [t_2,   t_3,   t_4,   ..., t_{n+1}]

The model outputs logits at each position.
Loss = CrossEntropy(logits[:-1], labels[1:])
```

### The Shift Mechanism

```
Sequence: "The cat sat on the mat"

Position:    0      1      2      3      4      5
Token:      The    cat    sat    on     the    mat

Input:      The    cat    sat    on     the    mat
             │      │      │      │      │      │
             ▼      ▼      ▼      ▼      ▼      ▼
          ┌─────────────────────────────────────────┐
          │         Transformer (causal mask)         │
          └─────────────────────────────────────────┘
             │      │      │      │      │      │
             ▼      ▼      ▼      ▼      ▼      ▼
Predict:   cat    sat    on     the    mat    <eos>
            ↕      ↕      ↕      ↕      ↕      ↕
Label:     cat    sat    on     the    mat    <eos>

Loss = average CrossEntropy at each position

Key: the model at position i can only see tokens [0..i] (causal mask)
     but must predict token [i+1]
```

### Teacher Forcing

During training, the model always receives the **ground truth** tokens as input, regardless of what it would have predicted. This is called **teacher forcing**:

```
Teacher Forcing (training):
  Input at position i = ground truth token t_i     (always correct)
  Predict: t_{i+1}

Free Running (inference):
  Input at position i = model's own prediction      (may be wrong)
  Predict: t_{i+1}

Teacher forcing enables parallel training of all positions simultaneously
(because inputs don't depend on model outputs).
```

## Why Next-Token Prediction Works

Next-token prediction is a surprisingly powerful objective because it implicitly requires understanding at every level:

```
To predict the next token well, the model must learn:

Level 1: Syntax
  "The cat sat on the ___"  → requires grammar

Level 2: Semantics
  "The capital of France is ___"  → requires world knowledge

Level 3: Reasoning
  "If all roses are flowers and all flowers are plants, then all roses are ___"
  → requires logical inference

Level 4: Style/Pragmatics
  "Dear Sir, I am writing to ___"  → requires discourse understanding
```

The key insight (articulated by Ilya Sutskever): **prediction is compression**. A model that can predict any text well must have built an internal representation of the world that generated that text. Language modeling is an implicit form of unsupervised multitask learning.

## Training Data

### Data Sources

| Source | Scale | Content | Quality |
|--------|-------|---------|---------|
| Common Crawl / C4 | Trillions of tokens | Web pages | Medium (needs heavy filtering) |
| Wikipedia | ~4B tokens | Encyclopedic | High |
| Books (Books3, Gutenberg) | ~30B tokens | Long-form prose | High |
| Code (GitHub, The Stack) | ~500B tokens | Programming languages | High for code tasks |
| ArXiv | ~30B tokens | Scientific papers | High for technical |
| StackExchange | ~15B tokens | Q&A | High for reasoning |

### Data Quality > Quantity

```
Chinchilla (Hoffmann et al., 2022):
  Optimal tokens ≈ 20 × parameters
  → 7B model should train on ~140B tokens
  → Previous models (e.g., GPT-3 175B on 300B tokens) were undertrained

LLaMA (Touvron et al., 2023):
  Trained 7B model on 1T tokens (7× Chinchilla-optimal)
  → Smaller model + more data can match larger model + less data
  → LLaMA-7B matched GPT-3 175B on many benchmarks

Lesson: high-quality data matters more than raw model size.
```

### Data Processing Pipeline

```
Raw Web Crawl
     │
     ▼
┌──────────────────┐
│ Language filter   │  Keep English (or target language)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Quality filter    │  Heuristics: text length, symbol ratio,
│                   │  perplexity filter (train small LM first)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Deduplication     │  Exact dedup (hashing)
│                   │  Near-dedup (MinHash / SimHash)
│                   │  ~30-50% of Common Crawl is duplicated
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Toxicity removal  │  Classifier-based filtering
│                   │  Block/allow list of domains
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ PII removal       │  Remove emails, phone numbers, SSNs
└────────┬─────────┘
         │
         ▼
Clean Dataset (~1-5T tokens)
```

## Training Pipeline: From Text to Batches

### Step 1: Tokenize Entire Corpus

```
Document 1: "The cat sat on the mat."  → [464, 3797, 3332, 319, 278, 5765, 29889]
Document 2: "Dogs are loyal."          → [29928, 12099, 526, 12843, 284, 29889]
...
```

### Step 2: Concatenate and Chunk

```
All documents concatenated with <eos> separators:

[464, 3797, 3332, 319, 278, 5765, 29889, <eos>, 29928, 12099, 526, ...]

Chunk into fixed-length sequences (e.g., 2048 or 4096 tokens):

Chunk 1: [464, 3797, 3332, ..., 12099]     (exactly 2048 tokens)
Chunk 2: [526, 12843, 284, ..., 7451]      (exactly 2048 tokens)
...

Note: document boundaries may fall in the middle of a chunk.
Some implementations add special tokens at boundaries or use
attention masks to prevent cross-document attention.
```

### Step 3: Shuffle and Batch

```
Chunks are shuffled and grouped into batches.

Batch size is measured in tokens:
  batch_size_tokens = micro_batch_size × seq_len × grad_accum_steps × n_gpus

Typical: 1M-4M tokens per optimizer step
  e.g., 4 × 4096 × 32 × 8 = 4,194,304 tokens per step
```

## Mixed Precision Training

Modern LLMs use mixed precision to save memory and increase throughput:

```
┌──────────────────────────────────────────────────────┐
│               Mixed Precision Training                │
│                                                       │
│  Master weights:      FP32  (32-bit, full precision) │
│  Forward activations: BF16  (16-bit, saves memory)   │
│  Backward gradients:  BF16  (16-bit)                 │
│  Gradient accumulator: FP32 (32-bit, avoid drift)    │
│  Optimizer states:    FP32  (32-bit, Adam m and v)   │
│                                                       │
│  BF16 vs FP16:                                        │
│    BF16: 1 sign + 8 exponent + 7 mantissa            │
│    FP16: 1 sign + 5 exponent + 10 mantissa           │
│                                                       │
│    BF16 has same range as FP32 (8-bit exponent)      │
│    but less precision (7 mantissa vs 23)              │
│    → No loss scaling needed (unlike FP16)             │
│    → Preferred for LLM training                       │
└──────────────────────────────────────────────────────┘
```

| Format | Bits | Range | Precision | Use in LLM Training |
|--------|------|-------|-----------|---------------------|
| FP32 | 32 | +/- 3.4e38 | ~7 digits | Optimizer states, accumulation |
| BF16 | 16 | +/- 3.4e38 | ~3 digits | Activations, gradients |
| FP16 | 16 | +/- 6.5e4 | ~4 digits | Less common (needs loss scaling) |

## PyTorch Training Loop

```python
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

# --- Configuration ---
model = DecoderOnlyLM(...)      # your model
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps, eta_min=3e-5
)
grad_accum_steps = 32
max_grad_norm = 1.0

model.train()

for step, batch in enumerate(dataloader):
    input_ids = batch["input_ids"].cuda()   # (B, seq_len)

    # --- Forward pass in BF16 ---
    with autocast(dtype=torch.bfloat16):
        logits = model(input_ids)           # (B, seq_len, vocab_size)

        # Shift: predict next token
        shift_logits = logits[:, :-1, :].contiguous()   # (B, seq_len-1, V)
        shift_labels = input_ids[:, 1:].contiguous()    # (B, seq_len-1)

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        loss = loss / grad_accum_steps   # normalize for accumulation

    # --- Backward pass ---
    loss.backward()

    # --- Optimizer step (every grad_accum_steps) ---
    if (step + 1) % grad_accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # --- Logging ---
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item() * grad_accum_steps:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
```

## Training Scale

### Compute Requirements

The total compute (in FLOPs) for pretraining is well-approximated by:

```
C ≈ 6 × N × D

Where:
  C = total FLOPs (floating point operations)
  N = number of model parameters
  D = number of training tokens

The factor of 6 comes from:
  - 2× for forward pass (multiply-add = 2 FLOPs per parameter per token)
  - 2× for backward pass (roughly 2× forward)
  - Total: 2 + 4 = 6 (forward + backward)
```

### Typical Training Scales

| Model | Parameters | Tokens | Compute (FLOPs) | GPUs | Training Time |
|-------|-----------|--------|-----------------|------|---------------|
| GPT-2 | 1.5B | ~40B | ~3.6e20 | 256 V100 | ~1 week |
| GPT-3 | 175B | 300B | ~3.1e23 | 10K V100 | ~1 month |
| LLaMA-1 7B | 7B | 1T | ~4.2e22 | 2048 A100 | ~21 days |
| LLaMA-1 65B | 65B | 1.4T | ~5.5e23 | 2048 A100 | ~21 days |
| LLaMA-2 70B | 70B | 2T | ~8.4e23 | 2048 A100-80GB | ~35 days |
| LLaMA-3 405B | 405B | 15T | ~3.6e25 | 16K H100 | ~54 days |

### Chinchilla Optimal Scaling

```
Chinchilla found that compute-optimal training balances model size and data:

  N_opt ∝ C^0.5     (optimal parameters for compute budget C)
  D_opt ∝ C^0.5     (optimal tokens for compute budget C)

  Ratio: D_opt / N_opt ≈ 20

  In practice, many models overtrain (use more tokens than
  Chinchilla-optimal) because inference cost only depends on N,
  not D. A smaller model trained longer is cheaper to deploy.
```

## Training Dynamics

### Loss Curve

```
Loss
  │
6 │ ·
  │  ··
5 │    ···
  │       ····
4 │           ·····
  │                ······
3 │                      ·········
  │                               ···········
2 │                                          ···················
  │
1 │
  └──────────────────────────────────────────────────────────────
  0                        Steps →                           300K

Typical behavior:
  - Rapid drop in first few thousand steps
  - Slow, steady decrease for the bulk of training
  - Loss spikes may occur (gradient instability)
  - Final loss depends on model size and data quality
```

### Common Training Failures

| Failure | Symptom | Fix |
|---------|---------|-----|
| Loss spike | Sudden jump in loss | Reduce LR, skip batch, gradient clipping |
| Divergence | Loss goes to NaN | Lower LR, check data, fix numerical issues |
| Slow convergence | Loss plateaus early | Increase LR, check data quality |
| Memorization | Train loss drops, eval doesn't | More data, deduplication, regularization |

## Key Takeaways

1. Next-token prediction is the sole objective -- simple but implicitly captures all language understanding
2. Data quality and deduplication matter more than raw dataset size
3. Teacher forcing enables parallel training of all positions in a sequence
4. Mixed precision (BF16) is essential for memory efficiency without sacrificing training stability
5. Compute scales as 6ND -- training costs are dominated by model size times data size
6. Modern models overtrain beyond Chinchilla-optimal ratios to minimize inference cost

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [GPT (Radford et al., 2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | Generative pretraining with next-token prediction |
| [GPT-3 (Brown et al., 2020)](https://arxiv.org/abs/2005.14165) | Scaling laws for pretraining + in-context learning |
| [Chinchilla (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556) | Optimal data-to-parameters ratio (~20:1) |
| [LLaMA (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971) | Open-source, data-efficient pretraining recipe |
| [LLaMA 2 (Touvron et al., 2023)](https://arxiv.org/abs/2307.09288) | Scaling pretraining to 2T tokens |

## Related

- [Optimizer (AdamW)](02_Optimizer.md) -- the optimizer used to update parameters
- [Learning Rate Schedule](03_Learning_Rate.md) -- warmup and decay strategies
- [Regularization](04_Regularization.md) -- weight decay, gradient clipping, dropout
