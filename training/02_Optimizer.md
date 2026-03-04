# Optimizer (AdamW)

> Parent: [Training](00_Training.md)

## Overview

The optimizer determines how gradients are converted into parameter updates. Modern LLMs universally use **AdamW** -- a variant of Adam with decoupled weight decay. Understanding the evolution from SGD to Adam to AdamW reveals why AdamW is the default choice, and its memory cost is a major factor in distributed training design.

## Evolution: SGD to Adam to AdamW

```
SGD (Stochastic Gradient Descent):
  θ_t = θ_{t-1} - lr · g_t

  Problem: same learning rate for all parameters.
  Dense gradients (embeddings) and sparse gradients (rare tokens)
  get the same update magnitude.

SGD + Momentum:
  m_t = β · m_{t-1} + g_t
  θ_t = θ_{t-1} - lr · m_t

  Better: momentum smooths noisy gradients.
  Problem: still one global learning rate.

Adam (Adaptive Moment Estimation):
  Maintains per-parameter first and second moment estimates.
  Adapts learning rate per parameter.
  → Fast convergence, works well across many settings.

AdamW (Adam with decoupled Weight decay):
  Fixes a subtle bug in how Adam handles L2 regularization.
  → The standard for LLM training.
```

## Adam: The Math

Adam maintains two exponential moving averages per parameter:

```
g_t = gradient at step t

First moment (mean of gradients):
  m_t = β₁ · m_{t-1} + (1 - β₁) · g_t

Second moment (mean of squared gradients):
  v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²

Bias correction (needed because m_0 = v_0 = 0):
  m̂_t = m_t / (1 - β₁^t)
  v̂_t = v_t / (1 - β₂^t)

Parameter update:
  θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε)
```

### Why Bias Correction Matters

```
At step 1 with β₁ = 0.9:
  m_1 = 0.9 · 0 + 0.1 · g_1 = 0.1 · g_1     ← biased toward 0

Without correction: update uses 0.1 · g_1 (too small)
With correction:    m̂_1 = 0.1 · g_1 / (1 - 0.9) = g_1  ← correct scale

As t → ∞: (1 - β^t) → 1, so correction vanishes.
```

### Intuition for Per-Parameter Learning Rates

```
Parameter A: gradients are consistently large
  → v_t is large → √v̂_t is large → effective LR is small
  → "I'm already learning fast, slow down"

Parameter B: gradients are consistently small
  → v_t is small → √v̂_t is small → effective LR is large
  → "I'm learning slowly, speed up"

Parameter C: gradients vary wildly
  → v_t is large (due to squared terms) → effective LR is small
  → "Uncertain direction, take small steps"

Effective learning rate per parameter ≈ lr / √v̂_t
```

## AdamW: Decoupled Weight Decay

### The Problem with L2 Regularization in Adam

```
L2 regularization adds a penalty term to the loss:
  L_total = L_original + (λ/2) · ||θ||²

The gradient of L2 penalty: ∂(λ/2 · ||θ||²)/∂θ = λ · θ

In standard Adam with L2:
  g_t = ∇L_original + λ · θ_{t-1}        ← L2 gradient added
  m_t = β₁ · m_{t-1} + (1-β₁) · g_t      ← weight decay is smoothed by momentum
  v_t = β₂ · v_{t-1} + (1-β₂) · g_t²     ← weight decay inflates v_t
  θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε)  ← update is adapted

Problem: The adaptive scaling (dividing by √v̂_t) is applied to
the weight decay gradient too! This weakens the regularization
effect for parameters with large gradients.

Key insight: L2 regularization ≠ weight decay when using Adam.
```

### The AdamW Fix

```
AdamW separates weight decay from the gradient computation:

  g_t = ∇L_original                           ← no L2 term in gradient
  m_t = β₁ · m_{t-1} + (1 - β₁) · g_t
  v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²
  θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε) - lr · λ · θ_{t-1}
                   ↑                           ↑
               adaptive update              weight decay
               (same as Adam)               (applied directly)

Weight decay is NOT scaled by the adaptive term.
Every parameter decays at the same rate relative to its magnitude.
```

### Visual Comparison

```
Adam + L2:
  gradient ─→ [add L2 penalty] ─→ [Adam adaptive scaling] ─→ update
                                         ↑
                              L2 penalty is also scaled
                              (weakened for large-gradient params)

AdamW:
  gradient ─→ [Adam adaptive scaling] ─→ adaptive update ─┐
                                                           ├─→ total update
  weight decay ────────────────────────→ decay term ───────┘
                  (applied directly, not scaled)
```

## Typical Hyperparameters

| Hyperparameter | Symbol | Typical Value | Notes |
|----------------|--------|---------------|-------|
| Learning rate | lr | 1e-4 to 6e-4 | See [Learning Rate Schedule](03_Learning_Rate.md) |
| Beta1 | beta1 | 0.9 | First moment decay. Some use 0.9 early, lower later |
| Beta2 | beta2 | 0.95 | Second moment decay. GPT-3 used 0.95 (not 0.999) |
| Epsilon | eps | 1e-8 | Prevents division by zero |
| Weight decay | lambda | 0.1 | Applied to weight matrices only |

### Why beta2=0.95 instead of 0.999?

```
Default Adam: β₂ = 0.999 → v_t averages over ~1000 steps
LLM training: β₂ = 0.95  → v_t averages over ~20 steps

In LLM training, the loss landscape changes rapidly:
- Different batches come from very different text distributions
- The model's representation changes significantly during training
- A shorter window for v_t tracks the current landscape better

GPT-3, LLaMA, and most modern LLMs use β₂ = 0.95.
```

## Memory Cost of Adam/AdamW

Adam stores two additional states per parameter (m and v), both in FP32:

```
Memory per parameter:
  - Weight:          2 bytes (BF16) or 4 bytes (FP32)
  - Gradient:        2 bytes (BF16)
  - Optimizer m:     4 bytes (FP32)
  - Optimizer v:     4 bytes (FP32)
  - FP32 master copy: 4 bytes (if weights stored in BF16)

Total per parameter: 2 + 2 + 4 + 4 + 4 = 16 bytes (with BF16 weights)
                     4 + 2 + 4 + 4     = 14 bytes (with FP32 weights, no copy)

Example: 7B parameter model
  Weights (BF16):         7B × 2  = 14 GB
  Gradients (BF16):       7B × 2  = 14 GB
  Optimizer states (FP32): 7B × 8  = 56 GB   ← dominates!
  FP32 master weights:    7B × 4  = 28 GB
                                   ────────
  Total:                           112 GB

  A single A100-80GB cannot even hold the optimizer states +
  master weights for a 7B model!
```

### Memory Breakdown Diagram

```
7B Model Training Memory (approximate):

Weights (BF16):     ██████████████ 14 GB
Gradients (BF16):   ██████████████ 14 GB
Master weights:     ████████████████████████████ 28 GB
Optimizer m (FP32): ████████████████████████████ 28 GB
Optimizer v (FP32): ████████████████████████████ 28 GB
                    ────────────────────────────────────
Total:              112 GB (without activations!)

Activations add another 20-100+ GB depending on seq_len and batch size.
→ This is why distributed training (ZeRO, FSDP) is essential.
```

## Gradient Accumulation

Simulate larger batch sizes without more GPU memory:

```
Effective batch size = micro_batch_size × grad_accum_steps × n_gpus

Example:
  micro_batch_size = 4 (sequences per GPU per forward pass)
  grad_accum_steps = 32
  n_gpus = 8
  seq_len = 4096

  Tokens per step = 4 × 32 × 8 × 4096 = 4,194,304 ≈ 4M tokens

The process:
  for i in range(grad_accum_steps):
      loss = model(micro_batch[i]) / grad_accum_steps  ← normalize
      loss.backward()                                   ← accumulate gradients
  optimizer.step()                                      ← single update
  optimizer.zero_grad()

Memory: only need to store activations for one micro-batch at a time.
Gradients accumulate in-place.
```

## 8-bit Adam and Memory-Efficient Variants

| Variant | Memory Savings | Idea |
|---------|---------------|------|
| 8-bit Adam (bitsandbytes) | ~50% optimizer states | Quantize m, v to INT8 with dynamic scaling |
| Adafactor | ~67% optimizer states | Factor v into row and column vectors |
| CAME | ~50% optimizer states | Confidence-guided adaptive memory efficient |
| LION | 50% optimizer states | Only tracks first moment (sign-based update) |
| Sophia | Similar to Adam | Uses second-order Hessian info instead of v |

```
8-bit Adam (Dettmers et al., 2022):
  Store m and v in INT8 instead of FP32.
  Use block-wise dynamic quantization.

  Standard:  m, v → 4 bytes each → 8 bytes per param
  8-bit:     m, v → 1 byte each  → 2 bytes per param

  7B model optimizer states:
    Standard: 56 GB → 8-bit: 14 GB  (4× reduction)

  Quality: nearly identical training loss curves in practice.
```

## PyTorch Code

### Standard AdamW Training

```python
import torch
import torch.nn as nn

# --- Model setup ---
model = MyLLM(...)
model = model.cuda()

# --- Separate parameter groups for weight decay ---
decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        # Don't apply weight decay to biases, LayerNorm, or RMSNorm
        if "bias" in name or "norm" in name or "layernorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
)

print(f"Decay params:    {sum(p.numel() for p in decay_params):,}")
print(f"No-decay params: {sum(p.numel() for p in no_decay_params):,}")
```

### Gradient Accumulation Pattern

```python
grad_accum_steps = 32

for step, batch in enumerate(dataloader):
    # Forward + backward (accumulate gradients)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        loss = compute_loss(model, batch) / grad_accum_steps
    loss.backward()

    # Update only every grad_accum_steps
    if (step + 1) % grad_accum_steps == 0:
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
```

### 8-bit Adam with bitsandbytes

```python
import bitsandbytes as bnb

optimizer = bnb.optim.Adam8bit(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

# Usage is identical to standard Adam
optimizer.step()
```

## Comparison Table: SGD vs Adam vs AdamW

| Property | SGD + Momentum | Adam | AdamW |
|----------|---------------|------|-------|
| Per-parameter LR | No | Yes | Yes |
| Momentum | Yes (one state) | Yes (first moment) | Yes (first moment) |
| Adaptive scaling | No | Yes (second moment) | Yes (second moment) |
| Weight decay | Equivalent to L2 | NOT equivalent to L2 | Properly decoupled |
| Memory per param | +4 bytes (momentum) | +8 bytes (m + v) | +8 bytes (m + v) |
| Convergence speed | Slow for LLMs | Fast | Fast |
| Typical use | CV, small models | NLP (historical) | LLM pretraining (standard) |
| Hyperparameter sensitivity | High (LR is critical) | Moderate | Moderate |
| 7B model optimizer mem | 28 GB | 56 GB | 56 GB |

## Key Takeaways

1. Adam adapts the learning rate per parameter using first moment (direction) and second moment (scale)
2. AdamW fixes a subtle but important issue: weight decay in Adam should NOT be scaled by the adaptive term
3. Optimizer states (m, v in FP32) dominate memory -- 2x the model size for Adam/AdamW
4. Weight decay is applied to weight matrices but NOT to biases, LayerNorm/RMSNorm parameters
5. LLMs use beta2=0.95 (not 0.999) for faster adaptation to changing loss landscapes
6. 8-bit Adam and Adafactor can reduce optimizer memory by 2-4x with minimal quality loss

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Adam (Kingma & Ba, 2015)](https://arxiv.org/abs/1412.6980) | Adaptive moment estimation optimizer |
| [AdamW (Loshchilov & Hutter, 2019)](https://arxiv.org/abs/1711.05101) | Decoupled weight decay regularization |
| [8-bit Adam (Dettmers et al., 2022)](https://arxiv.org/abs/2110.02861) | Memory-efficient 8-bit optimizer states |
| [Adafactor (Shazeer & Stern, 2018)](https://arxiv.org/abs/1804.04235) | Factored second moment for memory efficiency |
| [LION (Chen et al., 2023)](https://arxiv.org/abs/2302.06675) | Sign-based optimizer with only first moment |

## Related

- [Pretraining](01_Pretraining.md) -- the training loop that uses the optimizer
- [Learning Rate Schedule](03_Learning_Rate.md) -- how lr changes during training
- [Regularization](04_Regularization.md) -- weight decay and gradient clipping details
