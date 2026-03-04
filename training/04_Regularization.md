# Regularization

> Parent: [Training](00_Training.md)

## Overview

Regularization techniques prevent overfitting and stabilize training. Surprisingly, modern LLM pretraining uses **very little regularization** — the sheer scale of training data makes overfitting unlikely. The main techniques are weight decay, gradient clipping, and occasionally dropout.

---

## Dropout

### Mechanism

Randomly zero out activations during training with probability p:

```
Training:     h = [0.5, 0.0, 0.3, 0.0, 0.8, 0.2, 0.0, 0.4]  (p=0.3)
                    ↑    ×    ↑    ×    ↑    ↑    ×    ↑
              Scale surviving values by 1/(1-p) to maintain expected value

Inference:    h = [0.5, 0.7, 0.3, 0.1, 0.8, 0.2, 0.6, 0.4]  (no dropout)
```

### Where Dropout is Applied

```
Transformer Block:
┌───────────────────────┐
│  Attention             │
│    └─ dropout(attn_weights)  ← On attention probabilities
│    └─ dropout(attn_output)   ← On output projection
├───────────────────────┤
│  FFN                   │
│    └─ dropout(ffn_output)    ← On FFN output
├───────────────────────┤
│  Residual + Norm       │
│    └─ dropout before add     ← Before residual addition
└───────────────────────┘
```

### Modern LLMs: No Dropout

| Model | Dropout Rate | Notes |
|-------|-------------|-------|
| GPT-2 | 0.1 | Used dropout everywhere |
| GPT-3 | 0.1 | Still used dropout |
| LLaMA | **0.0** | No dropout |
| LLaMA-2 | **0.0** | No dropout |
| Mistral | **0.0** | No dropout |
| PaLM | **0.0** | No dropout |
| Qwen | **0.0** | No dropout |

**Why no dropout in large-scale pretraining?**
- Training data is massive (trillions of tokens) — each example seen ~1-2 times
- Underfitting (not overfitting) is the dominant problem
- Dropout slows convergence for a given number of tokens
- Dropout is still useful for **fine-tuning** on small datasets

### PyTorch

```python
import torch.nn as nn

# In model definition
self.dropout = nn.Dropout(p=0.0)  # Set to 0.0 for pretraining

# In forward pass
x = x + self.dropout(self.attention(self.norm1(x)))
x = x + self.dropout(self.ffn(self.norm2(x)))
```

---

## Weight Decay

### What It Does

Penalizes large weight values by shrinking weights toward zero each step:

```
AdamW update:
    θ_t = θ_{t-1} - lr × (adam_update + λ × θ_{t-1})
                                          ↑
                              weight decay: directly shrink weights
```

### L2 Regularization vs Weight Decay

```
L2 Regularization:          Weight Decay (AdamW):
loss = L + (λ/2)||θ||²      θ -= lr × λ × θ  (applied separately)
gradient: ∇L + λθ            Not through the gradient!

In SGD: equivalent
In Adam: NOT equivalent! (because Adam scales gradients by 1/√v)
→ This is why AdamW exists (decoupled weight decay)
```

### Which Parameters Get Weight Decay?

```python
# Standard practice: decay weights, NOT biases or norms
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if 'bias' in name or 'norm' in name or 'layernorm' in name:
        no_decay_params.append(param)    # No decay
    else:
        decay_params.append(param)       # Decay with λ=0.1

optimizer = torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.1},
    {'params': no_decay_params, 'weight_decay': 0.0},
], lr=3e-4)
```

**Why not decay biases and norms?**
- Biases: small, don't contribute much to overfitting
- Norm params (γ, β): control scale/shift, decaying them hurts training stability

### Typical Values

| Model | Weight Decay (λ) |
|-------|------------------|
| GPT-2 | 0.01 |
| GPT-3 | 0.1 |
| LLaMA | 0.1 |
| LLaMA-2 | 0.1 |
| Chinchilla | 0.1 |

---

## Gradient Clipping

### Why It's Needed

Training loss spikes can produce enormous gradients that destabilize training:

```
Normal training:
    loss: 3.2 → 3.1 → 3.0 → 2.9 → 2.8

With gradient explosion:
    loss: 3.2 → 3.1 → 3.0 → 15.7 → NaN → dead
                              ↑
                         gradient spike!

With gradient clipping:
    loss: 3.2 → 3.1 → 3.0 → 3.5 → 3.1 → 2.9
                              ↑
                         clipped, recovers
```

### Global Norm Clipping

The standard method clips the **global** gradient norm (across all parameters):

```
1. Compute global norm:  g_norm = √(Σᵢ ||∇θᵢ||²)
2. If g_norm > max_norm:
       scale = max_norm / g_norm
       ∇θᵢ = ∇θᵢ × scale   (for all i)
```

This preserves the **direction** of the gradient while limiting its magnitude.

### PyTorch

```python
import torch.nn.utils as utils

# Standard: clip global norm to 1.0
loss.backward()
utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### Typical Values

| Model | Max Gradient Norm |
|-------|-------------------|
| GPT-2 | 1.0 |
| GPT-3 | 1.0 |
| LLaMA | 1.0 |
| PaLM | 1.0 |
| Nearly all LLMs | **1.0** |

The value 1.0 is nearly universal.

---

## Other Techniques

### Z-Loss (PaLM)

Penalizes large logits to prevent the softmax from becoming too peaked:

```
z_loss = 1e-4 × log²(Σⱼ exp(logitsⱼ))
total_loss = cross_entropy_loss + z_loss
```

Stabilizes training for very large models (500B+).

### Embedding Normalization

Some models normalize the embedding output:

```python
# Scale embeddings by √d_model (original Transformer)
x = self.embedding(token_ids) * math.sqrt(d_model)
```

This compensates for the small magnitude of embedding vectors.

### Label Smoothing

Rarely used in LLM pretraining. Replaces hard targets with smoothed distribution:

```
Hard target:     [0, 0, 1, 0, 0]  (one-hot)
Smoothed (ε=0.1): [0.025, 0.025, 0.9, 0.025, 0.025]
```

---

## Why LLMs Need Less Regularization

```
Small model + small data:     Overfitting risk HIGH
    → Need: dropout, L2, data augmentation, early stopping

Large model + massive data:   Overfitting risk LOW
    → Need: weight decay + gradient clipping (that's it)

The scaling laws explain why:
    L(D) = (D_c / D)^β
    Loss keeps decreasing with more data
    → With trillions of tokens, underfitting is the problem
```

### Regularization Summary for Modern LLMs

| Technique | Used? | Typical Value | Purpose |
|-----------|-------|---------------|---------|
| Dropout | **No** (pretraining) | 0.0 | — |
| Weight decay | **Yes** | 0.1 | Prevent weight explosion |
| Gradient clipping | **Yes** | 1.0 | Prevent gradient spikes |
| Z-loss | Sometimes | 1e-4 | Stabilize logits |
| Label smoothing | Rarely | — | — |
| Data augmentation | **No** | — | Not applicable to LM |
| Early stopping | **No** | — | Train to token budget |

---

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Dropout (Srivastava et al., 2014)](https://jmlr.org/papers/v15/srivastava14a.html) | Dropout regularization |
| [AdamW (Loshchilov & Hutter, 2019)](https://arxiv.org/abs/1711.05101) | Decoupled weight decay |
| [PaLM (Chowdhery et al., 2022)](https://arxiv.org/abs/2204.02311) | Z-loss for large-scale stability |

## Related

- [Optimizer](02_Optimizer.md) — AdamW handles weight decay
- [Learning Rate](03_Learning_Rate.md) — LR schedule is the most important "regularizer"
- [Pretraining](01_Pretraining.md) — Training pipeline context
