# Layer Normalization

> Parent: [Transformer Block](00_Transformer_Block.md)

## Overview

Normalization layers stabilize training by controlling the magnitude of activations. Without normalization, activations can grow or shrink across layers, causing unstable gradients and training divergence. In LLMs, **LayerNorm** and its simpler variant **RMSNorm** are the standard choices. BatchNorm, dominant in vision, fails for language due to variable sequence lengths and autoregressive constraints.

## Why Normalization?

The core problem: in a deep network, each layer's output distribution shifts as the parameters of preceding layers change during training (internal covariate shift). This forces later layers to constantly adapt, slowing convergence.

Normalization fixes this by ensuring each layer's inputs have a consistent statistical profile (zero mean, unit variance -- or just unit RMS).

```
Without normalization:              With normalization:

Layer 1 output: mean=2.3, std=15    Layer 1 output: mean≈0, std≈1
Layer 2 output: mean=47, std=892     Layer 2 output: mean≈0, std≈1
Layer 3 output: mean=-2k, std=50k    Layer 3 output: mean≈0, std≈1
         ↓                                    ↓
    Training explodes               Training is stable
```

## BatchNorm vs LayerNorm: Why BN Fails for Sequences

```
Batch of sequences (padded):

         ┌──────────────────────── seq_len ────────────────────────┐
         │                                                         │
Batch 1: [ The  cat  sat  on   the  mat   .   PAD  PAD  PAD  PAD ]
Batch 2: [ Hello  world  PAD  PAD  PAD  PAD  PAD  PAD  PAD  PAD  ]
Batch 3: [ A   long  sentence  with  many  tokens  in   it   .   ]
         │                                                         │
         │                                                         │
         ▼                                                         ▼

BatchNorm: normalizes down each COLUMN (across batch)
  - Column "position 5": mixes "mat", PAD, "tokens"
  - Statistics polluted by padding
  - With batch_size=1 at inference: no batch statistics at all
  - Variable lengths → inconsistent statistics per position

LayerNorm: normalizes across each ROW (across features/d_model)
  - Each token normalized independently using its own features
  - No dependency on other sequences in the batch
  - Works identically at train and inference time
  - Handles variable lengths naturally
```

## Normalization Across Dimensions

```
Input tensor shape: (batch, seq_len, d_model)

                    d_model (e.g., 4096)
                ┌───────────────────────┐
                │                       │
         seq    │  ░░░░░░░░░░░░░░░░░░  │ ← LayerNorm / RMSNorm
         len    │  ░░░░░░░░░░░░░░░░░░  │   normalizes across this
                │  ░░░░░░░░░░░░░░░░░░  │   dimension (per token)
                │  ░░░░░░░░░░░░░░░░░░  │
                │  ░░░░░░░░░░░░░░░░░░  │
                └───────────────────────┘

                ┌───────────────────────┐
                │  ▓                    │ ← BatchNorm normalizes
                │  ▓                    │   down this column
                │  ▓                    │   (across batch dim,
                │  ▓                    │    shown as depth)
                │  ▓                    │
                └───────────────────────┘
```

## LayerNorm Formula

For an input vector `x` of dimension `d`:

```
         x_i - mu
y_i = ------------- * gamma_i + beta_i
      sqrt(sigma^2 + eps)

where:
  mu    = (1/d) * sum(x_i)              mean across features
  sigma^2 = (1/d) * sum((x_i - mu)^2)  variance across features
  gamma ∈ R^d                           learnable scale (initialized to 1)
  beta  ∈ R^d                           learnable shift (initialized to 0)
  eps   = 1e-5 (typically)              numerical stability
```

Steps:
1. **Center**: subtract the mean (zero-mean)
2. **Scale**: divide by standard deviation (unit variance)
3. **Affine**: apply learnable per-feature scale (gamma) and shift (beta)

## RMSNorm Formula

RMSNorm (Zhang & Sennrich, 2019) simplifies LayerNorm by dropping the centering step:

```
          x_i
y_i = ------------- * gamma_i
      sqrt(RMS + eps)

where:
  RMS = (1/d) * sum(x_i^2)             root mean square (no mean subtraction)
  gamma ∈ R^d                           learnable scale only (no beta)
  eps   = 1e-6 (typically)
```

Key differences from LayerNorm:
- **No mean subtraction** (no centering) -- the re-centering was found to be unnecessary
- **No beta** (no learnable shift) -- fewer parameters
- ~10-15% faster due to fewer operations (no mean computation, one fewer pass)

## Comparison: BatchNorm vs LayerNorm vs RMSNorm

| Aspect | BatchNorm | LayerNorm | RMSNorm |
|--------|-----------|-----------|---------|
| Normalizes across | Batch dimension | Feature dimension | Feature dimension |
| Running statistics | Yes (train/eval differ) | No | No |
| Train = Inference | No | Yes | Yes |
| Handles variable seq length | Poorly | Yes | Yes |
| Handles batch_size=1 | Poorly | Yes | Yes |
| Centers (subtract mean) | Yes | Yes | **No** |
| Learnable shift (beta) | Yes | Yes | **No** |
| Learnable scale (gamma) | Yes | Yes | Yes |
| Parameters (for d=4096) | 2 x d = 8192 | 2 x d = 8192 | 1 x d = 4096 |
| Relative speed | Baseline | ~1.0x | ~1.10-1.15x faster |
| Used in LLMs | Rarely | GPT-2, BERT, OG Transformer | LLaMA, Mistral, Gemma, Qwen |

## Which Models Use What

| Model | Normalization | Notes |
|-------|--------------|-------|
| Original Transformer (2017) | LayerNorm | Post-Norm placement |
| BERT (2018) | LayerNorm | Post-Norm |
| GPT-2 (2019) | LayerNorm | Pre-Norm (key innovation) |
| GPT-3 (2020) | LayerNorm | Pre-Norm |
| PaLM (2022) | LayerNorm | Pre-Norm, no bias |
| LLaMA / LLaMA-2 (2023) | RMSNorm | Pre-Norm |
| Mistral / Mixtral (2023-24) | RMSNorm | Pre-Norm |
| Gemma (2024) | RMSNorm | Pre-Norm |
| Qwen-2 (2024) | RMSNorm | Pre-Norm |

## PyTorch Implementation

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """Standard Layer Normalization (as in GPT-2)."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        mu = x.mean(dim=-1, keepdim=True)              # (batch, seq, 1)
        sigma2 = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mu) / torch.sqrt(sigma2 + self.eps)
        return self.gamma * x_norm + self.beta


class RMSNorm(nn.Module):
    """Root Mean Square Normalization (as in LLaMA)."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        rms = x.pow(2).mean(dim=-1, keepdim=True)      # (batch, seq, 1)
        x_norm = x * torch.rsqrt(rms + self.eps)       # rsqrt = 1/sqrt
        return self.gamma * x_norm


# Verify equivalence on centered data
if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(2, 8, 512)

    ln = LayerNorm(512)
    rmsn = RMSNorm(512)

    y_ln = ln(x)
    y_rmsn = rmsn(x)

    print(f"LayerNorm output — mean: {y_ln.mean():.4f}, std: {y_ln.std():.4f}")
    print(f"RMSNorm output   — mean: {y_rmsn.mean():.4f}, std: {y_rmsn.std():.4f}")

    # Note: PyTorch has built-in nn.LayerNorm and (since 2.4) nn.RMSNorm
    # The above are educational implementations
```

## Numerical Stability Note

Both norms use an `eps` constant to avoid division by zero:
- LayerNorm default: `eps = 1e-5`
- RMSNorm default: `eps = 1e-6` (smaller because RMS is always >= 0, no cancellation risk)

In mixed-precision training (fp16/bf16), normalization is typically computed in **fp32** to avoid overflow/underflow, then cast back.

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Batch Normalization (Ioffe & Szegedy, 2015)](https://arxiv.org/abs/1502.03167) | Original normalization for deep networks |
| [Layer Normalization (Ba et al., 2016)](https://arxiv.org/abs/1607.06450) | Normalization across features, not batch |
| [RMSNorm (Zhang & Sennrich, 2019)](https://arxiv.org/abs/1910.07467) | Simplified LayerNorm without centering |

## Related

- [Residual Connection](01_Residual_Connection.md) -- normalization is applied within residual blocks
- [Pre-Norm vs Post-Norm](03_Pre_Norm_Post_Norm.md) -- where normalization is placed relative to sublayers
