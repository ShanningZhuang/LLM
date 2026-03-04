# Context Length Extension

> Parent: [Position Encoding](00_Position_Encoding.md)

## Overview

Models trained with RoPE on short contexts (e.g., 4K tokens) encounter severe quality degradation when applied to longer sequences. The rotation angles at positions beyond the training range produce out-of-distribution frequency patterns that the model has never learned to handle. Context extension methods modify RoPE's frequency structure to map longer sequences into the model's learned frequency range, enabling 4K-trained models to work at 32K, 128K, or even 1M+ tokens.

This is one of the most practically important topics in modern LLM engineering: the ability to cheaply extend a pre-trained model's context window without full retraining.

## The Problem: RoPE's Out-of-Distribution Frequencies

RoPE rotates each dimension pair by `m * theta_i` where `m` is the position and `theta_i = base^{-2i/d}`. During training with max length `L_train`, the model only sees rotation angles in the range `[0, L_train * theta_i]`:

```
Training (L_train = 4096):

    Low-freq pair (i = d/2-1):  max angle = 4096 * 0.0001 = 0.4096 radians
    High-freq pair (i = 0):     max angle = 4096 * 1.0    = 4096 radians
                                                             (many full rotations)

Inference at L_target = 32768:

    Low-freq pair:  max angle = 32768 * 0.0001 = 3.2768 radians  ← 8x larger!
    High-freq pair: max angle = 32768 * 1.0    = 32768 radians
                                                   ← also 8x, but already wraps

    The low-frequency components are the most affected because they
    have NOT completed many full rotations during training and are
    now pushed into unseen angle ranges.
```

### Visualizing the Frequency Problem

```
   Rotation angle vs position for different frequency components:

   angle
    ▲
    │                                            high freq (i=0)
    │                                         ╱  angle grows fast,
    │                                       ╱    wraps many times
    │                                     ╱      (model has seen this pattern)
    │                                   ╱
    │                                 ╱
    │            ╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱
    │
    │                          low freq (i=d/2-1)
    │                        ╱ angle grows slowly,
    │                      ╱   NOW enters unseen range
    │                    ╱
    │              ╱╱╱╱╱
    │         ╱╱╱╱     ┆
    │    ╱╱╱╱          ┆
    │╱╱╱               ┆
    ┼──────────────────┬────────────────────► position
    0              L_train              L_target
                   (4096)               (32768)
                              │
                         ◄────┤
                     OOD zone │
                     for low  │
                     freq     │
```

## Position Interpolation (PI)

**Chen et al., 2023** -- The simplest approach: scale all positions to fit within the training range.

### Method

Instead of using raw positions `[0, 1, 2, ..., L_target-1]`, linearly interpolate them into `[0, L_train-1]`:

```
pos' = pos * (L_train / L_target)

Example (L_train=4096, L_target=32768):
    pos  0 → pos' 0
    pos  8 → pos' 1
    pos 16 → pos' 2
    ...
    pos 32767 → pos' 4095.875
```

All rotation angles now fall within the range the model saw during training.

### Tradeoff

```
Original RoPE positions:          PI positions:
                                  (compressed into training range)
0  1  2  3  4  5  ...  32767     0  0.125  0.25  0.375 ... 4095.875
│  │  │  │  │  │       │         │  │  │  │  │  │       │
▼  ▼  ▼  ▼  ▼  ▼       ▼         ▼  ▼  ▼  ▼  ▼  ▼       ▼
[──────────────────────]         [──────────────────────]
 Original angles for             Same angle range, but
 positions 0..32767              positions are crowded
                                 → reduced resolution for
                                   nearby tokens
```

PI sacrifices **local resolution** (nearby positions are harder to distinguish) to gain **global coverage** (all positions fit in the trained range). Short fine-tuning (200-1000 steps) partially recovers the local resolution.

### PyTorch Implementation

```python
import torch
import math


def rope_frequencies_with_pi(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    original_max_len: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE frequencies with Position Interpolation.

    Args:
        dim: head dimension (must be even)
        max_seq_len: target sequence length
        base: RoPE base frequency
        original_max_len: training sequence length

    Returns:
        cos, sin tables of shape (max_seq_len, dim)
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len).float()

    # Position Interpolation: scale positions down
    scale = original_max_len / max_seq_len
    positions = positions * scale   # compress into [0, original_max_len)

    angles = torch.outer(positions, freqs)
    angles = angles.repeat(1, 2)
    return torch.cos(angles), torch.sin(angles)
```

## NTK-Aware Scaling

**Reddit community + theory, 2023** -- A smarter approach: instead of compressing positions, modify the base frequency to spread out the rotation angles.

### Key Insight

PI treats all frequency components equally, but high-frequency components (which have already completed many rotations) can tolerate more stretching than low-frequency components (which are near their angle limits). NTK-aware scaling modifies the base frequency to achieve a more nuanced adjustment:

```
Standard RoPE:         theta_i = base^{-2i/d}      where base = 10000
NTK-aware scaling:     theta_i = (base * alpha)^{-2i/d}

alpha = (L_target / L_train)^{d/(d-2)}
```

### Why It Works

Increasing the base has a differential effect on different frequency components:

```
Effect of increasing base from 10000 to 40000 (alpha=4):

High-freq dimensions (i near 0):
    theta_i changes from base^{0} = 1.0  to  (base*alpha)^{0} = 1.0
    → NO change for the highest frequency!

Low-freq dimensions (i near d/2):
    theta_i changes from base^{-1} = 0.0001  to  (base*alpha)^{-1} = 0.000025
    → 4x reduction in frequency → angles fit in training range

Mid-freq dimensions:
    Moderate reduction → smooth interpolation between extremes
```

```
   Frequency adjustment across dimensions:

   frequency
   multiplier
    ▲
    │
  1 │ ●──●──●──●─                          ← high-freq dims: unchanged
    │              ─●──●
    │                    ──●
    │                       ──●
    │                          ──●
    │                             ──●
    │                                ──●
0.25│                                   ──●  ← low-freq dims: reduced by alpha
    │
    └──────────────────────────────────────► dimension index i
    0                                    d/2
```

### PyTorch Implementation

```python
def rope_frequencies_ntk(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    original_max_len: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE frequencies with NTK-aware scaling.

    Modifies the base frequency instead of compressing positions.
    """
    scale = max_seq_len / original_max_len
    alpha = scale ** (dim / (dim - 2))
    new_base = base * alpha

    freqs = 1.0 / (new_base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len).float()

    angles = torch.outer(positions, freqs)
    angles = angles.repeat(1, 2)
    return torch.cos(angles), torch.sin(angles)
```

## YaRN (Yet another RoPE extensioN)

**Peng et al., 2023** -- The current state-of-the-art, combining the best ideas from PI and NTK-aware scaling with additional refinements.

### Three Key Components

#### 1. NTK-by-Parts Interpolation

YaRN divides frequency components into three categories and treats each differently:

```
Frequency spectrum divided into three regions:

   ┌──────────────────────────────────────────────────────┐
   │                                                      │
   │   High-freq          Mid-freq           Low-freq     │
   │   (local info)       (medium range)     (long range) │
   │                                                      │
   │   No interpolation   Smooth ramp        Full PI      │
   │   (keep original)    (gradual blend)    (scale down)  │
   │                                                      │
   │   ●●●●●●●           ●●●●●●●●●          ●●●●●●●●●   │
   │   unchanged          NTK blend          compressed   │
   │                                                      │
   └──────────────────────────────────────────────────────┘
        dim 0           dim d/4              dim d/2
```

The boundary between regions is determined by the wavelength of each frequency relative to the original training length:
- If `wavelength < L_train`: high-frequency, keep as-is
- If `wavelength > L_train * scale`: low-frequency, fully interpolate
- Otherwise: blend between the two

#### 2. Attention Scaling (Temperature)

YaRN applies a temperature factor `t` to the attention logits to compensate for the distributional shift in attention entropy:

```
attention = softmax(Q K^T / (sqrt(d) * t))

t = 0.1 * ln(s) + 1    where s = L_target / L_train
```

This prevents the attention distribution from becoming too sharp or too flat after extension.

#### 3. Fine-tuning

YaRN requires minimal fine-tuning (~400 steps on long data) to recover quality, compared to PI which typically needs 1000+ steps.

### PyTorch Implementation

```python
def rope_frequencies_yarn(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    original_max_len: int = 4096,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Compute RoPE frequencies with YaRN extension.

    Args:
        dim: head dimension
        max_seq_len: target sequence length
        base: RoPE base frequency
        original_max_len: training sequence length
        beta_fast: boundary for high-frequency region
        beta_slow: boundary for low-frequency region

    Returns:
        cos, sin tables and attention temperature factor
    """
    scale = max_seq_len / original_max_len

    # Compute wavelengths for each frequency
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    wavelengths = 2 * math.pi / freqs

    # Boundaries in wavelength space
    low_freq_wavelen = original_max_len / beta_slow
    high_freq_wavelen = original_max_len / beta_fast

    # Ramp function: 0 for high-freq, 1 for low-freq, linear between
    ramp = (wavelengths - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
    ramp = ramp.clamp(0, 1)

    # Interpolated frequencies
    freqs_pi = freqs / scale       # Position Interpolation frequencies
    new_freqs = (1 - ramp) * freqs + ramp * freqs_pi   # blend

    # Positions
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, new_freqs)
    angles = angles.repeat(1, 2)

    # Attention temperature
    temperature = 0.1 * math.log(scale) + 1.0

    return torch.cos(angles), torch.sin(angles), temperature
```

## Comparison: PI vs NTK vs YaRN

| Aspect | Position Interpolation (PI) | NTK-Aware Scaling | YaRN |
|--------|---------------------------|-------------------|------|
| Core idea | Compress positions linearly | Increase base frequency | NTK-by-parts + temperature |
| Local resolution | Reduced (positions crowded) | Preserved for high-freq | Best (high-freq untouched) |
| Extension factor | Up to ~8x reliable | Up to ~4-8x | Up to ~16-32x |
| Fine-tuning | Required (1000+ steps) | Optional (works zero-shot) | Minimal (400 steps) |
| Quality at L_train | Slight drop | Good | Best |
| Quality at L_target | Good after fine-tuning | Moderate | Best |
| Implementation | Trivial (one line change) | Simple (change base) | Moderate (three components) |
| Used by | Early LLaMA extensions | Yi, CodeLlama | DeepSeek, Qwen2, LLaMA 3 |
| Paper/Origin | Chen et al., 2023 | Community research, 2023 | Peng et al., 2023 |

## Practical Guide: Extending Context Length

### Step-by-Step Process

```
1. Choose your target length
   └── L_target (e.g., 32768)

2. Choose extension method
   ├── PI:   simplest, needs fine-tuning
   ├── NTK:  moderate, may work zero-shot
   └── YaRN: best quality, minimal fine-tuning

3. Modify RoPE frequencies
   └── Apply chosen scaling to theta values

4. (Optional) Fine-tune on long data
   ├── PI:   ~1000 steps on long documents
   ├── NTK:  ~200-400 steps (or zero-shot)
   └── YaRN: ~400 steps on long documents

5. Evaluate
   ├── Perplexity on long documents
   ├── Needle-in-a-haystack test
   └── Downstream task performance
```

### Quick Reference: How to Modify Frequencies

```python
# Original RoPE (no extension)
freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
angles = torch.outer(positions, freqs)

# PI: scale positions
positions_pi = positions * (L_train / L_target)
angles = torch.outer(positions_pi, freqs)

# NTK: scale base
alpha = (L_target / L_train) ** (dim / (dim - 2))
freqs_ntk = 1.0 / ((base * alpha) ** (torch.arange(0, dim, 2).float() / dim))
angles = torch.outer(positions, freqs_ntk)

# YaRN: per-frequency interpolation (see full implementation above)
```

## Frequency Distribution Changes Under Each Method

```
   Original (L_train = 4096)             After PI (L_target = 32768)
   ┌──────────────────────┐              ┌──────────────────────┐
   │ ● ● ● ● ● ● ● ● ● ●│              │●●●●●●●●●●           │
   │ spread across full   │              │ all compressed to    │
   │ frequency range      │              │ lower freqs (1/8x)   │
   └──────────────────────┘              └──────────────────────┘
   low-freq         high-freq            low-freq         high-freq

   After NTK (L_target = 32768)          After YaRN (L_target = 32768)
   ┌──────────────────────┐              ┌──────────────────────┐
   │ ●● ● ● ●  ●  ●  ● ●│              │●● ●  ●  ●  ●  ● ● ●│
   │ low-freq compressed  │              │ low-freq compressed  │
   │ high-freq preserved  │              │ mid-freq ramped      │
   │ (smooth transition)  │              │ high-freq preserved  │
   └──────────────────────┘              └──────────────────────┘
   low-freq         high-freq            low-freq         high-freq
```

## LongRoPE and Other Recent Methods

### LongRoPE (Ding et al., 2024)

Uses an evolutionary search to find optimal per-dimension scaling factors, rather than applying a uniform formula. Key innovations:
- **Non-uniform scaling**: each frequency component gets its own optimal scale factor
- **Progressive extension**: extend in stages (e.g., 4K -> 128K -> 2M)
- Achieves up to 2M context length with LLaMA-based models

### Self-Extend (Jin et al., 2024)

Uses a two-level approach without fine-tuning:
- Nearby tokens: original RoPE (preserving local quality)
- Distant tokens: grouped positions (coarse but in-distribution)

### Scaling the RoPE Base Directly

Some models simply train with a larger base from the start:
- CodeLlama: base = 1,000,000 (vs standard 10,000) for 100K context
- LLaMA 3: base = 500,000 for 128K context
- This avoids post-hoc extension but requires training with long data

| Method | Max Extension | Fine-tuning | Key Innovation |
|--------|--------------|-------------|----------------|
| PI | ~8x | Required | Position compression |
| NTK-Aware | ~8x | Optional | Base frequency scaling |
| YaRN | ~32x | Minimal | Per-frequency interpolation + temperature |
| LongRoPE | ~500x | Minimal | Evolutionary search for optimal scales |
| Self-Extend | ~4-8x | None | Two-level position grouping |
| Large base | Unlimited | Trained from scratch | Pre-training with large base |

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Position Interpolation (Chen et al., 2023)](https://arxiv.org/abs/2306.15595) | Linear scaling of positions for RoPE extension |
| [YaRN (Peng et al., 2023)](https://arxiv.org/abs/2309.00071) | NTK-by-parts interpolation with attention temperature |
| [LongRoPE (Ding et al., 2024)](https://arxiv.org/abs/2402.13753) | Evolutionary search for per-dimension scaling |
| [Self-Extend (Jin et al., 2024)](https://arxiv.org/abs/2401.01325) | Two-level grouping for training-free extension |
| [Code Llama (Roziere et al., 2023)](https://arxiv.org/abs/2308.12950) | Large RoPE base for long-context code models |

## Related

- [RoPE](02_RoPE.md) -- the base position encoding that these methods extend
- [Absolute Encoding](01_Absolute_Encoding.md) -- earlier approaches without extension capabilities
- [ALiBi](03_ALiBi.md) -- alternative that naturally generalizes without extension
