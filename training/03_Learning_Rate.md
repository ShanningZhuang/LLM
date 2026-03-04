# Learning Rate Schedule

> Parent: [Training](00_Training.md)

## Overview

The learning rate is the single most important hyperparameter in LLM training. Too high and training diverges; too low and convergence is painfully slow. Modern LLM training uses a **schedule** that varies the learning rate across training: warmup from zero, then decay toward a minimum. Getting this right can mean the difference between a strong model and a wasted training run.

## Why a Schedule Is Needed

```
Constant learning rate problems:

High LR throughout:
  - Good: fast initial progress
  - Bad: oscillates around minimum, never converges precisely
  - Risk: catastrophic loss spikes or divergence

Low LR throughout:
  - Good: stable, precise convergence
  - Bad: takes forever, may get stuck in bad local minimum
  - Wasted compute

Solution: start low → ramp up → decay down
  - Warmup: stabilize early training
  - Peak: fast learning during main training
  - Decay: precise convergence in final phase
```

## Linear Warmup

### The Concept

```
During the first warmup_steps, linearly increase LR from ~0 to peak_lr:

  lr(t) = peak_lr × (t / warmup_steps)    for t < warmup_steps

Typical warmup: 2000 steps (sometimes 0.1-1% of total training)
```

### Why Warmup Helps

```
At initialization:
  - Weights are random → activations and gradients are noisy
  - Adam's v_t (second moment) is initialized to 0
  - Bias correction makes early updates disproportionately large
  - Large LR + noisy gradients = risk of divergence

Warmup gives Adam time to:
  1. Build up accurate estimates of m_t and v_t
  2. Let the model find a reasonable loss basin
  3. Stabilize gradient statistics before taking large steps

After ~2000 steps:
  - v_t has seen enough gradients to be a good variance estimate
  - The model has moved to a reasonable region of parameter space
  - Safe to use the full learning rate
```

## Cosine Decay

After warmup, the learning rate decays following a cosine curve:

```
lr(t) = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t_decay / T_decay))

Where:
  t_decay = t - warmup_steps       (steps since warmup ended)
  T_decay = total_steps - warmup_steps   (total decay period)
  lr_max  = peak learning rate
  lr_min  = minimum learning rate (typically 0.1 × lr_max)
```

### Why Cosine?

```
Cosine decay properties:
  1. Starts slow (gentle transition from peak LR)
  2. Fastest decay in the middle of training
  3. Ends slow (gradual approach to minimum LR)

Compare with linear decay:
  - Linear: constant rate of decrease
  - Cosine: S-shaped, spends more time near peak and near minimum

The slow start preserves fast learning early.
The slow end allows precise convergence.
```

## Learning Rate Schedule Diagram

```
LR
 │
 │          peak_lr = 3e-4
 │         ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
 │        ╱                    ╲
 │       ╱                      ╲
 │      ╱         Cosine         ╲
 │     ╱          Decay           ╲
 │    ╱                            ╲
 │   ╱                              ╲
 │  ╱                                ╲___________  lr_min = 3e-5
 │ ╱
 │╱
 │ Linear
 │ Warmup
 └────────┬──────────────────────────────────────
          │                                    Steps
     warmup_steps                        total_steps
      (2000)                              (300K)

 Phase 1: Linear warmup     (steps 0 → 2000)
 Phase 2: Cosine decay       (steps 2000 → 300K)
```

### Comparison of Decay Schedules

```
LR
 │
 │     * ─ ─ ─ ─ ─ ─ ─ ─ ─ ─      Constant (no decay)
 │    *╲ ·
 │   *  ╲   ·····
 │  *    ╲       ·····
 │ *      ╲           ·····
 │*        ╲               ·····
 │          ╲                   ·····
 │           ╲                       ·····
 │            ╲                           ····· ─  Cosine
 │             ╲
 │              ╲  ─  Linear
 └────────────────────────────────────────────────
                        Steps →

 ─ ─ ─  Constant: no decay, bad convergence
 ─────  Linear: constant rate of decrease
 ·····  Cosine: slow start/end, fast middle (most popular)
```

## WSD Schedule (Warmup-Stable-Decay)

A simpler alternative used by some recent models:

```
LR
 │
 │      ┌──────────────────────────────┐
 │     ╱│          Stable Phase         │╲
 │    ╱ │                               │ ╲
 │   ╱  │       (constant LR)          │  ╲
 │  ╱   │                               │   ╲
 │ ╱    │                               │    ╲___
 │╱     │                               │
 │      │                               │
 │Warmup│                               │Decay
 └──────┴───────────────────────────────┴───────
      2K              Most of training       Last
     steps                                  10-20%

Phases:
  1. Warmup:  linear increase (same as cosine schedule)
  2. Stable:  constant peak LR
  3. Decay:   rapid decay (linear, cosine, or exponential)
```

### WSD Advantages

```
Key advantage: training can be stopped at any point during the stable phase
and the model is still useful (just apply the decay phase).

With cosine decay:
  - Must train for the full planned duration
  - Stopping early means the LR is still high → suboptimal

With WSD:
  - Train in stable phase as long as budget allows
  - Apply decay phase at the end
  - Can decide training duration dynamically

Used by: MiniCPM, some DeepSeek experiments
```

## Choosing the Peak Learning Rate

Peak LR scales approximately inversely with model size:

| Model Size | Typical Peak LR | Source |
|------------|-----------------|--------|
| 125M | 6.0e-4 | GPT-3 paper |
| 350M | 3.0e-4 | GPT-3 paper |
| 1.3B | 2.0e-4 | GPT-3 paper |
| 7B | 3.0e-4 | LLaMA |
| 13B | 3.0e-4 | LLaMA |
| 33B | 1.5e-4 | LLaMA |
| 65B | 1.5e-4 | LLaMA |
| 70B | 1.5e-4 | LLaMA-2 |

```
Rough scaling rule:
  peak_lr ∝ 1 / sqrt(d_model)

  d_model = 768  (125M)  → lr ≈ 6e-4
  d_model = 4096 (7B)    → lr ≈ 3e-4
  d_model = 8192 (65B)   → lr ≈ 1.5e-4

Intuition: larger models have more parameters interacting,
so each individual update should be smaller to maintain stability.

μP (Maximal Update Parameterization):
  Provides a principled way to transfer optimal LR from a small
  proxy model to a large target model.
  → Tune hyperparameters on a 125M model, transfer to 7B+
```

## Batch Size Warmup

Some training runs also ramp up the batch size during early training:

```
Batch Size Warmup:

Tokens/step
    │
 4M │                    ┌──────────────────────────
    │                   ╱
 3M │                  ╱
    │                 ╱
 2M │                ╱
    │               ╱
 1M │  ────────────╱
    │
    └──────────────┬────────────────────────────────
                  4K                              Steps

Phase 1 (steps 0-4K): small batch (1M tokens)
  → More optimizer steps per token (faster exploration)
  → Lower memory usage during unstable early training

Phase 2 (steps 4K+): full batch (4M tokens)
  → Larger batch → lower gradient noise → stable training
  → Better GPU utilization
```

### Why Batch Size Warmup Helps

```
Small batch:
  + More gradient updates per token seen
  + Better exploration of loss landscape early on
  + Lower memory (useful if activations are large)
  - Noisier gradients

Large batch:
  + Lower gradient noise → more stable training
  + Better hardware utilization (higher throughput)
  - Fewer updates per token (each step sees more tokens)

Strategy: start small (explore) → go large (exploit)

Used by: GPT-3, BLOOM, various other large-scale runs
```

## PyTorch Implementation

### Cosine Schedule with Warmup

```python
import math
import torch

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,   # lr_min = min_lr_ratio × peak_lr
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear warmup followed by cosine decay.
    This is the most common schedule for LLM pretraining.
    """
    def lr_lambda(current_step: int) -> float:
        # Phase 1: Linear warmup
        if current_step < warmup_steps:
            return current_step / warmup_steps

        # Phase 2: Cosine decay
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --- Usage ---
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    warmup_steps=2000,
    total_steps=300_000,
    min_lr_ratio=0.1,    # decay to 3e-5
)

for step, batch in enumerate(dataloader):
    loss = train_step(model, batch)
    loss.backward()
    optimizer.step()
    scheduler.step()     # update LR after each step
    optimizer.zero_grad()
```

### WSD Schedule

```python
def get_wsd_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warmup-Stable-Decay schedule."""
    total_steps = warmup_steps + stable_steps + decay_steps

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Phase 1: Linear warmup
            return current_step / warmup_steps
        elif current_step < warmup_steps + stable_steps:
            # Phase 2: Constant
            return 1.0
        else:
            # Phase 3: Cosine decay
            decay_progress = (current_step - warmup_steps - stable_steps) / decay_steps
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### Using Built-in PyTorch Schedulers

```python
# Option 1: CosineAnnealingLR (no warmup built-in)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_steps,
    eta_min=3e-5,
)

# Option 2: SequentialLR (combine warmup + cosine)
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps - warmup_steps, eta_min=3e-5
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup, cosine],
    milestones=[warmup_steps],
)
```

## LR Schedules Used by Different Models

| Model | Schedule | Peak LR | Warmup Steps | Min LR | Total Tokens |
|-------|----------|---------|-------------|--------|-------------|
| GPT-3 175B | Cosine | 6e-5 | 375 | 0 | 300B |
| LLaMA-1 7B | Cosine | 3e-4 | 2000 | 3e-5 | 1T |
| LLaMA-1 65B | Cosine | 1.5e-4 | 2000 | 1.5e-5 | 1.4T |
| LLaMA-2 70B | Cosine | 1.5e-4 | 2000 | 1.5e-5 | 2T |
| LLaMA-3 8B | Cosine | 3e-4 | 8000 | 3e-5 | 15T |
| Mistral 7B | Cosine | 3e-4 | ~2000 | ~3e-5 | ~2T |
| MiniCPM 2.4B | WSD | 1e-3 | 2000 | 0 | ~1T |
| Chinchilla 70B | Cosine | 1e-4 | ~400 | 1e-5 | 1.4T |

## Practical Tips

```
Debugging LR issues:

1. Loss doesn't decrease at all:
   → LR too low, or warmup too long
   → Try 10x higher peak LR

2. Loss spikes frequently:
   → LR too high
   → Try 2-3x lower peak LR
   → Check gradient clipping is active

3. Loss plateaus in the middle of training:
   → Cosine decay may be too aggressive
   → Try WSD to maintain peak LR longer
   → Check data quality (may have exhausted good data)

4. Final loss is higher than expected:
   → min_lr might be too high (or too low)
   → Try min_lr = 0 vs min_lr = 0.1 × peak

General rule of thumb:
  If unsure, start with LLaMA's recipe:
    peak_lr = 3e-4, warmup = 2000, cosine to 10% of peak
```

## Key Takeaways

1. The learning rate schedule has three phases: warmup (stabilize), peak (learn), decay (converge)
2. Linear warmup lets Adam's moment estimates stabilize before taking large steps
3. Cosine decay is the most common schedule -- it smoothly reduces the LR to a minimum
4. WSD is a simpler alternative that allows dynamic training duration decisions
5. Peak LR scales roughly as 1/sqrt(d_model) -- larger models need smaller learning rates
6. Batch size warmup complements LR warmup by starting with more updates per token

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [SGDR: Stochastic Gradient Descent with Warm Restarts (Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1608.03983) | Cosine annealing schedule |
| [GPT-3 (Brown et al., 2020)](https://arxiv.org/abs/2005.14165) | LR schedule recipe for large-scale training |
| [Chinchilla (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556) | LR tuning at scale |
| [muP (Yang et al., 2022)](https://arxiv.org/abs/2203.03466) | Principled hyperparameter transfer across model sizes |
| [MiniCPM (Hu et al., 2024)](https://arxiv.org/abs/2404.06395) | WSD schedule for flexible training |

## Related

- [Pretraining](01_Pretraining.md) -- the training loop that uses this schedule
- [Optimizer (AdamW)](02_Optimizer.md) -- the optimizer whose LR is being scheduled
- [Regularization](04_Regularization.md) -- other knobs for controlling training dynamics
