# Scaling Laws

> Parent: [Architecture](00_Architecture.md)

## Overview

Scaling laws describe how LLM performance (measured as cross-entropy loss) improves predictably with increases in model size (N), dataset size (D), and compute budget (C). These power-law relationships allow researchers to predict the performance of large models from small-scale experiments and make optimal resource allocation decisions before committing to expensive training runs.

## Learning Objectives

- [x] Kaplan et al. (2020) scaling laws
- [x] Chinchilla (Hoffmann et al., 2022) optimal compute allocation
- [x] Key formulas and power-law exponents
- [x] Practical implications for model training
- [x] Emergent abilities debate

## Resources

### Papers

| Paper | Year | Key Contribution |
|-------|------|------------------|
| [Scaling Laws for Neural Language Models (Kaplan et al.)](https://arxiv.org/abs/2001.08361) | 2020 | Power-law relationships for N, D, C |
| [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) | 2022 | Revised optimal N:D ratio |
| [Emergent Abilities of LLMs (Wei et al.)](https://arxiv.org/abs/2206.07682) | 2022 | Emergent abilities claim |
| [Are Emergent Abilities a Mirage? (Schaeffer et al.)](https://arxiv.org/abs/2304.15004) | 2023 | Metric artifact argument |
| [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) | 2023 | What to do when data runs out |

### Blogs & Tutorials

- [Chinchilla's Wild Implications (Nostalgebraist)](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications)
- [The Bitter Lesson (Rich Sutton)](http://www.incompleteideas.net/IncsijBitterLesson.html)

---

## Kaplan et al. (2020) -- Original Scaling Laws

### Core Finding

Loss follows a **power law** in each of three factors independently: model parameters (N), dataset size (D), and compute (C). The relationships are smooth and predictable over many orders of magnitude.

### Key Formulas

```
Loss as a function of model size (parameters):
  L(N) = (N_c / N)^alpha_N       alpha_N ≈ 0.076
                                   N_c ≈ 8.8 x 10^13

Loss as a function of dataset size (tokens):
  L(D) = (D_c / D)^alpha_D       alpha_D ≈ 0.095
                                   D_c ≈ 5.4 x 10^13

Loss as a function of compute (FLOPs):
  L(C) = (C_c / C)^alpha_C       alpha_C ≈ 0.050
                                   C_c ≈ 3.1 x 10^8

Combined (when neither N nor D is bottleneck):
  L(N, D) = [(N_c/N)^(alpha_N/alpha_D) + (D_c/D)]^alpha_D

Compute-model size relationship:
  C ≈ 6 * N * D     (FLOPs for training)

where:
  N = number of non-embedding parameters
  D = number of training tokens
  C = compute budget in FLOPs
  L = cross-entropy loss (nats)
```

### Kaplan's Key Claims

```
1. Performance depends strongly on scale, weakly on shape:
   - Doubling width vs doubling depth → similar loss reduction
   - N (total params) matters much more than architecture details

2. Smooth power laws hold over 7+ orders of magnitude:
   - 10^3 to 10^10 parameters
   - 10^7 to 10^12 tokens
   - 10^12 to 10^21 FLOPs

3. Larger models are more sample-efficient:
   - A 10x larger model needs ~3x fewer tokens to reach same loss
   - Implication: scale up N faster than D

4. Optimal allocation (Kaplan):
   - Given fixed compute C, allocate most to model size
   - N should grow faster than D
   - Specifically: N ∝ C^0.73, D ∝ C^0.27
```

---

## Chinchilla (Hoffmann et al., 2022) -- Revised Scaling Laws

### The Chinchilla Correction

Chinchilla showed that Kaplan's recommendations were **wrong about the optimal N:D ratio**. Models were being trained with too many parameters and too few tokens.

```
Kaplan (2020) recommendation:
  "Scale model size faster than data"
  N ∝ C^0.73,  D ∝ C^0.27
  → For 10x more compute: 5.3x bigger model, 1.9x more data

Chinchilla (2022) correction:
  "Scale model size and data EQUALLY"
  N ∝ C^0.50,  D ∝ C^0.50
  → For 10x more compute: 3.2x bigger model, 3.2x more data

Rule of thumb: optimal training tokens ≈ 20 * N
```

### Chinchilla Formulas

```
Optimal model size given compute:
  N_opt ∝ C^0.50

Optimal dataset size given compute:
  D_opt ∝ C^0.50

Parametric loss:
  L(N, D) = E + A/N^alpha + B/D^beta

  where:
    E ≈ 1.69        (irreducible loss / entropy of natural language)
    A ≈ 406.4        alpha ≈ 0.34
    B ≈ 410.7        beta  ≈ 0.28

This predicts loss to within ~0.05 nats across the entire range tested.
```

### The Chinchilla Experiment

```
Setup: 400 models from 70M to 16B parameters
       Each trained with varying token counts
       Total compute budget per group held constant

Key result -- Gopher (280B) vs Chinchilla (70B):

  Model      Params    Tokens    Compute     Loss
  ─────────────────────────────────────────────────
  Gopher     280B      300B      5.0x10^23   ~2.10
  Chinchilla  70B     1.4T      5.0x10^23   ~1.94  ← BETTER!

Same compute, 4x fewer params, 4.7x more data → lower loss!

Gopher was severely under-trained:
  280B params should have been trained on 280B * 20 = 5.6T tokens
  But it only saw 300B tokens (18x too few!)
```

---

## Kaplan vs Chinchilla Comparison

| Aspect | Kaplan (2020) | Chinchilla (2022) |
|--------|---------------|-------------------|
| Optimal N:D ratio | N grows faster than D | N and D grow equally |
| N scaling with C | N ~ C^0.73 | N ~ C^0.50 |
| D scaling with C | D ~ C^0.27 | D ~ C^0.50 |
| Tokens per param | ~1.7 tokens/param | ~20 tokens/param |
| Implication | Build bigger models | Train longer on more data |
| Who was right? | Wrong on ratio | Correct (widely adopted) |
| Impact | GPT-3: 175B on 300B tokens | LLaMA: 7B on 1T tokens |

### Why Kaplan Got the Ratio Wrong

```
Kaplan's methodology:
  - Trained each model size to CONVERGENCE on fixed data
  - Measured "compute-efficient frontier"
  - But their models hit DATA bottleneck, not compute bottleneck
  - This biased results toward "more params help, data doesn't matter as much"

Chinchilla's correction:
  - Varied BOTH model size AND data size for fixed compute
  - Found true trade-off: both matter equally
  - Key insight: most models in 2020-2022 were under-trained
```

---

## Loss vs Compute Curves

```
Loss
(nats)
  │
3.0├─ ×                                 × = Under-trained (Kaplan regime)
  │    ×                                o = Chinchilla-optimal
  │      ×  ×
2.5├─       o  ×
  │          o   ×
  │            o    ×
2.0├─            o     ×
  │               o      ×
  │                 o       ×  ×  ×     ← Diminishing returns
1.8├─                  o         o
  │                      o         o
  │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ E ≈ 1.69 (irreducible loss)
1.5├─
  │
  └───┬───┬───┬───┬───┬───┬───┬───┬───►
   10^18 10^19 10^20 10^21 10^22 10^23 10^24
                    Compute (FLOPs)

The Chinchilla-optimal curve (o) always achieves lower loss
than under-trained models (×) at the same compute budget.


Model Size vs Data at Fixed Compute:

Loss
  │
  │  ×                    × = too few params, too much data
  │    ×
  │      ×
  │        ×
  │          o ← Chinchilla-optimal sweet spot
  │            ×
  │              ×
  │                ×      × = too many params, too little data
  │                  ×
  └───────────────────────────────────►
     Small model              Large model
     Many tokens              Few tokens
       ◄── Fixed compute budget ──►
```

---

## Compute-Optimal Model Sizes

### Chinchilla-Optimal Configurations

| Compute (FLOPs) | Optimal N (params) | Optimal D (tokens) | Equivalent GPU Hours (A100) |
|-----------------|--------------------|--------------------|---------------------------|
| 10^18 | 400M | 8.0B | ~10 |
| 10^19 | 1.3B | 25.3B | ~100 |
| 10^20 | 4.0B | 80.0B | ~1,000 |
| 10^21 | 12.7B | 253B | ~10,000 |
| 10^22 | 40.0B | 800B | ~100,000 |
| 10^23 | 127B | 2.5T | ~1,000,000 |
| 10^24 | 400B | 8.0T | ~10,000,000 |

```
How to use this table:
  1. Determine your compute budget (how many GPU hours you can afford)
  2. Find the corresponding row
  3. Train a model of size N_opt on D_opt tokens
  4. This minimizes loss for your budget

Example: You have 1000 A100-hours
  → ~10^20 FLOPs
  → Train a 4B model on 80B tokens
  → NOT a 13B model on 24B tokens (under-trained!)
  → NOT a 1B model on 320B tokens (over-trained, wasted compute)
```

### Relation: C = 6ND

```
The compute cost of training a Transformer:

C ≈ 6 * N * D  (FLOPs)

where:
  6 comes from: 2 (multiply-add) * 3 (forward + backward ≈ 3x forward)
  N = non-embedding parameters
  D = training tokens

Example: LLaMA-2 7B
  C = 6 * 6.7B * 2T = 8.04 x 10^22 FLOPs
  On 2048 A100 GPUs at 50% utilization:
  Time = 8.04 x 10^22 / (2048 * 312 TFLOPS * 0.5)
       ≈ 250,000 seconds ≈ 2.9 days

This matches Meta's reported training time for LLaMA-2 7B.
```

---

## Practical Implications

### What Chinchilla Means for Practice

```
Before Chinchilla (2020-2022):
  "Just make the model bigger"
  GPT-3: 175B params on 300B tokens (1.7 tokens/param)
  PaLM:  540B params on 780B tokens (1.4 tokens/param)
  → Models were 10-20x under-trained!

After Chinchilla (2023+):
  "Train longer on more data"
  LLaMA:   7B params on 1.0T tokens  (143 tokens/param)
  LLaMA-2: 7B params on 2.0T tokens  (286 tokens/param)
  LLaMA-3: 8B params on 15T+ tokens  (1875 tokens/param!)
  → Models now trained far BEYOND Chinchilla-optimal

Why go beyond Chinchilla-optimal?
  - Inference cost depends on N, not D
  - Training cost is one-time; inference cost is ongoing
  - A smaller model trained on more data has:
    - Same quality as a larger Chinchilla-optimal model
    - But cheaper inference (fewer params to serve)
  - This is called "inference-optimal" or "over-training"
```

### The Data Wall

```
Problem: Chinchilla says D_opt = 20 * N
  For a 1T param model: need 20T tokens
  For a 10T param model: need 200T tokens

  Estimated high-quality internet text: ~5-15T tokens
  We are approaching the data ceiling!

Solutions:
  1. Synthetic data (models generate training data)
  2. Multi-epoch training (repeat data, with some quality loss)
  3. Multimodal data (images, video, audio → more tokens)
  4. Code data (high quality, well-structured)
  5. Data quality > quantity (careful filtering and dedup)
  6. Curriculum learning (order data for maximum learning)

Scaling Data-Constrained LMs (Muennighoff et al., 2023):
  - Repeating data up to 4 epochs: minimal quality loss
  - Beyond 4 epochs: diminishing returns, possible memorization
  - Mixing in code data helps even for non-code tasks
```

---

## Emergent Abilities Debate

### The Claim (Wei et al., 2022)

Certain capabilities appear to **emerge** abruptly at a specific scale, rather than improving smoothly.

```
Performance on task X:

Random
chance ─── ─── ─── ─── ─── ─── ─── ──┐
  │                                     │   ← Abrupt jump!
  │                                     │
  │                                     ●───●───●
  │                                    /
  │     ●───●───●───●───●───●───●───●
  │
  └───┬───┬───┬───┬───┬───┬───┬───┬───►
   10M 100M  1B  10B 100B 500B
              Model size

Examples of claimed emergent abilities:
  - Chain-of-thought reasoning (~100B params)
  - Multi-digit arithmetic (~10B params)
  - Word unscrambling (~60B params)
  - Instruction following (~10B params)
```

### The Counter-Argument (Schaeffer et al., 2023)

```
Emergence might be an artifact of the METRIC, not the model.

Argument: If you use a discontinuous metric (e.g., exact match),
smooth improvement LOOKS like a sudden jump.

Exact match metric (discontinuous):
  "What is 2+2?" → "4"   = 1.0
  "What is 2+2?" → "4."  = 0.0  ← Close but scores 0!

Log-likelihood (continuous):
  Shows smooth improvement at all scales

Visualization:

Exact Match:                    Log-Likelihood:
  1.0│          ●●●●●           │         ●●●●●
     │         /                │       ●●
  0.5│        │                 │     ●●
     │        │                 │   ●●
  0.0│●●●●●●●                  │●●●
     └──────────────►           └──────────────►
       Model size                 Model size

  "Emergence"?                   Smooth improvement!
  Or metric artifact?            No discontinuity.

Key insight: The CAPABILITY improves smoothly.
The METRIC creates the illusion of a phase transition.
```

### Current Consensus

```
Both sides have valid points:

1. Smooth improvement in log-loss:
   - Cross-entropy loss improves smoothly with scale (always)
   - The underlying capability is gradually improving

2. Practical thresholds exist:
   - Some tasks have a minimum capability threshold
   - Below threshold: random-looking performance
   - Above threshold: useful performance
   - This IS practically meaningful even if not a "true" phase transition

3. Task complexity creates thresholds:
   - Multi-step reasoning needs all steps correct
   - P(all correct) = P(step)^n_steps
   - Even smooth improvement in P(step) creates sharp threshold
     for P(all correct) when n_steps is large

Example: 5-step reasoning, each step at probability p:
  p=0.5: P(all correct) = 0.5^5 = 3.1%
  p=0.8: P(all correct) = 0.8^5 = 32.8%
  p=0.9: P(all correct) = 0.9^5 = 59.0%
  p=0.95: P(all correct) = 0.95^5 = 77.4%

  Small improvement in p → large jump in task success!
```

---

## Beyond Chinchilla: Modern Scaling Considerations

### Inference-Optimal Scaling

```
Chinchilla-optimal:   minimize L(N, D) for fixed C_train
Inference-optimal:    minimize L(N, D) for fixed C_inference

Since C_inference ∝ N (active params per token):
  → Train smaller models on MORE data
  → Higher one-time training cost
  → Lower ongoing inference cost

This explains LLaMA-3 8B trained on 15T tokens:
  Chinchilla says: 8B model needs ~160B tokens
  LLaMA-3 used: 15T tokens (93x more!)
  Result: 8B model that matches much larger models
```

### Scaling Law Predictions

```
Before training a large model, run small-scale experiments:

1. Train 5-10 small models (100M - 1B params) at various D
2. Fit the scaling law: L(N, D) = E + A/N^alpha + B/D^beta
3. Extrapolate to predict large model performance
4. Decide whether to proceed with full training

Example workflow:
  Budget: 10^23 FLOPs (≈ $5M on cloud GPUs)
  Run: 10 experiments at 10^19 FLOPs each (≈ $500 total)
  Fit: scaling curves
  Predict: 127B model on 2.5T tokens → L ≈ 1.85
  Decision: proceed if predicted loss meets requirements
```

---

## Summary of Key Numbers

| Quantity | Value | Source |
|----------|-------|--------|
| Loss power-law exponent (N) | alpha ~ 0.34 | Chinchilla |
| Loss power-law exponent (D) | beta ~ 0.28 | Chinchilla |
| Irreducible loss (English) | E ~ 1.69 nats | Chinchilla |
| Chinchilla-optimal tokens/param | ~20 | Chinchilla |
| Modern practice tokens/param | 100-2000 | LLaMA-3, etc. |
| C = 6ND | Approximate training FLOPs | Standard |
| Safe data repetition | Up to 4 epochs | Muennighoff et al. |
| Internet text available | ~5-15T tokens | Various estimates |

---

## Related

- [Decoder-Only Architecture](01_Decoder_Only.md) -- the architecture being scaled
- [Model Families](02_Model_Family.md) -- how scaling laws influenced each family's design
- [Training](../training/00_Training.md) -- practical training considerations
