# Gated DeltaNet

> Parent: [Linear Attention](00_Linear_Attention.md)

## Overview

This file covers three models that progressively improve the linear attention update rule: **GLA** (data-dependent gating), **DeltaNet** (delta rule for overwriting), and **Gated DeltaNet** (combining both). Gated DeltaNet achieves the best quality among linear attention models and is used in production hybrid architectures like Qwen 3.5.

---

## GLA: Gated Linear Attention

### Motivation

Mamba-2 uses a **scalar** decay α_t — the same forgetting rate for all dimensions of the state matrix. GLA generalizes this to **data-dependent, per-dimension gating**:

```
Mamba-2:   S_t = α_t · S_{t-1} + v_t k_t^T          α_t ∈ ℝ (scalar)
GLA:       S_t = G_t ⊙ S_{t-1} + v_t k_t^T          G_t ∈ ℝ^{d_v × d_k} (matrix)

where G_t = α_t · 1 + (data-dependent component)
```

In practice, GLA parameterizes the gate as:

```
G_t = exp(-gate(x_t))    where gate: ℝ^d → ℝ^{d_v × d_k}

This gives element-wise control: each entry in the state matrix
can be decayed at a different rate depending on the input.
```

### Chunk-Wise Training with WY Representation

GLA can't use simple chunk-wise parallelism because the gate G_t varies per-step (unlike Mamba-2's scalar α). The solution is the **WY representation** — a matrix decomposition from numerical linear algebra:

```
Within a chunk of C steps, the cumulative effect of sequential updates:
  S_C = G_C ⊙ (G_{C-1} ⊙ (... ⊙ (G_1 ⊙ S_0 + v_1 k_1^T) ...) + v_C k_C^T)

Can be decomposed as:
  S_C = Γ_C ⊙ S_0 + W · Y^T

where:
  Γ_C  = cumulative gate (product of all G_t in the chunk)
  W, Y = matrices that capture the net effect of all updates
         W ∈ ℝ^{d_v × C}, Y ∈ ℝ^{d_k × C}

This allows:
  1. Computing all intra-chunk outputs in parallel (quadratic in C)
  2. Passing only S_C to the next chunk (linear across chunks)
```

**What WY representation is**: In numerical linear algebra, the WY representation expresses a product of Householder reflectors as `I + WY^T`. GLA adapts this idea: instead of applying C sequential state updates one by one, you decompose their cumulative effect into a low-rank update `WY^T` plus a scaled copy of the initial state. This turns an inherently sequential computation into a parallelizable one.

---

## DeltaNet: The Delta Rule

### Motivation

All prior linear attention variants (including GLA) use **additive** updates:

```
Additive:   S_t = (decay) · S_{t-1} + v_t k_t^T

Problem: If key k_t was previously associated with value v_old, adding v_t k_t^T
doesn't remove the old association. The state accumulates both:
  S contains: ... + v_old k_t^T + ... + v_t k_t^T + ...

This is like a hash table that can INSERT but never UPDATE.
```

### The Delta Rule: One Step of SGD

DeltaNet replaces the additive update with the **delta rule** — a single step of online gradient descent on the hidden state:

```
Goal: Make S_t satisfy S_t k_t = v_t  (retrieve v_t when queried with k_t)

Objective: minimize ||S k_t - v_t||²

One step of gradient descent on S:
  error_t = v_t - S_{t-1} k_t                    ← retrieval error
  S_t = S_{t-1} + β_t · error_t · k_t^T           ← gradient step

Expanding:
  S_t = S_{t-1} + β_t (v_t - S_{t-1} k_t) k_t^T
      = S_{t-1} + β_t v_t k_t^T - β_t S_{t-1} k_t k_t^T
      = (I - β_t k_t k_t^T) S_{t-1} + β_t v_t k_t^T

where β_t ∈ (0, 1] is a learnable step size
```

### Why the Delta Rule Works Better

```
Linear Attention (additive):          DeltaNet (delta rule):
  Store "Paris" at "France"            Store "Paris" at "France"
  S += v_paris k_france^T             S += β(v_paris - S·k_france) k_france^T

  Later: store "Berlin" at "France"   Later: store "Berlin" at "France"
  S += v_berlin k_france^T            error = v_berlin - S·k_france
                                      S += β · error · k_france^T

  Query "France":                     Query "France":
  S k_france = v_paris + v_berlin     S k_france ≈ v_berlin  ✓
  (contaminated!)                     (old association overwritten!)
```

The delta rule enables **in-context retrieval and correction** — the model can update its stored associations, which is crucial for tasks like in-context learning and factual recall.

### Online Learning Interpretation

Each model in the family corresponds to a different online learning objective:

```
Model             Objective at step t                    Update
─────             ──────────────────                    ──────
Linear Attention  (none — just accumulate)              S += v_t k_t^T
GLA               (none — accumulate with decay)        S = G_t ⊙ S + v_t k_t^T
DeltaNet          min ||S k_t - v_t||²                  S += β(v - Sk)k^T
Gated DeltaNet    min ||S k_t - v_t||² + decay          S = α(I - βkk^T)S + βvk^T
```

---

## Gated DeltaNet: The Best of Both Worlds

### Combining Mamba-2's Gating with DeltaNet's Delta Rule

Gated DeltaNet (Yang, Kautz & Hatamizadeh, ICLR 2025) combines:
- **α_t** (scalar decay from Mamba-2) — forget irrelevant old state
- **β_t** (delta rule from DeltaNet) — overwrite stale associations

```
S_t = α_t (I - β_t k_t k_t^T) S_{t-1} + β_t v_t k_t^T
      ▲     ▲                ▲            ▲
      │     │                │            └── write new association
      │     │                └── correct old retrieval error
      │     └── delta rule factor
      └── global decay (forget)

Equivalently:
  S_t = α_t S_{t-1}                        ← decay old state
       - α_t β_t (S_{t-1} k_t) k_t^T      ← erase old association at k_t
       + β_t v_t k_t^T                      ← write new association at k_t
```

### Table 1: All Models as Special Cases

| Model | α_t | β_t | Update Rule |
|-------|-----|-----|-------------|
| Linear Attention | 1 | 0 | `S_t = S_{t-1} + v_t k_t^T` |
| RetNet | γ (fixed scalar) | 0 | `S_t = γ S_{t-1} + v_t k_t^T` |
| Mamba-2 / GLA | α_t (data-dep) | 0 | `S_t = α_t S_{t-1} + v_t k_t^T` |
| DeltaNet | 1 | β_t | `S_t = (I - β_t k_t k_t^T)S_{t-1} + β_t v_t k_t^T` |
| **Gated DeltaNet** | α_t | β_t | `S_t = α_t(I - β_t k_t k_t^T)S_{t-1} + β_t v_t k_t^T` |

### Chunk-Wise Training

Gated DeltaNet extends GLA's WY-based chunk-wise algorithm to handle the delta rule term:

```
Within chunk (C steps):
  1. Compute the cumulative effect of α_t and (I - β_t k_t k_t^T) updates
  2. Use a modified WY decomposition that accounts for the delta rule
  3. This gives quadratic-in-C computation, fully parallel

Across chunks:
  4. Pass state S_chunk to next chunk (linear)

The delta rule term (I - β_t k_t k_t^T) makes the WY decomposition
more complex than GLA but still tractable.
```

---

## Evolution of the Update Rule

```
Step 1: Linear Attention (2020)
        S_t = S_{t-1} + v_t k_t^T
        Problem: no forgetting, no overwriting
                    │
                    ▼
Step 2: Add decay (RWKV 2023, RetNet 2023, Mamba-2 2024)
        S_t = α_t · S_{t-1} + v_t k_t^T
        ✓ Forgetting    ✗ Still can't overwrite specific entries
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
Step 3a: Rich gating (GLA)    Step 3b: Delta rule (DeltaNet)
  S_t = G_t ⊙ S + v k^T       S_t = (I - β kk^T) S + β v k^T
  ✓ Per-dim forgetting         ✓ Can overwrite associations
  ✗ Still additive             ✗ No global decay
        │                       │
        └───────────┬───────────┘
                    ▼
Step 4: Gated DeltaNet (ICLR 2025)
        S_t = α_t (I - β_t k_t k_t^T) S_{t-1} + β_t v_t k_t^T
        ✓ Global decay (α)
        ✓ Overwriting (β + delta rule)
        ✓ Best quality among linear attention models
```

---

## FLA: Flash Linear Attention Library

The [FLA library](https://github.com/fla-org/flash-linear-attention) provides optimized Triton kernels for the entire linear attention family:

- GLA, DeltaNet, Gated DeltaNet, RetNet, RWKV, Mamba-2
- Chunk-wise parallel training kernels
- Efficient recurrent inference kernels
- Drop-in replacement for standard attention in Transformer architectures

FLA is the practical backbone that makes training these models feasible — similar to how FlashAttention made long-context softmax attention practical.

---

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [GLA (Yang et al., 2024)](https://arxiv.org/abs/2312.06635) | Data-dependent gating, WY chunk-wise training |
| [DeltaNet (Yang et al., 2024)](https://arxiv.org/abs/2406.06484) | Delta rule for associative memory overwriting |
| [Gated DeltaNet (Yang, Kautz & Hatamizadeh, 2025)](https://arxiv.org/abs/2412.06464) | Unified gating + delta rule, ICLR 2025 |
| [FLA Library](https://github.com/fla-org/flash-linear-attention) | Triton kernels for linear attention family |

## Related

- [Mamba](03_Mamba.md) — Mamba-2's SSD framework that Gated DeltaNet extends
- [Linear Attention Basics](01_Linear_Attention_Basics.md) — The foundation: associative memory and dual forms
- [Hybrid Architecture](05_Hybrid_Architecture.md) — Using Gated DeltaNet in production models
