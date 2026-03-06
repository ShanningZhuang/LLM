# Mamba

> Parent: [Linear Attention](00_Linear_Attention.md)

## Overview

Mamba transforms SSMs from fixed linear systems into **content-aware** sequence models. Mamba-1 introduces selective state spaces where B, C, and Δ depend on the input. Mamba-2 reveals the deep connection between SSMs, linear attention, and structured matrices through the **State Space Duality (SSD)** framework, enabling much more efficient training.

---

## Mamba-1: Selective State Spaces

### The Problem with Fixed SSMs

S4's parameters (A, B, C, Δ) are the same for every input token. This means:

```
Fixed SSM:   x_k = Ā x_{k-1} + B̄ u_k     ← same Ā, B̄ for all k
             Every token gets the same treatment
             Can't select what to remember vs. forget

Example: "The capital of France, which was founded by ... , is [MASK]"
  A fixed SSM treats every token equally — can't selectively
  retain "France" while filtering filler words.
```

### Selective SSM: Make Parameters Input-Dependent

Mamba makes B, C, and Δ (step size) functions of the input:

```
u_k (input)
  │
  ├──→ B_k = Linear(u_k)        ← input-dependent input projection
  ├──→ C_k = Linear(u_k)        ← input-dependent output projection
  └──→ Δ_k = softplus(Linear(u_k))  ← input-dependent step size

A remains fixed (learned diagonal)

Discretize with input-dependent Δ_k:
  Ā_k = exp(Δ_k · A)            ← NOW input-dependent!
  B̄_k = Δ_k · B_k

Recurrence:
  x_k = Ā_k x_{k-1} + B̄_k u_k
  y_k = C_k x_k
```

**Why Δ matters**: Δ_k controls how much to "let in" at each step. Large Δ → strong update (attend to this token). Small Δ → Ā_k ≈ I → mostly copy previous state (ignore this token). This gives content-aware **selection**.

### Hardware-Aware Selective Scan

Making parameters input-dependent breaks the convolution mode (no longer a fixed kernel). Mamba uses a **parallel scan** algorithm on GPU:

```
Parallel scan for: x_k = a_k x_{k-1} + b_k

Step 1 (pairs):     [x₁] [x₂] [x₃] [x₄] [x₅] [x₆] [x₇] [x₈]
                      ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓
Step 2 (combine):   [x₁] [x₁₋₂] [x₃] [x₃₋₄] [x₅] [x₅₋₆] [x₇] [x₇₋₈]
                             ↓              ↓              ↓              ↓
Step 3 (combine):   [x₁] [x₁₋₂] [x₃] [x₁₋₄] [x₅] [x₅₋₆] [x₇] [x₅₋₈]
                                          ↓                              ↓
Step 4 (combine):   [x₁] [x₁₋₂] [x₃] [x₁₋₄] [x₅] [x₅₋₆] [x₇] [x₁₋₈]

O(log n) depth, O(n) work total
```

Mamba further optimizes by keeping the scan in GPU SRAM (similar philosophy to FlashAttention): load inputs from HBM, compute the full scan in SRAM, write only the outputs back.

### Mamba-1 Architecture

```
┌─────────────────────────────────┐
│          Mamba Block             │
│                                 │
│  x ──→ Linear ──→ Conv1d ──→ SiLU ──→ SSM ──┐
│  │                                            │
│  └──→ Linear ──→ SiLU ──────────────────→ × ──→ Linear ──→ out
│                                (gate)         │
└─────────────────────────────────┘

No attention at all — the entire model is:
  Embedding → [Mamba Block × N] → LM Head

Key differences from Transformer:
  - No attention mechanism
  - No positional encoding (position is implicit in recurrence)
  - Conv1d provides local context (like a small sliding window)
```

---

## Mamba-2: State Space Duality (SSD)

### The Core Insight

Mamba-2 proves that **SSMs with scalar-valued decay are equivalent to a form of linear attention with structured masking**.

```
SSM recurrence:                     Structured attention:
  x_t = α_t x_{t-1} + B_t u_t      y = (M ⊙ L) · u

  where α_t is scalar              where L_ij = C_i B_j · Π_{k=j+1}^{i} α_k
                                          M is a causal mask

SAME COMPUTATION, two forms:

  Recurrent:  O(Nd) per step           ← good for inference
  Quadratic:  O(n²d) total             ← but parallelizable for training!
```

This duality means we can:
- **Train** using the quadratic (attention-like) form — fully parallel
- **Infer** using the recurrent form — O(1) per token

### The Unified Recurrence

Mamba-2 expresses the SSM with a **matrix-valued hidden state**, bridging to linear attention:

```
S_t = α_t · S_{t-1} + v_t k_t^T

where:
  S_t ∈ ℝ^{d_v × d_k}   — matrix-valued state (= linear attention's state!)
  k_t = B_t^T x_t        — "key" (from SSM's input projection)
  v_t = output            — "value"
  α_t ∈ ℝ (scalar)       — decay factor (from discretized A)

Output: o_t = S_t q_t     where q_t = C_t  (SSM's output projection)
```

This is exactly linear attention's recurrence `S_t = S_{t-1} + v_t k_t^T` with an added scalar decay α_t.

### Chunk-Wise Parallel Algorithm

The key training algorithm: **quadratic within chunks, linear across chunks**.

```
Sequence: [──── chunk 1 ────][──── chunk 2 ────][──── chunk 3 ────]
           ◄── C tokens ──►

Within each chunk:
  Use the quadratic (attention-like) form
  Compute L_ij = C_i B_j · Π α_k    ← small C×C "attention matrix"
  This is fully parallel (like FlashAttention)

Across chunks:
  Pass the state S_{chunk} from one chunk to the next
  S_{chunk+1} = α^C · S_{chunk} + (contribution from new chunk)
  This is sequential but only n/C steps

Total: O(C · n · d) where C is chunk size (typically 64-256)
  ≈ O(n · d) with good constants
```

```
                     Chunk-wise processing:

Chunk 1:    [t₁ t₂ ... t_C]  →  quadratic within  →  S₁ (state after chunk 1)
                                                        │
Chunk 2:    [t_{C+1} ... t_{2C}]  + S₁  →  quadratic  →  S₂
                                                           │
Chunk 3:    [t_{2C+1} ... t_{3C}]  + S₂  →  quadratic  →  S₃
                                                           ...
```

### Multi-Head Structure

Mamba-2 introduces multi-head SSMs analogous to multi-head attention:

```
Multi-head attention:              Multi-head SSM (SSD):
  Q, K, V per head                   Q (= C), K (= B), V per head
  Each head: d_k, d_k, d_v           Each head: d_k, d_k, d_v
  Output: concat + linear            Output: concat + linear

The head dimension determines state size:
  State per head: d_v × d_k
  Total state: n_heads × d_v × d_k
```

---

## Duality Diagram

```
            ┌──────────────────────────────────────────────────┐
            │              Computation Modes                    │
            │                                                  │
            │   Recurrence          Convolution*        Matrix │
            │   x = Āx + B̄u        y = K * u         y = Lu   │
            │   O(Nd)/step         O(n log n)          O(n²d)  │
            │                                                  │
            │   * Only for fixed parameters (S4, not Mamba)    │
            └──────────────────────────────────────────────────┘
                        │                               │
                        │    Mamba-2 SSD proves:       │
                        │    SSM (scalar α) ═══════    │
                        │    Structured Linear Attn    │
                        │                               │
            ┌───────────▼───────────────────────────────▼──────┐
            │                                                  │
            │   SSM with scalar decay    ←→    Linear Attention│
            │   x_t = α_t x_{t-1} + b_t u_t   S_t = α S + vk^T│
            │                                                  │
            │   State space perspective    Attention perspective│
            │   (control theory)           (key-value memory)  │
            │                                                  │
            └──────────────────────────────────────────────────┘
```

---

## Mamba-1 vs Mamba-2

| Aspect | Mamba-1 | Mamba-2 |
|--------|---------|---------|
| State | Vector x ∈ ℝ^N | Matrix S ∈ ℝ^(d_v × d_k) |
| Decay | Diagonal (per-dim) | Scalar (per-head) |
| Training | Parallel scan | Chunk-wise SSD |
| Speed (training) | Fast | **8× faster** |
| Multi-head | No | Yes |
| Duality | Not explicit | SSM = linear attention |
| State size | N per channel | d_v × d_k per head |

The scalar decay restriction in Mamba-2 is the key trade-off: it loses per-dimension control but gains the duality that enables much faster chunk-wise training.

---

## Key Takeaways

1. **Selection is the key innovation**: making SSM parameters input-dependent enables content-aware processing — the model can choose what to remember and forget
2. **SSD duality**: SSMs with scalar decay = structured linear attention. This unifies two research communities
3. **Chunk-wise training**: quadratic within chunks (parallel), linear across chunks (sequential) — the dominant training paradigm for all subsequent models
4. **Matrix-valued state**: Mamba-2's `S_t ∈ ℝ^{d_v × d_k}` is the same as linear attention's key-value memory — this is the shared abstraction

---

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Mamba (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752) | Selective SSM, hardware-aware scan |
| [Mamba-2 / SSD (Dao & Gu, 2024)](https://arxiv.org/abs/2405.21060) | State Space Duality, chunk-wise training |

## Related

- [State Space Models](02_State_Space_Models.md) — The S4 foundation Mamba builds on
- [Gated DeltaNet](04_Gated_DeltaNet.md) — Extends Mamba-2's gating with delta rule
- [Hybrid Architecture](05_Hybrid_Architecture.md) — Using Mamba/linear attention layers alongside softmax
