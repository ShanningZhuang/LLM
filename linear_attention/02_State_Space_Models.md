# State Space Models

> Parent: [Linear Attention](00_Linear_Attention.md)

## Overview

State Space Models (SSMs) provide a principled framework for sequence modeling rooted in control theory. The key innovation of **S4** (Structured State Spaces for Sequence Modeling) is showing that continuous-time linear systems with carefully designed state matrices can model extremely long-range dependencies while being trainable as convolutions and deployable as recurrences.

---

## Continuous-Time State Space

An SSM maps an input signal u(t) to an output y(t) through a hidden state x(t):

```
State equation:     x'(t) = A x(t) + B u(t)
Output equation:    y(t)  = C x(t) + D u(t)

where:
  x(t) ∈ ℝ^N     — hidden state (N = state dimension, e.g., 64)
  u(t) ∈ ℝ       — input (scalar, one channel)
  y(t) ∈ ℝ       — output (scalar)
  A ∈ ℝ^{N×N}    — state transition matrix (the key design choice)
  B ∈ ℝ^{N×1}    — input projection
  C ∈ ℝ^{1×N}    — output projection
  D ∈ ℝ         — skip connection (often omitted)
```

This is a **linear** dynamical system — the same family as linear attention's recurrence, but formulated in continuous time. The continuous formulation provides principled initialization through HiPPO and enables discretization for any sampling rate.

---

## Discretization

For discrete sequences (like tokens), we discretize with step size Δ:

### Zero-Order Hold (ZOH)

Assumes input is constant between time steps:

```
Ā = exp(Δ A)                    — matrix exponential
B̄ = (Δ A)^{-1} (exp(Δ A) - I) · Δ B    — simplified: B̄ ≈ Δ B for small Δ

Discrete recurrence:
  x_k = Ā x_{k-1} + B̄ u_k
  y_k = C x_k
```

### Bilinear (Tustin) Transform

```
Ā = (I - Δ/2 · A)^{-1} (I + Δ/2 · A)
B̄ = (I - Δ/2 · A)^{-1} · Δ B
```

The choice of discretization affects numerical stability and frequency response, but both yield a linear recurrence that can be unrolled as either a recurrence or a convolution.

---

## Convolutional View (Parallel Training)

The discrete recurrence can be unrolled:

```
x_0 = B̄ u_0
x_1 = Ā B̄ u_0 + B̄ u_1
x_2 = Ā² B̄ u_0 + Ā B̄ u_1 + B̄ u_2
...

y_k = C x_k = Σ_{j=0}^{k} C Ā^{k-j} B̄ · u_j

This is a causal convolution with kernel:
  K = (C B̄,  C Ā B̄,  C Ā² B̄,  ...,  C Ā^{L-1} B̄)

y = K * u     (1D convolution, computed via FFT in O(n log n))
```

```
Three equivalent computation modes:

                Recurrence               Convolution              "Attention"
              ┌────────────┐          ┌──────────────┐         ┌────────────────┐
              │ x_k = Āx + B̄u│       │ y = K * u    │         │ y = (M ⊙ L) u  │
              │ y_k = Cx_k  │        │ via FFT      │         │ (masked linear)│
              └────────────┘          └──────────────┘         └────────────────┘
  Inference:    O(N) per step           O(n log n)               O(n²)
  Training:     O(nN) sequential        O(n log n) ✓             O(n²)
  Advantage:    Constant memory         Parallelizable            Interpretable
```

---

## HiPPO: The Initialization That Matters

The state matrix A is the most important design choice. Random initialization fails for long sequences. **HiPPO** (High-order Polynomial Projection Operators) initializes A so that the hidden state x_t optimally approximates the history of inputs.

### HiPPO-LegS (Legendre, Scaled)

```
A_nk = -(2n + 1)^{1/2} (2k + 1)^{1/2}    if n > k
        n + 1                                if n = k
        0                                    if n < k

This makes x_t encode the coefficients of a Legendre polynomial
approximation of the input history u(s) for s ≤ t.
```

**Intuition**: The hidden state stores a compressed "summary" of the entire input history, with a mathematically optimal approximation. This is why S4 handles sequences of length 16,000+ where vanilla RNNs and Transformers struggle.

---

## S4: Structured State Spaces

S4 (Gu et al., 2022) makes the SSM framework practical:

### Key Challenges Solved

1. **Computing Ā^k efficiently**: The convolution kernel requires powers of Ā. S4 uses the NPLR (Normal Plus Low-Rank) structure of HiPPO to compute this in O(N log N) per kernel element.

2. **Long convolution**: Kernel length = sequence length. S4 computes the full kernel, then uses FFT for O(n log n) training.

3. **Stability**: HiPPO initialization ensures the eigenvalues of A have negative real parts → stable dynamics.

### S4 Architecture

```
Input u ∈ ℝ^{n × d}
     │
     ▼  (apply SSM independently to each of d channels)
┌─────────────────────┐
│  SSM (per channel)  │
│  x_k = Ā x + B̄ u_k │
│  y_k = C x_k        │
└─────────────────────┘
     │
     ▼
  LayerNorm + nonlinearity + linear mixing across channels
     │
     ▼
Output y ∈ ℝ^{n × d}

Note: The SSM itself is LINEAR — nonlinearity comes from the
surrounding architecture (similar to how attention is linear
in V, with softmax providing nonlinearity).
```

---

## Diagonal SSMs (S4D, DSS)

A major simplification: restrict A to be **diagonal**.

```
Full A ∈ ℝ^{N×N}:     x_k = Ā x_{k-1} + B̄ u_k      — N² parameters, coupled
Diagonal A ∈ ℝ^N:     x_k^i = ā_i x_{k-1}^i + b̄_i u_k  — N independent 1D recurrences

Convolution kernel:    K_j = Σ_i  c_i · (ā_i)^j · b̄_i
                       = sum of geometric series (easy to compute!)
```

S4D shows that diagonal A with complex-valued eigenvalues captures most of S4's expressivity with much simpler implementation. Each state dimension is an independent exponential decay with a complex phase.

### Connection to Linear Attention

With diagonal A, the SSM recurrence becomes:

```
x_k^i = ā_i · x_{k-1}^i + b̄_i · u_k

Compare with linear attention + decay:
  s_k = α_k · s_{k-1} + k_k · v_k

SSM = linear attention where:
  - α (decay) comes from the diagonal of Ā (structured, from HiPPO)
  - The "key-value" outer product is replaced by scalar input × projection
  - State dimension N replaces d_k × d_v
```

This connection is made precise in Mamba-2's SSD framework.

---

## Key Takeaways

1. **Continuous→discrete**: SSMs start in continuous time, then discretize. This gives principled initialization (HiPPO) and separates the model from the sampling rate.

2. **Three modes**: Recurrence (O(N) per step inference), convolution (O(n log n) training), and attention-like (O(n²) but interpretable). Same model, different computation.

3. **HiPPO matters**: The initialization of A is what makes SSMs work for long-range dependencies. Random A fails.

4. **Diagonal SSMs** are nearly as good as full SSMs and much simpler — this insight leads directly to Mamba.

5. **Limitation**: All parameters (A, B, C) are **fixed** (input-independent). The model can't selectively attend to or ignore specific tokens. Mamba fixes this.

---

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [HiPPO (Gu et al., 2020)](https://arxiv.org/abs/2008.07669) | Optimal polynomial projection for memory |
| [S4 (Gu et al., 2022)](https://arxiv.org/abs/2111.00396) | Structured state spaces, NPLR, long-range |
| [S4D (Gu et al., 2022)](https://arxiv.org/abs/2206.11893) | Diagonal simplification |
| [DSS (Gupta et al., 2022)](https://arxiv.org/abs/2203.14343) | Diagonal state spaces |

## Related

- [Linear Attention Basics](01_Linear_Attention_Basics.md) — Linear attention as the general framework SSMs fit into
- [Mamba](03_Mamba.md) — Making SSMs input-dependent (selective)
