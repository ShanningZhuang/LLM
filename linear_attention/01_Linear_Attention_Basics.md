# Linear Attention Basics

> Parent: [Linear Attention](00_Linear_Attention.md)

## Overview

Standard attention computes `softmax(QK^T)V` in O(n^2 d) time. **Linear attention** replaces the softmax with a decomposable kernel, enabling an equivalent recurrent form that runs in O(nd^2) time — linear in sequence length. This is the foundation for all subsequent models in this family.

---

## From Softmax to Linear

### Standard Attention

```
For query q_i, the output is:

            Σ_j exp(q_i^T k_j) v_j
  o_i  =  ─────────────────────────
            Σ_j exp(q_i^T k_j)

The exp(q_i^T k_j) couples q_i and k_j through softmax → must compute all n² pairs.
```

### The Kernel Trick (Katharopoulos et al., 2020)

Replace `exp(q^T k)` with `φ(q)^T φ(k)` for some feature map φ:

```
            Σ_j φ(q_i)^T φ(k_j) v_j       φ(q_i)^T  Σ_j φ(k_j) v_j^T
  o_i  =  ──────────────────────────  =   ──────────────────────────────
            Σ_j φ(q_i)^T φ(k_j)            φ(q_i)^T  Σ_j φ(k_j)

                    φ(q_i)^T  S
        =          ──────────────        where S = Σ_j φ(k_j) v_j^T
                    φ(q_i)^T  z                  z = Σ_j φ(k_j)
```

**Key insight**: S and z are cumulative sums — they don't depend on i. We compute them once, then each o_i is just a matrix-vector product.

### Dual Forms

```
Parallel form (training):          Recurrent form (inference):
─────────────────────              ─────────────────────────
O = φ(Q) · (φ(K)^T V)             S_t = S_{t-1} + φ(k_t) v_t^T
                                   z_t = z_{t-1} + φ(k_t)
Time: O(n d²)                     o_t = S_t φ(q_t) / (z_t^T φ(q_t))
      (matrix multiply,
       right-to-left)              Time: O(d²) per step → O(n d²) total
                                   Memory: O(d²) constant state!
```

This duality is fundamental: same computation, two execution strategies. The parallel form is efficient for training (GPU-friendly), while the recurrent form enables O(1) per-token inference.

---

## The Hidden State as Associative Memory

The state `S_t = Σ_{j≤t} φ(k_j) v_j^T` is a **key-value associative memory**:

```
Writing: S_t = S_{t-1} + φ(k_t) v_t^T     ← "store v_t at address k_t"
Reading: o_t = S_t φ(q_t)                   ← "retrieve value at address q_t"

S_t ∈ ℝ^{d_v × d_k} — a matrix that stores ALL key-value pairs seen so far
```

This is why the recurrent state size is d_v × d_k (typically 64×64 = 4096 per head), independent of sequence length. Compare with softmax attention's KV cache which grows as O(n × d).

### The Forgetting Problem

Vanilla linear attention **never forgets**: S_t only accumulates, so early tokens have equal weight to recent ones. This causes:

1. **Memory dilution** — as S_t accumulates more entries, each retrieval becomes noisier
2. **No overwriting** — can't update the value stored at a key; old associations persist
3. **Quality gap** — softmax attention dynamically reweights, linear attention doesn't

This limitation motivates every subsequent model: RWKV adds decay, Mamba adds selective gating, DeltaNet adds overwriting.

---

## RWKV: Channel-Wise Decay

RWKV (Receptance Weighted Key Value) introduces **exponential decay** to the recurrence:

```
Standard linear attention:    S_t = S_{t-1} + k_t v_t^T
RWKV-style:                   S_t = Λ ⊙ S_{t-1} + k_t v_t^T

where Λ is a diagonal matrix of channel-wise decay rates (learned, fixed after training)
```

This gives recent tokens more influence than distant ones — a simple but effective form of forgetting. RWKV also introduces:

- **Receptance gate** (r): controls how much of the retrieved value to use
- **WKV operator**: efficiently computes the weighted key-value aggregation
- **Time-mixing and channel-mixing** blocks that parallel attention and FFN

RWKV achieves competitive perplexity with pure Transformers while maintaining O(1) per-token inference cost, proving that linear-time models can match softmax quality at scale.

---

## RetNet: Retention Mechanism

RetNet (Retentive Network) proposes **retention** as a dual of attention:

```
Retention:  S_t = γ · S_{t-1} + k_t v_t^T      (γ = scalar decay, typically per-head)

Three computation modes:
1. Parallel:    O = (Q K^T ⊙ D) V      where D is a causal decay mask
2. Recurrent:   S_t = γ S_{t-1} + k_t v_t^T;  o_t = q_t^T S_t
3. Chunk-wise:  Parallel within chunks, recurrent across chunks
```

The chunk-wise mode is particularly important: it enables GPU-efficient training (parallel within fixed-size chunks) while maintaining the linear recurrence across chunks. This pattern recurs in GLA, Mamba-2, and Gated DeltaNet.

---

## Feature Maps

Different feature maps φ have been proposed:

| Feature Map | Formula | Notes |
|-------------|---------|-------|
| ELU + 1 (Katharopoulos) | φ(x) = ELU(x) + 1 | Simple, ensures non-negativity |
| Random Fourier (Performers) | φ(x) = exp(ωx - \|\|x\|\|²/2) | Approximates softmax kernel |
| Identity (no map) | φ(x) = x | Used when gating handles non-linearity |
| Exponential (RWKV) | Built into WKV operator | Implicitly applies exp decay |

In practice, modern linear attention models (GLA, DeltaNet, Gated DeltaNet) often use simple feature maps or none at all, relying on gating and the delta rule for expressivity instead.

---

## Simple Linear Attention in PyTorch

```python
import torch
import torch.nn.functional as F

def linear_attention(q, k, v):
    """
    Linear attention with ELU+1 feature map.

    Args:
        q, k, v: (batch, heads, seq_len, d_k)
    Returns:
        output: (batch, heads, seq_len, d_k)
    """
    # Feature map: ELU(x) + 1 ensures non-negative "attention weights"
    q = F.elu(q) + 1
    k = F.elu(k) + 1

    # Parallel form: O = φ(Q) · (φ(K)^T · V)
    # Right-to-left multiplication: first compute K^T V → (d_k, d_k)
    kv = torch.einsum('bhnd,bhne->bhde', k, v)       # (B, H, d_k, d_v)

    # Then multiply Q by the result
    out = torch.einsum('bhnd,bhde->bhne', q, kv)      # (B, H, N, d_v)

    # Normalize (denominator of attention)
    z = torch.einsum('bhnd,bhnd->bhn', q, k.cumsum(dim=2))  # simplified
    out = out / (z.unsqueeze(-1) + 1e-6)

    return out

def linear_attention_recurrent(q, k, v):
    """
    Recurrent form — O(d²) per step, for autoregressive inference.
    """
    q = F.elu(q) + 1
    k = F.elu(k) + 1

    B, H, N, D = q.shape
    S = torch.zeros(B, H, D, D, device=q.device)     # hidden state
    z = torch.zeros(B, H, D, device=q.device)          # normalizer
    outputs = []

    for t in range(N):
        k_t = k[:, :, t]      # (B, H, D)
        v_t = v[:, :, t]      # (B, H, D)
        q_t = q[:, :, t]      # (B, H, D)

        # Update state: S_t = S_{t-1} + k_t v_t^T
        S = S + torch.einsum('bhd,bhe->bhde', k_t, v_t)
        z = z + k_t

        # Read: o_t = S_t q_t / (z_t^T q_t)
        o_t = torch.einsum('bhde,bhd->bhe', S, q_t)
        denom = torch.einsum('bhd,bhd->bh', z, q_t).unsqueeze(-1)
        o_t = o_t / (denom + 1e-6)

        outputs.append(o_t)

    return torch.stack(outputs, dim=2)
```

---

## Key Takeaways

1. **Linear attention = associative memory**: the hidden state stores key-value pairs in a matrix
2. **Dual forms**: parallel (O(nd²), training) and recurrent (O(d²) per step, inference)
3. **The forgetting problem** is the central limitation: vanilla linear attention can't forget or overwrite
4. **Every subsequent model** in this family adds a mechanism for forgetting: decay (RWKV/RetNet), selectivity (Mamba), gating (GLA), or overwriting (DeltaNet)

---

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Transformers are RNNs (Katharopoulos et al., 2020)](https://arxiv.org/abs/2006.16236) | Linear attention, kernel trick, dual forms |
| [Random Feature Attention — Performers (Choromanski et al., 2021)](https://arxiv.org/abs/2009.14794) | Random Fourier features for softmax approximation |
| [RWKV (Peng et al., 2023)](https://arxiv.org/abs/2305.13048) | Channel-wise decay, competitive at scale |
| [RetNet (Sun et al., 2023)](https://arxiv.org/abs/2307.08621) | Retention, chunk-wise computation |

## Related

- [State Space Models](02_State_Space_Models.md) — SSMs as structured linear attention
- [Gated DeltaNet](04_Gated_DeltaNet.md) — Modern solutions to the forgetting problem
- [Efficient Attention](../attention/05_Efficient_Attention.md) — FlashAttention (complementary approach)
