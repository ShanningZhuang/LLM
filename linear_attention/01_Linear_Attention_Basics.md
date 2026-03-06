# Linear Attention Basics

> Parent: [Linear Attention](00_Linear_Attention.md)

## Overview

Standard attention computes `softmax(QK^T)V` in O(n^2 d) time. **Linear attention** replaces the softmax with a decomposable kernel, enabling an equivalent recurrent form that runs in O(nd^2) time — linear in sequence length. This is the foundation for all subsequent models in this family.

---

## From Softmax to Linear

### Standard Attention

For query \( \mathbf{q}_i \), the output is:

$$
\mathbf{o}_i = \frac{\sum_j \exp(\mathbf{q}_i^\top \mathbf{k}_j) \, \mathbf{v}_j}{\sum_j \exp(\mathbf{q}_i^\top \mathbf{k}_j)}
$$

The \( \exp(\mathbf{q}_i^\top \mathbf{k}_j) \) couples \( \mathbf{q}_i \) and \( \mathbf{k}_j \) through softmax — must compute all \( n^2 \) pairs.

### The Kernel Trick (Katharopoulos et al., 2020)

Replace \( \exp(\mathbf{q}^\top \mathbf{k}) \) with \( \phi(\mathbf{q})^\top \phi(\mathbf{k}) \) for some feature map \( \phi \):

$$
\mathbf{o}_i = \frac{\sum_j \phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j) \, \mathbf{v}_j}{\sum_j \phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j)} = \frac{\phi(\mathbf{q}_i)^\top \sum_j \phi(\mathbf{k}_j) \mathbf{v}_j^\top}{\phi(\mathbf{q}_i)^\top \sum_j \phi(\mathbf{k}_j)} = \frac{\phi(\mathbf{q}_i)^\top S}{\phi(\mathbf{q}_i)^\top \mathbf{z}}
$$

where \( S = \sum_j \phi(\mathbf{k}_j) \mathbf{v}_j^\top \) and \( \mathbf{z} = \sum_j \phi(\mathbf{k}_j) \).

**Key insight**: \( S \) and \( \mathbf{z} \) are cumulative sums — they don't depend on \( i \). We compute them once, then each \( \mathbf{o}_i \) is just a matrix-vector product.

### Dual Forms

**Parallel form** (training):

$$
O = \phi(Q) \cdot \left( \phi(K)^\top V \right) \qquad \text{Time: } O(nd^2) \text{ (right-to-left matmul)}
$$

**Recurrent form** (inference):

$$
S_t = S_{t-1} + \phi(\mathbf{k}_t) \mathbf{v}_t^\top, \qquad \mathbf{z}_t = \mathbf{z}_{t-1} + \phi(\mathbf{k}_t), \qquad \mathbf{o}_t = \frac{S_t \, \phi(\mathbf{q}_t)}{\mathbf{z}_t^\top \phi(\mathbf{q}_t)}
$$

- Time: \( O(d^2) \) per step \( \to O(nd^2) \) total
- Memory: \( O(d^2) \) constant state

This duality is fundamental: same computation, two execution strategies. The parallel form is efficient for training (GPU-friendly), while the recurrent form enables O(1) per-token inference.

---

## The Hidden State as Associative Memory

The state \( S_t = \sum_{j \leq t} \phi(\mathbf{k}_j) \mathbf{v}_j^\top \) is a **key-value associative memory**:

$$
\text{Writing: } S_t = S_{t-1} + \phi(\mathbf{k}_t) \mathbf{v}_t^\top \quad \leftarrow \text{"store } \mathbf{v}_t \text{ at address } \mathbf{k}_t\text{"}
$$

$$
\text{Reading: } \mathbf{o}_t = S_t \, \phi(\mathbf{q}_t) \quad \leftarrow \text{"retrieve value at address } \mathbf{q}_t\text{"}
$$

\( S_t \in \mathbb{R}^{d_v \times d_k} \) — a matrix that stores all key-value pairs seen so far.

This is why the recurrent state size is \( d_v \times d_k \) (typically 64×64 = 4096 per head), independent of sequence length. Compare with softmax attention's KV cache which grows as \( O(n \times d) \).

### The Forgetting Problem

Vanilla linear attention **never forgets**: S_t only accumulates, so early tokens have equal weight to recent ones. This causes:

1. **Memory dilution** — as S_t accumulates more entries, each retrieval becomes noisier
2. **No overwriting** — can't update the value stored at a key; old associations persist
3. **Quality gap** — softmax attention dynamically reweights, linear attention doesn't

This limitation motivates every subsequent model: RWKV adds decay, Mamba adds selective gating, DeltaNet adds overwriting.

---

## RWKV: Channel-Wise Decay

RWKV (Receptance Weighted Key Value) introduces **exponential decay** to the recurrence:

Standard linear attention:

$$
S_t = S_{t-1} + \mathbf{k}_t \mathbf{v}_t^\top
$$

RWKV-style:

$$
S_t = \Lambda \odot S_{t-1} + \mathbf{k}_t \mathbf{v}_t^\top
$$

where \( \Lambda \) is a diagonal matrix of channel-wise decay rates (learned, fixed after training).

This gives recent tokens more influence than distant ones — a simple but effective form of forgetting. RWKV also introduces:

- **Receptance gate** (r): controls how much of the retrieved value to use
- **WKV operator**: efficiently computes the weighted key-value aggregation
- **Time-mixing and channel-mixing** blocks that parallel attention and FFN

RWKV achieves competitive perplexity with pure Transformers while maintaining O(1) per-token inference cost, proving that linear-time models can match softmax quality at scale.

---

## RetNet: Retention Mechanism

RetNet (Retentive Network) proposes **retention** as a dual of attention:

$$
\text{Retention: } S_t = \gamma \cdot S_{t-1} + \mathbf{k}_t \mathbf{v}_t^\top \qquad (\gamma = \text{scalar decay, typically per-head})
$$

Three computation modes:

1. **Parallel:** \( O = (QK^\top \odot D) V \) where \( D \) is a causal decay mask
2. **Recurrent:** \( S_t = \gamma \, S_{t-1} + \mathbf{k}_t \mathbf{v}_t^\top; \quad \mathbf{o}_t = \mathbf{q}_t^\top S_t \)
3. **Chunk-wise:** Parallel within chunks, recurrent across chunks

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
