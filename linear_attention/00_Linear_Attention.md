# Linear Attention & Hybrid Architectures

> Parent: [LLM Algorithms](../00_LLM.md)

## Overview

Modern LLMs are moving beyond pure softmax attention. A family of **linear attention** methods — including state space models, gated linear attention, and delta-rule variants — offers O(1) per-token memory during inference by maintaining a fixed-size recurrent state instead of a growing KV cache. The latest models like **Qwen 3.5** use **hybrid architectures** that interleave these linear attention layers with traditional softmax attention layers, getting the best of both worlds.

## The Unification Story

All these models share a single recurrence:

$$
S_t = \alpha_t \cdot \left( S_{t-1} - \beta_t \cdot S_{t-1} \mathbf{k}_t \mathbf{k}_t^\top \right) + \mathbf{v}_t \mathbf{k}_t^\top
$$

where:

- \( S_t \in \mathbb{R}^{d_v \times d_k} \) — matrix-valued hidden state (associative memory)
- \( \mathbf{k}_t, \mathbf{v}_t \) — key and value at step \( t \)
- \( \alpha_t \) — decay/forgetting gate (scalar or diagonal)
- \( \beta_t \) — delta-rule correction strength

By setting different combinations of α and β, you get every model in this family:

| Model | Decay α_t | Delta β_t | Update Rule |
|-------|-----------|-----------|-------------|
| Linear Attention | 1 | 0 | `S_t = S_{t-1} + v_t k_t^T` |
| RWKV / RetNet | channel decay | 0 | `S_t = Λ S_{t-1} + v_t k_t^T` |
| SSM / Mamba-2 | scalar α_t | 0 | `S_t = α_t S_{t-1} + v_t k_t^T` |
| GLA | data-dep gate | 0 | `S_t = G_t ⊙ S_{t-1} + v_t k_t^T` |
| DeltaNet | 1 | β_t | `S_t = S_{t-1} + β_t(v_t - S_{t-1}k_t)k_t^T` |
| **Gated DeltaNet** | α_t | β_t | `S_t = α_t(I - β_t k_t k_t^T)S_{t-1} + β_t v_t k_t^T` |

*Adapted from Table 1 of Yang, Kautz & Hatamizadeh (ICLR 2025)*

## Evolution

```
Softmax Attention (2017)          ← O(n²) exact, gold standard
    │
    ▼
Linear Attention (2020)           ← Drop softmax, kernel trick → O(n)
    │                                but quality drops (no forgetting)
    ├──────────────────┐
    ▼                  ▼
RWKV / RetNet (2023)   State Space Models
channel decay           S4 → S4D → Mamba-1
    │                       │
    │                       ▼
    │                  Mamba-2 / SSD (2024)
    │                  scalar decay + duality framework
    │                       │
    ▼                       │
GLA (2024)         ◄───────┘
data-dependent gating
    │
    ▼
DeltaNet (2024)               ← delta rule: can OVERWRITE associations
    │
    ▼
Gated DeltaNet (ICLR 2025)   ← combines gating + delta rule
    │
    ▼
Hybrid Architecture (2025)    ← interleave with softmax attention
(Qwen 3.5, Jamba, Samba...)      best of both worlds
```

## Topics

| Topic | File | Key Idea |
|-------|------|----------|
| Linear Attention Basics | [01_Linear_Attention_Basics.md](01_Linear_Attention_Basics.md) | Kernel trick, RNN duality, RWKV |
| State Space Models | [02_State_Space_Models.md](02_State_Space_Models.md) | S4, HiPPO, continuous→discrete |
| Mamba | [03_Mamba.md](03_Mamba.md) | Selective scan, SSD duality |
| Gated DeltaNet | [04_Gated_DeltaNet.md](04_Gated_DeltaNet.md) | GLA, delta rule, Gated DeltaNet |
| Hybrid Architecture | [05_Hybrid_Architecture.md](05_Hybrid_Architecture.md) | Interleaving attention + linear, Qwen 3.5 |

## Key Papers

| # | Paper | Year | Key Contribution |
|---|-------|------|------------------|
| 1 | [Transformers are RNNs (Katharopoulos et al.)](https://arxiv.org/abs/2006.16236) | 2020 | Linear attention as RNN, kernel trick |
| 2 | [RWKV (Peng et al.)](https://arxiv.org/abs/2305.13048) | 2023 | Channel-wise decay, WKV operator |
| 3 | [Efficiently Modeling Long Sequences with Structured State Spaces — S4 (Gu et al.)](https://arxiv.org/abs/2111.00396) | 2022 | HiPPO, continuous-time SSM |
| 4 | [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao)](https://arxiv.org/abs/2312.00752) | 2023 | Selective scan, input-dependent SSM |
| 5 | [Transformers are SSMs — Mamba-2 (Dao & Gu)](https://arxiv.org/abs/2405.21060) | 2024 | SSD framework, SSM–attention duality |
| 6 | [Gated Linear Attention — GLA (Yang et al.)](https://arxiv.org/abs/2312.06635) | 2024 | Data-dependent gating, WY representation |
| 7 | [Parallelizing Linear Transformers with the Delta Rule — DeltaNet (Yang et al.)](https://arxiv.org/abs/2406.06484) | 2024 | Delta rule for associative memory |
| 8 | [Gated Delta Networks (Yang, Kautz & Hatamizadeh)](https://arxiv.org/abs/2412.06464) | 2025 | Unified gating + delta rule (ICLR 2025) |
| 9 | [FLA: Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) | 2024 | Triton kernels for linear attention family |
| 10 | Qwen 3.5 Technical Report | 2025 | Hybrid Gated DeltaNet + sliding window |

## Recommended Blog Posts

Unlike the papers above, these blogs walk through the full pipeline with intuition, code, and diagrams. Suggested reading order: fundamentals (1–3), then SSMs & Mamba (4–8), then DeltaNet & hybrids (9–10), then the unifying perspective (11).

### Linear Attention Fundamentals

| # | Blog | Author | Why Read It |
|---|------|--------|-------------|
| 1 | [Linear Attention Fundamentals](https://haileyschoelkopf.github.io/blog/2024/linear-attn/) | Hailey Schoelkopf | Clean derivation of linear attention → RNN duality → chunkwise parallel form. Best single-post overview of the core math. |
| 2 | [Efficient Attention Explained](https://blog.lambdaclass.com/efficient-attention-explained-the-math-behind-linear-time-transformers/) | Lambda Class | Intuitive walkthrough with PyTorch code and benchmarks. Great "global attention maps" analogy for what linear attention trades away vs softmax. |
| 3 | [LION Part I – Full Linear Attention](https://lions-epfl.github.io/2025/lion-part1-model/) | LIONS @ EPFL | Extends linear attention to the bi-directional (non-causal) setting. [Parts II–IV](https://lions-epfl.github.io/2025/lion-part2-theory/) cover the equivalent RNN and chunkwise form. |

### State Space Models & Mamba

| # | Blog | Author | Why Read It |
|---|------|--------|-------------|
| 4 | [Introduction to State Space Models (SSM)](https://huggingface.co/blog/lbourdois/get-on-the-ssm-train) | Loïck Bourdois (HF) | Most comprehensive SSM intro on the web (214+ upvotes). Continuous → discrete → recurrent → convolutional views, HiPPO initialization, and the pros/cons of each view. Start here if SSMs are new. |
| 5 | [The Annotated S4](https://srush.github.io/annotated-s4/) | Sasha Rush | Gold-standard annotated code walkthrough. Build a working S4 from scratch — trains as CNN, converts to RNN at test time. See also the [S4D companion](https://srush.github.io/annotated-s4/s4d.html) for the simplified diagonal variant. |
| 6 | [Mamba: The Easy Way](https://jackcook.com/2024/02/23/mamba.html) | Jack Cook | Beginner-friendly visual explanation of Mamba's selective scan and why input-dependent SSM parameters matter. |
| 7 | [Mamba: The Hard Way](https://srush.github.io/annotated-mamba/hard.html) | Sasha Rush | Step-by-step Triton GPU kernel implementation of Mamba's S6 selective scan. Read after "The Easy Way" if you want to understand the hardware-level details. |
| 8 | [State Space Duality (Mamba-2) Parts I–IV](https://goombalab.github.io/blog/2024/mamba2-part1-model/) | Albert Gu & Tri Dao | Official 4-part series by the Mamba authors. Part I: the SSD model and SSM↔attention duality. [Part II](https://goombalab.github.io/blog/2024/mamba2-part2-theory/): theory. [Part III](https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/): chunkwise algorithm. [Part IV](https://goombalab.github.io/blog/2024/mamba2-part4-systems/): systems/hardware. |

### DeltaNet, Gated DeltaNet & Hybrid Architectures

| # | Blog | Author | Why Read It |
|---|------|--------|-------------|
| 9 | [DeltaNet Explained (Parts I–III)](https://sustcsonglin.github.io/blog/2024/deltanet-1/) | Songlin Yang (first author) | Derives the delta rule from MSE minimization, explains why DeltaNet dominates MQAR retrieval benchmarks, and walks through the chunkwise parallel algorithm. [Part II](https://sustcsonglin.github.io/blog/2024/deltanet-2/): the algorithm. [Part III](https://sustcsonglin.github.io/blog/2024/deltanet-3/): neural architecture. |
| 10 | [Gated DeltaNet: The "Surgical Eraser"](https://pub.towardsai.net/gated-deltanet-the-surgical-eraser-solving-linear-attentions-memory-problem-1e50ca3e42ab) | Revanth Madamala (Towards AI) | Accessible overview with the "librarian" analogy. Covers gating + delta rule math, Householder interpretation, WY representation for chunkwise training, and hybrid architecture design (Gated DeltaNet + sliding window attention). |

### Unifying Perspectives

| # | Blog | Author | Why Read It |
|---|------|--------|-------------|
| 11 | [Your Linear Layers are Secretly Fast Weights](https://kyleliang919.github.io/The_Ultimate_Fast_Weights/) | Kyle Liang | Connects linear attention, fast weight programmers, and gradient descent into one unifying view. Provocative take on why hybrid models exist and what comes next. |

## Related

- [Attention Mechanism](../attention/00_Attention.md) — Softmax attention (the baseline these methods extend/replace)
- [Efficient Attention](../attention/05_Efficient_Attention.md) — FlashAttention (complementary optimization for softmax layers)
