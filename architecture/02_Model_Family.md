# LLM Model Families

> Parent: [Architecture](00_Architecture.md)

## Overview

This document traces the evolution of major LLM model families, from GPT-1 to the latest open-weight models. Each family introduced architectural innovations that became standard practice. Understanding these lineages helps you reason about which design choices matter and why.

## Learning Objectives

- [x] GPT series evolution (GPT-1 through GPT-4)
- [x] LLaMA family innovations (RMSNorm, SwiGLU, RoPE, GQA)
- [x] Mistral / Mixtral (sliding window attention, MoE)
- [x] DeepSeek (MLA, DeepSeekMoE)
- [x] Architectural comparison across families

---

## Timeline

```
2018       2019       2020       2021       2022       2023       2024       2025
 │          │          │          │          │          │          │          │
 ▼          ▼          ▼          ▼          ▼          ▼          ▼          ▼

GPT-1 ──► GPT-2 ──────────────► GPT-3 ───────────────► GPT-4 ──► GPT-4o
(117M)    (1.5B)                (175B)                  (~1.8T?)
                                  │
                                  │    Chinchilla ──────────────────────────►
                                  │    (70B, 2022)
                                  │
                                  └──────────────► LLaMA ──► LLaMA-2 ──► LLaMA-3
                                                   (7-65B)   (7-70B)    (8-405B)
                                                     │
                                                     └──► Mistral 7B ──► Mixtral 8x7B
                                                          (2023)         (MoE, 2024)
                                                                              │
                                                     DeepSeek ──► V2 ──► V3 ──► R1
                                                     (2024)      (MLA)  (MoE)  (reasoning)
                                                                  │
                                                          Qwen ──► Qwen-2 ──► Qwen-2.5
                                                          (2024)

Key innovations at each stage:
──────────────────────────────────
GPT-1:      Decoder-only pre-training + fine-tuning
GPT-2:      Pre-norm, larger scale, zero-shot capability
GPT-3:      In-context learning, few-shot prompting
LLaMA:      RMSNorm, SwiGLU, RoPE (open-weight recipe)
LLaMA-2:    GQA, longer context (4K), RLHF
Mistral:    Sliding window attention, efficient 7B
Mixtral:    Sparse MoE (8 experts, top-2 routing)
DeepSeek-V2: MLA (Multi-head Latent Attention)
DeepSeek-V3: DeepSeekMoE + MLA + auxiliary-loss-free balancing
```

---

## 1. GPT Series (OpenAI)

### GPT-1 (2018) -- Decoder-Only Pre-training

The first model to show that unsupervised pre-training on a large corpus, followed by supervised fine-tuning, produces strong NLP performance.

```
GPT-1 Architecture:
  - 12 layers, 768 hidden, 12 heads
  - 117M parameters
  - BooksCorpus (800M words)
  - Learned position embeddings
  - Post-norm (original Transformer style)

Key insight: Pre-train generatively, fine-tune discriminatively.
```

### GPT-2 (2019) -- Scale + Pre-Norm

```
GPT-2 changes from GPT-1:
  - Pre-norm (LayerNorm before sublayers, not after)
  - 10x scale: 1.5B parameters
  - WebText dataset (40GB)
  - Zero-shot task performance (no fine-tuning needed)
  - Modified initialization for deep residual streams

GPT-2 sizes:
  Small:  117M  (12 layers, 768 hidden)
  Medium: 345M  (24 layers, 1024 hidden)
  Large:  762M  (36 layers, 1280 hidden)
  XL:     1.5B  (48 layers, 1600 hidden)
```

### GPT-3 (2020) -- In-Context Learning

```
GPT-3 key innovations:
  - 175B parameters (116x GPT-2)
  - 96 layers, 12288 hidden, 96 heads
  - Alternating dense and locally banded sparse attention
  - 300B tokens training data
  - Context length: 2048 tokens

Emergent capability: few-shot learning via prompting
  "Translate English to French:
   sea otter => loutre de mer
   cheese => "
   → "fromage"  (no fine-tuning!)
```

### GPT-4 (2023) -- Mixture of Experts (rumored)

```
GPT-4 (details not officially published):
  - Rumored: ~1.8T total params, MoE with 16 experts
  - ~280B active parameters per forward pass (rumored)
  - 8K / 32K / 128K context lengths
  - Multimodal (text + vision)
  - Substantially improved reasoning
```

### GPT Series Summary

| Model | Year | Params | Layers | d_model | Heads | Context | Key Innovation |
|-------|------|--------|--------|---------|-------|---------|----------------|
| GPT-1 | 2018 | 117M | 12 | 768 | 12 | 512 | Decoder-only pre-training |
| GPT-2 | 2019 | 1.5B | 48 | 1600 | 25 | 1024 | Pre-norm, zero-shot |
| GPT-3 | 2020 | 175B | 96 | 12288 | 96 | 2048 | In-context learning |
| GPT-4 | 2023 | ~1.8T? | ? | ? | ? | 128K | MoE?, multimodal |

---

## 2. LLaMA Family (Meta)

The LLaMA family established the modern open-weight recipe. Almost every open model since 2023 adopts LLaMA's architectural choices.

### LLaMA (2023) -- The Open-Weight Recipe

```
LLaMA design choices (vs GPT-3):
┌───────────────────┬──────────────┬──────────────────────┐
│ Component         │ GPT-3        │ LLaMA                │
├───────────────────┼──────────────┼──────────────────────┤
│ Normalization     │ LayerNorm    │ RMSNorm (faster)     │
│ Activation        │ GELU         │ SwiGLU (better)      │
│ Position encoding │ Learned      │ RoPE (extrapolates)  │
│ Bias terms        │ Yes          │ No (fewer params)    │
│ Attention         │ MHA          │ MHA                  │
│ FFN hidden dim    │ 4 * d_model  │ 8/3 * d_model        │
│ Weight tying      │ Yes          │ No (7B), Yes (others)│
└───────────────────┴──────────────┴──────────────────────┘
```

**RMSNorm** -- faster than LayerNorm, removes mean-centering:

```
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
RMSNorm(x)   = x / sqrt(mean(x^2) + eps) * gamma

RMSNorm removes:
  - Mean subtraction (saves compute)
  - Beta (bias) parameter (fewer params)
  Performance: ~equivalent to LayerNorm, 10-15% faster
```

**SwiGLU** -- gated activation, better than GELU:

```
Standard FFN:   FFN(x) = W_2 * GELU(W_1 * x)          params: 2 * d * d_ff
SwiGLU FFN:     FFN(x) = W_down * (SiLU(W_gate * x) * (W_up * x))
                                                         params: 3 * d * d_ff

To match parameter count, SwiGLU uses d_ff = 8/3 * d instead of 4 * d:
  Standard: 2 * d * 4d = 8d^2
  SwiGLU:   3 * d * (8/3)d = 8d^2   ← same total parameters
```

**RoPE** (Rotary Position Embeddings):

```
Encodes position by rotating Q/K vectors in 2D subspaces:

  q_rotated[2i]   = q[2i]*cos(m*theta_i) - q[2i+1]*sin(m*theta_i)
  q_rotated[2i+1] = q[2i]*sin(m*theta_i) + q[2i+1]*cos(m*theta_i)

  where m = position index, theta_i = 10000^(-2i/d)

Benefits:
  - Relative position information in dot product
  - No learned parameters (0 additional params)
  - Extrapolates to longer sequences than trained on
  - Naturally decays attention with distance
```

### LLaMA-2 (2023) -- GQA + RLHF

```
LLaMA-2 changes from LLaMA:
  - Grouped-Query Attention (GQA) for 34B and 70B
  - Context length: 2048 → 4096
  - 40% more training data (2T tokens)
  - RLHF alignment (LLaMA-2-Chat)
  - Ghost Attention for multi-turn dialogue
```

**GQA** (Grouped-Query Attention):

```
MHA:  32 Q heads, 32 KV heads  → KV cache = 32 * 2 * d_h per layer
GQA:  32 Q heads,  8 KV heads  → KV cache =  8 * 2 * d_h per layer (4x smaller!)
MQA:  32 Q heads,  1 KV head   → KV cache =  1 * 2 * d_h per layer

LLaMA-2 70B uses GQA with 8 KV heads:
  KV cache per token = 2 * 80 * 8 * 128 * 2 bytes = 327,680 bytes ≈ 320 KB
  (vs MHA: 2 * 80 * 64 * 128 * 2 = 2.6 MB per token)
```

### LLaMA-3 (2024) -- Scale + Longer Context

```
LLaMA-3 changes from LLaMA-2:
  - Vocabulary: 32K → 128K tokens (better multilingual, code)
  - Context: 4K → 8K (base) → 128K (extended)
  - GQA for all sizes (including 8B)
  - Training data: 2T → 15T+ tokens
  - 405B flagship model
  - Improved tokenizer (tiktoken-based, BPE)
```

### LLaMA Family Summary

| Model | Year | Params | Layers | d_model | Heads | KV Heads | d_ff | Context | Vocab |
|-------|------|--------|--------|---------|-------|----------|------|---------|-------|
| LLaMA 7B | 2023 | 6.7B | 32 | 4096 | 32 | 32 | 11008 | 2048 | 32K |
| LLaMA 65B | 2023 | 65.2B | 80 | 8192 | 64 | 64 | 22016 | 2048 | 32K |
| LLaMA-2 7B | 2023 | 6.7B | 32 | 4096 | 32 | 32 | 11008 | 4096 | 32K |
| LLaMA-2 70B | 2023 | 68.9B | 80 | 8192 | 64 | 8 | 28672 | 4096 | 32K |
| LLaMA-3 8B | 2024 | 8.0B | 32 | 4096 | 32 | 8 | 14336 | 8192 | 128K |
| LLaMA-3 70B | 2024 | 70.6B | 80 | 8192 | 64 | 8 | 28672 | 8192 | 128K |
| LLaMA-3 405B | 2024 | 405B | 126 | 16384 | 128 | 8 | 53248 | 128K | 128K |

---

## 3. Mistral / Mixtral (Mistral AI)

### Mistral 7B (2023) -- Efficient Dense Model

```
Mistral 7B innovations:
  - Sliding Window Attention (SWA): window size = 4096
  - GQA: 8 KV heads (like LLaMA-2 70B, but in a 7B model)
  - Outperforms LLaMA-2 13B on most benchmarks with fewer params
```

**Sliding Window Attention (SWA)**:

```
Standard causal attention (token 8 attends to tokens 1-8):
  Token:  1  2  3  4  5  6  7  8
  Attend: [1, 1, 1, 1, 1, 1, 1, 1]  ← O(T^2) cost

Sliding window attention (window=4, token 8 attends to 5-8):
  Token:  1  2  3  4  5  6  7  8
  Attend: [0, 0, 0, 0, 1, 1, 1, 1]  ← O(T*W) cost

But information still propagates beyond the window via stacking:
  Layer 1: token 8 sees tokens 5-8 directly
  Layer 2: token 8 sees tokens 1-8 (through tokens 5-8 from layer 1)
  After L layers: effective context = L * W

  Mistral 7B: 32 layers * 4096 window = 131K effective context
```

```
Sliding Window Attention pattern (window=4):

         t1  t2  t3  t4  t5  t6  t7  t8
   t1   [ 1   .   .   .   .   .   .   . ]
   t2   [ 1   1   .   .   .   .   .   . ]
   t3   [ 1   1   1   .   .   .   .   . ]
   t4   [ 1   1   1   1   .   .   .   . ]
   t5   [ .   1   1   1   1   .   .   . ]  ← window slides
   t6   [ .   .   1   1   1   1   .   . ]
   t7   [ .   .   .   1   1   1   1   . ]
   t8   [ .   .   .   .   1   1   1   1 ]

KV cache only needs to store W entries per layer (rolling buffer):
  Memory: O(W) instead of O(T)  → fixed memory regardless of seq length
```

### Mixtral 8x7B (2024) -- Sparse Mixture of Experts

```
Mixtral architecture:
  - 8 expert FFN networks per layer (each ~7B FFN)
  - Top-2 routing: each token uses 2 of 8 experts
  - Total params: 46.7B
  - Active params: ~12.9B per token (2/8 experts + shared attention)
  - Matches or beats LLaMA-2 70B with 3-4x less compute per token!
```

```
Standard dense FFN (one FFN per layer):
  Input ──► [FFN] ──► Output

Mixtral MoE FFN (8 experts, top-2 per token):

  Input ──► Router ──┬──► Expert 0  ─── w=0.0 ──┐
            (Linear)  ├──► Expert 1  ─── w=0.0 ──┤
                      ├──► Expert 2  ─── w=0.7 ──┤──► Weighted Sum ──► Output
                      ├──► Expert 3  ─── w=0.0 ──┤
                      ├──► Expert 4  ─── w=0.3 ──┤
                      ├──► Expert 5  ─── w=0.0 ──┤
                      ├──► Expert 6  ─── w=0.0 ──┤
                      └──► Expert 7  ─── w=0.0 ──┘
                                ↑
                          Only top-2 are computed
                          (rest are skipped entirely)

Router: softmax(Linear(x))  → select top-2 weights
Output: w_2 * Expert_2(x) + w_4 * Expert_4(x)
```

```
Mixtral MoE cost vs dense model:

Dense 46.7B:   All 46.7B params computed per token
Mixtral 46.7B: Only 12.9B params computed per token

  Inference FLOPs: Mixtral ≈ Dense 13B model
  Quality:         Mixtral ≈ Dense 70B model
  Best of both worlds!
```

---

## 4. DeepSeek Family

### DeepSeek-V2 (2024) -- Multi-head Latent Attention (MLA)

The key innovation is **MLA**: compressing the KV cache into a low-rank latent space, dramatically reducing memory while maintaining quality.

```
Standard GQA (LLaMA-style):
  K, V ∈ R^(n_kv_heads x head_dim) per token
  KV cache per token: 2 * n_kv * d_h * bytes

MLA (DeepSeek-V2):
  Compress K, V into a shared low-rank latent vector:
  c = W_down * [K; V]    where c ∈ R^(d_c)   (d_c << n_kv * d_h)

  At attention time, decompress:
  K = W_K_up * c
  V = W_V_up * c

  KV cache per token: d_c * bytes   (5-13x smaller than GQA!)
```

```
GQA vs MLA KV cache comparison (per token per layer):

GQA (LLaMA-2 70B):
  KV cache = 2 * 8 * 128 * 2 bytes = 4,096 bytes

MLA (DeepSeek-V2):
  KV cache = 512 * 2 bytes = 1,024 bytes

  → 4x reduction, and quality is equivalent or better!
```

### DeepSeek-V3 (2024) -- MLA + DeepSeekMoE

```
DeepSeek-V3 architecture:
  - 671B total parameters
  - 37B active parameters per token
  - 61 layers
  - MLA attention (from V2)
  - DeepSeekMoE: 256 routed experts + 1 shared expert, top-8 routing
  - Auxiliary-loss-free load balancing
  - FP8 mixed-precision training
  - Trained on 14.8T tokens
  - Training cost: only $5.57M (extremely efficient)
```

**DeepSeekMoE** differs from Mixtral's approach:

```
Mixtral MoE:
  8 large experts, top-2 routing
  Each expert = full-size FFN (~7B params each)

DeepSeekMoE:
  256 small routed experts + 1 shared expert, top-8 routing
  Each routed expert = small FFN (~0.26B params each)
  Shared expert is always active (captures common knowledge)

  ┌──────────────────────────────────────────────────┐
  │                    Input                          │
  │                      │                            │
  │         ┌────────────┼────────────┐              │
  │         ▼            ▼            ▼              │
  │   ┌──────────┐ ┌──────────┐ ┌──────────┐        │
  │   │ Shared   │ │ Router   │ │          │        │
  │   │ Expert   │ │ → top-8  │ │          │        │
  │   │ (always  │ │ of 256   │ │          │        │
  │   │  active) │ │ experts  │ │          │        │
  │   └────┬─────┘ └────┬─────┘ └──────────┘        │
  │        │             │                            │
  │        └──────┬──────┘                            │
  │               │                                   │
  │         Weighted Sum                              │
  │               │                                   │
  │               ▼                                   │
  │            Output                                 │
  └──────────────────────────────────────────────────┘

Benefits of many small experts:
  - Finer-grained specialization
  - Better load balancing
  - Shared expert captures common patterns
```

**Auxiliary-loss-free load balancing**:

```
Traditional MoE balancing:
  L_total = L_language + alpha * L_balance
  Problem: alpha is hard to tune, degrades LM quality

DeepSeek-V3 approach:
  Add a bias term to the router per expert:
  routing_score = softmax(Linear(x) + bias)

  Adjust bias dynamically:
  - If expert is overloaded: decrease its bias
  - If expert is underused: increase its bias
  - No gradient through bias → no interference with LM loss!
```

### DeepSeek Family Summary

| Model | Year | Total Params | Active Params | Layers | Attention | FFN | MoE Config |
|-------|------|-------------|---------------|--------|-----------|-----|------------|
| DeepSeek 67B | 2024 | 67B | 67B | 95 | MHA | SwiGLU | Dense |
| DeepSeek-V2 | 2024 | 236B | 21B | 60 | MLA | DeepSeekMoE | 160 experts, top-6 + 2 shared |
| DeepSeek-V3 | 2024 | 671B | 37B | 61 | MLA | DeepSeekMoE | 256 experts, top-8 + 1 shared |
| DeepSeek-R1 | 2025 | 671B | 37B | 61 | MLA | DeepSeekMoE | Same as V3 (RL-tuned) |

---

## 5. Master Comparison Table

| Property | GPT-3 | LLaMA-2 7B | LLaMA-3 8B | Mistral 7B | Mixtral 8x7B | DeepSeek-V3 |
|----------|-------|------------|------------|------------|--------------|-------------|
| **Params (total)** | 175B | 6.7B | 8.0B | 7.3B | 46.7B | 671B |
| **Params (active)** | 175B | 6.7B | 8.0B | 7.3B | 12.9B | 37B |
| **Layers** | 96 | 32 | 32 | 32 | 32 | 61 |
| **d_model** | 12288 | 4096 | 4096 | 4096 | 4096 | 7168 |
| **Heads** | 96 | 32 | 32 | 32 | 32 | 128 |
| **KV Heads** | 96 | 32 | 8 | 8 | 8 | MLA (latent) |
| **d_ff** | 49152 | 11008 | 14336 | 14336 | 14336 | - |
| **Attention type** | MHA | MHA | GQA | GQA + SWA | GQA + SWA | MLA |
| **FFN type** | GELU | SwiGLU | SwiGLU | SwiGLU | SwiGLU MoE | DeepSeekMoE |
| **Position enc.** | Learned | RoPE | RoPE | RoPE | RoPE | RoPE |
| **Normalization** | LayerNorm | RMSNorm | RMSNorm | RMSNorm | RMSNorm | RMSNorm |
| **Vocab size** | 50257 | 32000 | 128256 | 32000 | 32000 | 129280 |
| **Context** | 2048 | 4096 | 8192 | 32K (SWA) | 32K (SWA) | 128K |
| **Training tokens** | 300B | 2T | 15T+ | 1T+ | 1T+ | 14.8T |

---

## Architectural Innovation Adoption

Which innovations were adopted by subsequent models:

```
Innovation         Introduced By      Now Used By
─────────────────────────────────────────────────────────
Pre-norm           GPT-2 (2019)       Everyone
RMSNorm            LLaMA (2023)       LLaMA, Mistral, DeepSeek, Qwen
SwiGLU             LLaMA (2023)       LLaMA, Mistral, DeepSeek, Qwen
RoPE               LLaMA (2023)       LLaMA, Mistral, DeepSeek, Qwen
GQA                LLaMA-2 (2023)     LLaMA-2/3, Mistral, Qwen
SWA                Mistral (2023)     Mistral, Mixtral
MLA                DeepSeek-V2 (2024) DeepSeek family
Sparse MoE         Mixtral (2024)     Mixtral, DeepSeek, Qwen-MoE
No bias terms      LLaMA (2023)       LLaMA, Mistral, DeepSeek
128K+ vocab        LLaMA-3 (2024)     LLaMA-3, DeepSeek-V3

The "LLaMA recipe" (RMSNorm + SwiGLU + RoPE + no bias)
is now the de facto standard for all open models.
```

---

## Key Papers

| Paper | Year | Model | Key Contribution |
|-------|------|-------|------------------|
| [Improving Language Understanding (GPT-1)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | 2018 | GPT-1 | Decoder-only pre-training |
| [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 2019 | GPT-2 | Zero-shot via scale |
| [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) | 2020 | GPT-3 | In-context learning |
| [LLaMA: Open and Efficient Foundation Models](https://arxiv.org/abs/2302.13971) | 2023 | LLaMA | Open-weight recipe |
| [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) | 2023 | LLaMA-2 | GQA + RLHF |
| [Mistral 7B](https://arxiv.org/abs/2310.06825) | 2023 | Mistral 7B | Sliding window attention |
| [Mixtral of Experts](https://arxiv.org/abs/2401.04088) | 2024 | Mixtral | Sparse MoE |
| [DeepSeek-V2: A Strong, Economical, and Efficient MoE LM](https://arxiv.org/abs/2405.04434) | 2024 | DeepSeek-V2 | MLA |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | 2024 | DeepSeek-V3 | Efficient MoE training |

---

## Related

- [Decoder-Only Architecture](01_Decoder_Only.md) -- full walkthrough of the base architecture
- [Scaling Laws](03_Scaling_Laws.md) -- how to decide model size vs data size
- [Attention](../attention/00_Attention.md) -- MHA, GQA, MQA, MLA details
- [FFN](../ffn/00_FFN.md) -- SwiGLU, MoE internals
- [Position Encoding](../position_encoding/00_Position_Encoding.md) -- RoPE, learned, ALiBi
