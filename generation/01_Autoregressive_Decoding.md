# Autoregressive Decoding

> Parent: [Generation](00_Generation.md)

## Overview

Autoregressive decoding is how LLMs generate text: tokens are produced one at a time, each conditioned on all previous tokens. The process decomposes into two distinct phases -- **prefill** (process the prompt) and **decode** (generate new tokens) -- with fundamentally different computational profiles. The **KV cache** is the key data structure that makes generation efficient by avoiding redundant computation.

## The Autoregressive Factorization

An LLM models the joint probability of a sequence by factoring it into a chain of conditional probabilities:

```
P(x_1, x_2, ..., x_T) = P(x_1) * P(x_2|x_1) * P(x_3|x_1,x_2) * ... * P(x_T|x_1,...,x_{T-1})

                       = ∏(t=1..T) P(x_t | x_{<t})
```

At each step, the model sees all previous tokens and predicts a probability distribution over the vocabulary for the next token. A sampling strategy (greedy, top-p, etc.) then selects one token from this distribution.

## Two Phases: Prefill and Decode

```
Input prompt: "Explain quantum computing in simple terms"
                              │
    ┌─────────────────────────▼──────────────────────────────┐
    │              PREFILL PHASE                              │
    │                                                         │
    │  All prompt tokens processed in ONE forward pass        │
    │  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐           │
    │  │Exp│→│lain│→│ qu│→│ant│→│um │→│com│→│...│            │
    │  └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘            │
    │     ↓     ↓     ↓     ↓     ↓     ↓     ↓             │
    │  ┌─────────────────────────────────────────┐            │
    │  │        KV Cache initialized             │            │
    │  │   K: [k_1, k_2, ..., k_P]              │            │
    │  │   V: [v_1, v_2, ..., v_P]              │            │
    │  │   (for each layer and each head)        │            │
    │  └─────────────────────────────────────────┘            │
    │                                                         │
    │  Characteristic: COMPUTE-BOUND                          │
    │  - Large matrix multiplications (all tokens at once)    │
    │  - High arithmetic intensity                            │
    │  - GPU compute fully utilized                           │
    │  - Measured by: TTFT (Time To First Token)              │
    └─────────────────────────┬──────────────────────────────┘
                              │
    ┌─────────────────────────▼──────────────────────────────┐
    │              DECODE PHASE                               │
    │                                                         │
    │  Tokens generated ONE AT A TIME                         │
    │                                                         │
    │  Step 1: Query with last token → read KV cache          │
    │     ┌───┐                                               │
    │     │ t1│ → logits → sample → "Quantum"                 │
    │     └───┘                                               │
    │     KV cache: [k_1..k_P, k_{P+1}]  (append)            │
    │                                                         │
    │  Step 2: Query with "Quantum" → read KV cache           │
    │     ┌───┐                                               │
    │     │ t2│ → logits → sample → "computing"               │
    │     └───┘                                               │
    │     KV cache: [k_1..k_P, k_{P+1}, k_{P+2}]             │
    │                                                         │
    │  Step 3: ... continue until <EOS> or max_length         │
    │                                                         │
    │  Characteristic: MEMORY-BOUND                           │
    │  - Tiny matmuls (1 token query vs full KV cache)        │
    │  - Low arithmetic intensity                             │
    │  - Bottleneck: loading KV cache from HBM                │
    │  - Measured by: TPOT (Time Per Output Token)            │
    └────────────────────────────────────────────────────────┘
```

### Prefill vs Decode Comparison

| Property | Prefill | Decode |
|----------|---------|--------|
| Tokens processed | All prompt tokens (P) | 1 token per step |
| Parallelism | High (all tokens at once) | None (sequential) |
| Bottleneck | Compute (FLOPs) | Memory bandwidth |
| Arithmetic intensity | High | Low |
| Key metric | TTFT (Time To First Token) | TPOT (Time Per Output Token) |
| Typical GPU utilization | 60-80% | 5-15% |
| Matrix shapes | (P, d) x (d, d) | (1, d) x (d, d) |

### Why Decode is Memory-Bound

During decode, the model performs a forward pass for a single token. The weight matrices are the same size as during prefill, but the input is a single vector instead of a matrix:

```
Prefill:   X @ W  →  (P, d_model) @ (d_model, d_model)  →  P * d^2 FLOPs
                                                              P * d^2 bytes to load

           Arithmetic intensity = FLOPs / bytes = O(P)   ← high when P is large

Decode:    x @ W  →  (1, d_model) @ (d_model, d_model)  →  d^2 FLOPs
                                                              d^2 bytes to load

           Arithmetic intensity = FLOPs / bytes = O(1)   ← always low!
```

The GPU must load all model weights from HBM for each decode step, but only uses them for a single vector-matrix multiply. This is why decode throughput is limited by memory bandwidth, not compute.

---

## KV Cache

### Why KV Cache is Necessary

In self-attention, each token's output depends on the keys and values of **all** previous tokens:

```
Attention(q_t, K_{1:t}, V_{1:t}) = softmax(q_t · K_{1:t}^T / sqrt(d_k)) · V_{1:t}
```

**Without KV cache:** At step t, you must recompute K and V for all t tokens from scratch. This requires running the full model on all t tokens, making total generation cost O(n^2) in sequence length.

**With KV cache:** At step t, you only compute K_t and V_t for the new token, then append them to the cached K_{1:t-1} and V_{1:t-1}. This makes each decode step O(n) -- reading the cache -- instead of O(n^2) -- recomputing everything.

```
WITHOUT KV Cache (naive):                WITH KV Cache:

Step 1: Process [t_1]              →  1   Step 1: Process [t_1]           →  1
Step 2: Process [t_1, t_2]        →  2   Step 2: Process [t_2] + cache   →  1
Step 3: Process [t_1, t_2, t_3]   →  3   Step 3: Process [t_3] + cache   →  1
Step 4: Process [t_1,...,t_4]      →  4   Step 4: Process [t_4] + cache   →  1
  ...                                       ...
Step n: Process [t_1,...,t_n]      →  n   Step n: Process [t_n] + cache   →  1

Total tokens processed:                   Total tokens processed:
  1 + 2 + 3 + ... + n = O(n^2)             1 + 1 + 1 + ... + 1 = O(n)

(Plus initial prefill of P tokens)
```

### KV Cache Size Calculation

For each layer, we store the Key and Value projections for all past tokens:

```
Per layer:
  K cache: seq_len × n_kv_heads × d_k × bytes_per_element
  V cache: seq_len × n_kv_heads × d_k × bytes_per_element
  Total:   2 × seq_len × n_kv_heads × d_k × bytes_per_element

All layers:
  KV cache = 2 × n_layers × seq_len × n_kv_heads × d_k × bytes_per_element
```

### Example: LLaMA-2 7B KV Cache Size

```
LLaMA-2 7B Configuration:
  n_layers   = 32
  n_kv_heads = 32    (full MHA, not GQA)
  d_k        = 128   (= d_model / n_heads = 4096 / 32)
  seq_len    = 4096
  dtype      = FP16  (2 bytes per element)

KV cache = 2 × 32 × 4096 × 32 × 128 × 2 bytes
         = 2 × 32 × 4096 × 32 × 128 × 2
         = 2,147,483,648 bytes
         ≈ 2 GB

Per token:
  2 × 32 × 32 × 128 × 2 = 524,288 bytes ≈ 0.5 MB per token
```

### KV Cache Size for Various Models

| Model | n_layers | n_kv_heads | d_k | KV per token (FP16) | KV at 4K context | KV at 128K context |
|-------|----------|------------|-----|---------------------|-------------------|---------------------|
| LLaMA-2 7B | 32 | 32 | 128 | 0.5 MB | 2 GB | 64 GB |
| LLaMA-2 70B | 80 | 8 (GQA) | 128 | 0.31 MB | 1.25 GB | 40 GB |
| LLaMA-3 8B | 32 | 8 (GQA) | 128 | 0.125 MB | 0.5 GB | 16 GB |
| Mistral 7B | 32 | 8 (GQA) | 128 | 0.125 MB | 0.5 GB | 16 GB |

**Key insight:** GQA (Grouped-Query Attention) dramatically reduces KV cache size by using fewer KV heads. LLaMA-2 70B with GQA (8 KV heads) uses **less** KV cache than LLaMA-2 7B with full MHA (32 KV heads).

---

## PyTorch Implementation

### Naive Generation (No KV Cache)

```python
import torch
import torch.nn.functional as F


def generate_naive(model, prompt_tokens: torch.Tensor, max_new_tokens: int) -> list[int]:
    """
    Naive autoregressive generation: recomputes attention over ALL tokens at every step.

    Args:
        model: A decoder-only LM that returns logits (batch, seq_len, vocab_size)
        prompt_tokens: (1, prompt_len) tensor of token IDs
        max_new_tokens: number of tokens to generate

    Returns:
        List of generated token IDs
    """
    generated = prompt_tokens.clone()  # (1, seq_len)

    for _ in range(max_new_tokens):
        # Full forward pass over ALL tokens every time — O(n^2) total
        logits = model(generated)                    # (1, seq_len, vocab_size)
        next_logits = logits[:, -1, :]               # (1, vocab_size)

        # Greedy decoding (simplest strategy)
        next_token = next_logits.argmax(dim=-1, keepdim=True)  # (1, 1)

        # Append new token to the sequence
        generated = torch.cat([generated, next_token], dim=1)

        # Check for EOS
        if next_token.item() == model.eos_token_id:
            break

    return generated[0].tolist()

# Problem: at step t, we process all t tokens.
# Total compute: O(1 + 2 + 3 + ... + n) = O(n^2)
```

### Generation with KV Cache (Prefill + Decode)

```python
import torch
import torch.nn.functional as F
from typing import Optional


def generate_with_kv_cache(
    model,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> list[int]:
    """
    Efficient autoregressive generation with KV cache.

    Phase 1 (Prefill): Process all prompt tokens at once, build KV cache.
    Phase 2 (Decode):  Generate one token at a time, reusing cached K,V.

    Args:
        model: Decoder-only LM supporting use_cache=True, past_key_values
        prompt_tokens: (1, prompt_len) tensor of token IDs
        max_new_tokens: number of tokens to generate
        temperature: sampling temperature (1.0 = no scaling)

    Returns:
        List of generated token IDs
    """
    generated_ids = []

    # ========== PHASE 1: PREFILL ==========
    # Process all prompt tokens in a single forward pass.
    # Returns logits + KV cache for every layer.
    logits, past_key_values = model(
        input_ids=prompt_tokens,      # (1, prompt_len)
        use_cache=True,               # Tell model to return KV cache
        past_key_values=None,         # No cache yet
    )
    # logits:           (1, prompt_len, vocab_size)
    # past_key_values:  tuple of (K, V) per layer
    #   K shape per layer: (batch, n_kv_heads, prompt_len, d_k)
    #   V shape per layer: (batch, n_kv_heads, prompt_len, d_k)

    # Sample next token from last position's logits
    next_logits = logits[:, -1, :] / temperature     # (1, vocab_size)
    probs = F.softmax(next_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
    generated_ids.append(next_token.item())

    # ========== PHASE 2: DECODE ==========
    # Generate one token at a time, passing only the NEW token
    # and reusing the cached K,V from all previous tokens.
    for _ in range(max_new_tokens - 1):
        logits, past_key_values = model(
            input_ids=next_token,                # (1, 1) — just the new token!
            use_cache=True,
            past_key_values=past_key_values,     # Reuse cached K,V
        )
        # Internally, the model:
        #   1. Computes Q, K, V for the new token only
        #   2. Appends new K, V to the cache
        #   3. Computes attention: q_new @ K_all^T (1 query vs all keys)
        #   4. Returns updated cache

        next_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_ids.append(next_token.item())

        if next_token.item() == model.eos_token_id:
            break

    return generated_ids


# --- Simplified KV cache attention (what happens inside the model) ---

def attention_with_kv_cache(
    q: torch.Tensor,          # (batch, n_heads, 1, d_k) — new token query only
    k_new: torch.Tensor,      # (batch, n_heads, 1, d_k) — new token key
    v_new: torch.Tensor,      # (batch, n_heads, 1, d_k) — new token value
    k_cache: torch.Tensor,    # (batch, n_heads, past_len, d_k) — cached keys
    v_cache: torch.Tensor,    # (batch, n_heads, past_len, d_k) — cached values
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single decode step with KV cache.

    Returns:
        output: (batch, n_heads, 1, d_k)
        updated_k_cache: (batch, n_heads, past_len+1, d_k)
        updated_v_cache: (batch, n_heads, past_len+1, d_k)
    """
    import math

    # Append new K, V to cache
    k_cache = torch.cat([k_cache, k_new], dim=2)  # (batch, n_heads, past_len+1, d_k)
    v_cache = torch.cat([v_cache, v_new], dim=2)  # (batch, n_heads, past_len+1, d_k)

    # Attention: new query attends to ALL keys (cached + new)
    d_k = q.size(-1)
    scores = torch.matmul(q, k_cache.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: (batch, n_heads, 1, past_len+1)

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v_cache)
    # output: (batch, n_heads, 1, d_k)

    return output, k_cache, v_cache
```

---

## Memory Allocation and Management

### Static vs Dynamic KV Cache Allocation

```
Strategy 1: DYNAMIC ALLOCATION (simple, slow)
  Step 1: Allocate cache for prompt_len tokens
  Step 2: For each new token, concatenate → new allocation
  Problem: Repeated memory allocation + copy → fragmentation

  Memory over time:
  ┌──────┐
  │ K,V  │  Step 0 (after prefill)
  └──────┘
  ┌──────┬─┐
  │ K,V  │+│  Step 1 (copy + extend)
  └──────┴─┘
  ┌──────┬──┐
  │ K,V  │++│  Step 2 (copy + extend again)
  └──────┴──┘


Strategy 2: STATIC PRE-ALLOCATION (fast, standard)
  Pre-allocate cache for max_seq_len upfront.
  Use a position pointer to track filled portion.
  No reallocation during generation.

  ┌──────┬──────────────────────────────────────┐
  │ used │           pre-allocated               │  Fixed buffer
  └──────┴──────────────────────────────────────┘
  ↑                                              ↑
  pos=P                                   max_seq_len

  Step 1: Write at pos=P+1, advance pointer
  Step 2: Write at pos=P+2, advance pointer
  No copy, no reallocation.


Strategy 3: PAGED ATTENTION (vLLM)
  Allocate cache in fixed-size blocks (pages).
  Tokens map to pages via a page table.
  Enables efficient batching of variable-length sequences.

  Page table:          Physical memory:
  ┌───┬───┬───┐       ┌─────┐ ┌─────┐ ┌─────┐
  │ 0 │ 3 │ 1 │  →    │pg 0 │ │pg 1 │ │pg 3 │
  └───┴───┴───┘       └─────┘ └─────┘ └─────┘
  Seq A pages         Non-contiguous, like virtual memory
```

### Memory Budget During Inference

For a typical inference server, the memory budget looks like:

```
Total GPU Memory (e.g., 80 GB A100)
├── Model weights:          ~14 GB  (7B params × FP16)
├── Activation memory:      ~1 GB   (temporary, reused)
├── KV cache budget:        ~60 GB  (remainder)
│   └── At 0.5 MB/token:   ~120K tokens total
│       ├── 30 requests × 4K tokens each, or
│       └── 1 request × 120K tokens
└── Framework overhead:     ~5 GB
```

This is why KV cache management is critical for serving throughput -- it determines how many concurrent requests you can handle.

---

## Stopping Conditions

Generation continues until one of these conditions is met:

| Condition | Description | Typical Value |
|-----------|-------------|---------------|
| EOS token | Model generates the end-of-sequence token | Model-specific |
| Max length | Total sequence length (prompt + generated) reaches limit | 2048-128K |
| Max new tokens | Number of generated tokens reaches limit | Task-specific |
| Stop strings | Generated text contains a specified string | `["\n\n", "Human:"]` |

---

## Key Takeaways

1. Autoregressive generation produces one token at a time, each conditioned on all previous tokens
2. **Prefill** is compute-bound (large matrices); **decode** is memory-bound (loading weights for 1 token)
3. The KV cache avoids recomputing K and V projections, reducing total generation from O(n^2) to O(n)
4. KV cache size scales linearly with sequence length and can dominate GPU memory (2 GB for LLaMA-2 7B at 4K context)
5. GQA reduces KV cache by up to 4x by sharing KV heads across query heads
6. Static pre-allocation and paged attention (vLLM) are key strategies for efficient memory management

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) | Original Transformer with autoregressive decoding |
| [Pope et al. (2022)](https://arxiv.org/abs/2211.05102) | Analysis of prefill vs decode performance characteristics |
| [Kwon et al. (2023)](https://arxiv.org/abs/2309.06180) | PagedAttention / vLLM for efficient KV cache management |
| [Ainslie et al. (2023)](https://arxiv.org/abs/2305.13245) | GQA: reducing KV cache size with grouped-query attention |

## Related

- [Sampling Strategies](02_Sampling.md) -- how to choose the next token from the logit distribution
- [Advanced Decoding](03_Advanced_Decoding.md) -- beam search, speculative decoding, constrained generation
