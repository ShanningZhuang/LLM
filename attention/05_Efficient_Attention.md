# Efficient Attention

> Parent: [Attention](00_Attention.md)

## Overview

Standard attention materializes the full n×n attention matrix in GPU HBM (high-bandwidth memory), consuming O(n²) memory and causing excessive memory traffic. FlashAttention and its successors solve this by **tiling** the computation so the attention matrix never leaves fast SRAM, reducing memory usage to O(n) and achieving 2-4× wall-clock speedup.

---

## The Memory Problem

### Standard Attention IO Pattern

```
                          HBM (slow, large)            SRAM (fast, small)
                         ┌──────────────┐             ┌──────────────┐
                         │  Q  (n × d)  │────load────▶│              │
                         │  K  (n × d)  │────load────▶│  Compute     │
                         │              │             │  S = QK^T    │
                         │              │◀───store────│  (n × n)     │
                         │  S  (n × n)  │             │              │  ← Problem!
                         │              │────load────▶│              │     n² reads +
                         │              │             │  P = softmax │     n² writes
                         │              │◀───store────│  (S)         │     for S
                         │  P  (n × n)  │             │              │
                         │              │────load────▶│              │
                         │  V  (n × d)  │────load────▶│  O = P·V    │
                         │              │◀───store────│              │
                         │  O  (n × d)  │             └──────────────┘
                         └──────────────┘

Total HBM reads/writes: O(n² + nd) — dominated by the n×n matrices S and P
```

For a 4K sequence with FP16: S alone = 4096² × 2 bytes = **32 MB per head per layer**.

---

## FlashAttention Algorithm

### Key Insight: Tiled Online Softmax

FlashAttention never materializes the n×n attention matrix. Instead, it:

1. **Tiles** Q into blocks of rows, K and V into blocks of columns
2. Computes attention **block by block** in SRAM
3. Uses the **online softmax trick** to accumulate the result incrementally

```
Standard:                          FlashAttention:

Q ──┐                              Q blocks:  [Q₁] [Q₂] [Q₃] ...
    ├── S = QK^T (n×n in HBM)                  │
K ──┘     │                        K,V blocks: [K₁,V₁] [K₂,V₂] ...
          ▼                                      │
     P = softmax(S)                For each Qᵢ:
          │                          For each Kⱼ,Vⱼ:
V ────────┤                            Sᵢⱼ = Qᵢ·Kⱼᵀ  (in SRAM!)
          ▼                            Update running softmax
     O = P·V                           Accumulate Oᵢ
                                     Write final Oᵢ to HBM
```

### Online Softmax Trick

The challenge: softmax requires knowing the max over the **entire row** before computing. The online algorithm maintains running statistics:

```
For each block j:
    1. Compute local scores:    Sᵢⱼ = Qᵢ · Kⱼᵀ / √d
    2. Find local max:          mᵢⱼ = max(Sᵢⱼ)
    3. Update global max:       mᵢ_new = max(mᵢ_old, mᵢⱼ)
    4. Rescale old accumulator:  Oᵢ *= exp(mᵢ_old - mᵢ_new)
    5. Accumulate new block:     Oᵢ += exp(Sᵢⱼ - mᵢ_new) · Vⱼ
    6. Update normalizer:        lᵢ = lᵢ * exp(mᵢ_old - mᵢ_new) + row_sum(exp(Sᵢⱼ - mᵢ_new))

Final: Oᵢ = Oᵢ / lᵢ
```

This is mathematically **exact** — no approximation.

### Tiling Diagram

```
         K₁    K₂    K₃    K₄
       ┌─────┬─────┬─────┬─────┐
  Q₁   │  ●  │  ●  │  ●  │  ●  │  ← Process 4 tiles for Q₁, accumulate O₁
       ├─────┼─────┼─────┼─────┤
  Q₂   │  ●  │  ●  │  ●  │  ●  │  ← Process 4 tiles for Q₂, accumulate O₂
       ├─────┼─────┼─────┼─────┤
  Q₃   │  ●  │  ●  │  ●  │  ●  │  ← Process 4 tiles for Q₃, accumulate O₃
       ├─────┼─────┼─────┼─────┤
  Q₄   │  ●  │  ●  │  ●  │  ●  │  ← Process 4 tiles for Q₄, accumulate O₄
       └─────┴─────┴─────┴─────┘

  Each ● = one tile computed entirely in SRAM
  Block size B chosen so Q_block, K_block, V_block fit in SRAM
  Typical: B = 64-256 depending on d and SRAM size
```

### Backward Pass

FlashAttention recomputes S and P from Q, K, V during backprop instead of storing them. This trades compute for memory — a favorable trade since:
- The recomputation is fast (stays in SRAM)
- Saving O(n²) memory is huge for long sequences

---

## Complexity Comparison

| Aspect | Standard Attention | FlashAttention |
|--------|-------------------|----------------|
| Time complexity | O(n²d) | O(n²d) (same) |
| HBM memory | O(n² + nd) | O(nd) — no n×n matrix |
| HBM IO | O(n²d + n²) | O(n²d² / M) where M = SRAM size |
| Wall-clock speed | Baseline | **2-4× faster** |
| Exact? | Yes | Yes (not an approximation) |

The speedup comes from reduced HBM IO, not fewer FLOPs.

---

## FlashAttention-2 Improvements

Key changes over FlashAttention-1:

1. **Better work partitioning**: parallelize over sequence length dimension (not just batch × heads)
2. **Reduced non-matmul FLOPs**: restructure online softmax to minimize non-tensor-core operations
3. **Forward pass**: ~2× faster than FA-1
4. **Causal masking**: skip tiles entirely above the diagonal

```
Causal mask optimization:
       K₁    K₂    K₃    K₄
     ┌─────┬─────┬─────┬─────┐
Q₁   │  ●  │ skip│ skip│ skip│  ← Only 1 tile needed
     ├─────┼─────┼─────┼─────┤
Q₂   │  ●  │  ●  │ skip│ skip│  ← 2 tiles
     ├─────┼─────┼─────┼─────┤
Q₃   │  ●  │  ●  │  ●  │ skip│  ← 3 tiles
     ├─────┼─────┼─────┼─────┤
Q₄   │  ●  │  ●  │  ●  │  ●  │  ← 4 tiles
     └─────┴─────┴─────┴─────┘

Saves ~50% computation for causal attention!
```

---

## FlashAttention-3

Key advances (Hopper architecture, H100):

1. **FP8 support**: 2× throughput with FP8 tensor cores, minimal accuracy loss
2. **Asynchronous execution**: overlap SRAM ↔ HBM transfers with computation using TMA
3. **Warp-specialization**: different warps handle producer (data movement) vs consumer (compute) roles

---

## Linear Attention

Linear attention replaces softmax with decomposable kernels, enabling O(n) sequence modeling. This has grown into a major subfield encompassing state space models, Mamba, and gated delta networks.

| Method | Quality | Speed | Key Idea |
|--------|---------|-------|----------|
| Linear Transformer | Lower | O(nd²) | Kernel trick |
| RWKV | Competitive | O(nd) | Channel decay |
| Mamba | Competitive | O(nd) | Selective SSM |
| Gated DeltaNet | Strong | O(nd) | Gating + delta rule |

**See [Linear Attention & Hybrid Architectures](../linear_attention/00_Linear_Attention.md) for deep coverage** — including SSMs, Mamba, Gated DeltaNet, and hybrid architectures (Qwen 3.5).

---

## Complementary Optimizations

FlashAttention and GQA/MQA solve different problems and combine well:

| Optimization | What it reduces | Type |
|-------------|----------------|------|
| FlashAttention | HBM memory and IO | Compute efficiency |
| GQA/MQA | KV cache size | Memory reduction |
| Quantization | Weight + KV cache memory | Compression |
| Speculative decoding | Decode latency | Parallelism |

```
Full modern stack:
┌──────────────────────────────┐
│  GQA (fewer KV heads)        │  ← Smaller KV cache
├──────────────────────────────┤
│  FlashAttention-2/3          │  ← Faster attention kernel
├──────────────────────────────┤
│  KV cache quantization       │  ← Compress cached KV
├──────────────────────────────┤
│  PagedAttention (vLLM)       │  ← Efficient memory management
└──────────────────────────────┘
```

---

## PyTorch Usage

```python
# FlashAttention is integrated into PyTorch via SDPA
import torch
import torch.nn.functional as F

# Automatic backend selection (uses FlashAttention when possible)
output = F.scaled_dot_product_attention(
    query,   # (batch, heads, seq_len, d_k)
    key,     # (batch, heads, seq_len, d_k)
    value,   # (batch, heads, seq_len, d_k)
    is_causal=True,  # applies causal mask efficiently
)

# Force FlashAttention backend
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
```

Requirements for FlashAttention:
- CUDA GPU (compute capability ≥ 8.0 for FA-2)
- FP16 or BF16 inputs
- Head dimension ≤ 256
- No explicit attention mask (use `is_causal=True` instead)

---

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) | IO-aware tiled attention |
| [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691) | Better parallelism, 2× faster |
| [FlashAttention-3 (Shah et al., 2024)](https://arxiv.org/abs/2407.08691) | FP8, async, Hopper-optimized |
| [Performers (Choromanski et al., 2021)](https://arxiv.org/abs/2009.14794) | Random feature linear attention |
| [Mamba (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752) | Selective state space model |
| [RetNet (Sun et al., 2023)](https://arxiv.org/abs/2307.08621) | Retentive network |

## Related

- [Self-Attention](01_Self_Attention.md) — The base attention operation FlashAttention optimizes
- [MQA & GQA](03_MQA_GQA.md) — Complementary KV cache optimization
- [Autoregressive Decoding](../generation/01_Autoregressive_Decoding.md) — Where attention efficiency matters most
