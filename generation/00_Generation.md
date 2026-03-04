# Generation & Decoding

> Parent: [LLM Algorithms](../00_LLM.md)

## Overview

Generation is how an LLM produces text: given a prompt, the model autoregressively generates tokens one at a time, each conditioned on all previous tokens. The choice of decoding strategy (greedy, sampling, beam search) dramatically affects output quality, diversity, and speed.

## The Generation Pipeline

```
Prompt: "Once upon a"
         │
    ┌────▼──────────────────┐
    │   PREFILL PHASE       │  Process all prompt tokens in parallel
    │   "Once" "upon" "a"   │  Build KV cache
    │   → KV cache created  │  Compute-bound
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │   DECODE PHASE        │  Generate tokens one at a time
    │                       │
    │   Step 1: → "time"    │  Read full KV cache
    │   Step 2: → ","       │  Append to KV cache
    │   Step 3: → "there"   │  Memory-bound
    │   ...                 │
    │   Step N: → <EOS>     │  Stop condition
    └───────────────────────┘
```

## Topics

| Topic | File | Priority |
|-------|------|----------|
| Autoregressive Decoding | [01_Autoregressive_Decoding.md](01_Autoregressive_Decoding.md) | ★★★★★ |
| Sampling Strategies | [02_Sampling.md](02_Sampling.md) | ★★★★☆ |
| Advanced Decoding | [03_Advanced_Decoding.md](03_Advanced_Decoding.md) | ★★★☆☆ |

## Decoding Strategy Comparison

| Strategy | Quality | Diversity | Speed | Use Case |
|----------|---------|-----------|-------|----------|
| Greedy | Medium | None | Fast | Code, math |
| Beam search | High (for short) | Low | Slow | Translation, summarization |
| Top-k sampling | Good | Medium | Fast | General text |
| Top-p (nucleus) | Good | High | Fast | Creative writing |
| Temperature scaling | Adjustable | Adjustable | Fast | Combined with above |
| Speculative decoding | Same as target | Same | 2-3× faster | Any (with draft model) |

## Key Metrics

| Metric | Definition | Typical Target |
|--------|------------|----------------|
| TTFT | Time To First Token (prefill latency) | < 500ms |
| TPOT | Time Per Output Token (decode latency) | < 50ms |
| Throughput | Total tokens/second across all requests | Maximize |

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Holtzman et al. (2020)](https://arxiv.org/abs/1904.09751) | Nucleus (top-p) sampling |
| [Leviathan et al. (2023)](https://arxiv.org/abs/2211.17192) | Speculative decoding |
| [Welleck et al. (2020)](https://arxiv.org/abs/1908.04319) | Analysis of decoding degeneracy |
