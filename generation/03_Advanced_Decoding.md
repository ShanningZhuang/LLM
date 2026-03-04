# Advanced Decoding

> Parent: [Generation](00_Generation.md)

## Overview

Beyond simple sampling, advanced decoding strategies optimize for quality, speed, or structured output. Beam search finds high-probability sequences, speculative decoding accelerates generation without changing the output distribution, and constrained decoding forces outputs to follow specific formats.

---

## Beam Search

### Algorithm

Maintain the top-b (beam width) most likely partial sequences at each step:

```
Step 0: "The"
        │
Step 1: ├── "The cat"   (log p = -2.1)
        ├── "The dog"   (log p = -2.3)
        └── "The bird"  (log p = -3.1)    ← beam width = 3
        │
Step 2: ├── "The cat sat"    (log p = -4.2)  ← from "The cat"
        ├── "The cat is"     (log p = -4.5)  ← from "The cat"
        ├── "The dog ran"    (log p = -4.3)  ← from "The dog"
        │   (dropped: "The bird..." — not in top 3)
        │
Step 3: ...continue until EOS for all beams
```

### Scoring

```
Raw score:           score(y) = Σ log P(yₜ | y<ₜ)
                     Problem: shorter sequences always score higher

Length normalization: score(y) = (1/|y|^α) × Σ log P(yₜ | y<ₜ)
                     α ∈ [0.6, 1.0], typically α = 0.6

Coverage penalty:    Encourages attending to all source tokens (for translation)
```

### When to Use Beam Search

| Use Case | Beam Search? | Why |
|----------|-------------|-----|
| Translation | Yes | Need most likely translation |
| Summarization | Yes | Factual accuracy matters |
| Code generation | Sometimes | Correctness over diversity |
| Chat / dialogue | **No** | Too repetitive, lacks diversity |
| Creative writing | **No** | Need randomness |

### Limitations

- Computationally expensive: O(b × V × n) per sequence
- Produces generic, repetitive text for open-ended generation
- "Beam search curse": higher beam width can produce worse text (Holtzman et al.)

---

## Speculative Decoding

### Motivation

Autoregressive decoding is memory-bound: GPU is underutilized because it loads the entire model to produce just one token. Can we generate multiple tokens per forward pass?

### Algorithm

Use a small, fast **draft model** to speculate, then **verify** with the large target model:

```
Draft Model (small, fast)          Target Model (large, accurate)
┌──────────────────┐               ┌──────────────────────┐
│ Generate k tokens │               │ Verify all k tokens   │
│ quickly (~5ms/tok)│               │ in ONE forward pass    │
│                   │               │ (~50ms total)          │
│ "The" → draft:    │               │                        │
│   "cat sat on"    │──── send ────▶│ Check: P_target(tᵢ)   │
│   (3 tokens)      │               │ vs P_draft(tᵢ)        │
│                   │               │                        │
│                   │◀── result ────│ Accept: "cat sat"      │
│                   │               │ Reject: "on" → "the"   │
└──────────────────┘               └──────────────────────┘

Result: Generated "cat sat the" (3 tokens) in ~65ms instead of ~150ms
```

### Acceptance Criterion

For each draft token tᵢ, accept with probability:

```
P_accept(tᵢ) = min(1, P_target(tᵢ) / P_draft(tᵢ))
```

If rejected at position i:
1. Discard tokens from position i onward
2. Sample a correction token from an adjusted distribution:
   `P_corrected(t) = max(0, P_target(t) - P_draft(t))` (normalized)

**Key guarantee**: The output distribution is **identical** to sampling from the target model alone.

### Speedup Analysis

```
Without speculative decoding:
    k tokens = k forward passes of target model
    Time: k × T_target

With speculative decoding (draft k tokens, accept α fraction on average):
    Time ≈ k × T_draft + 1 × T_target
    Effective tokens = 1 + α × k  (on average)
    Speedup ≈ (1 + αk) / (1 + k × T_draft/T_target)

Typical: α ≈ 0.7-0.9, k = 3-5, speedup ≈ 2-3×
```

### When Speculative Decoding Works Best

| Factor | Good for Spec. Decoding | Bad for Spec. Decoding |
|--------|------------------------|----------------------|
| Draft model quality | Close to target | Very different from target |
| Domain | Well-covered by draft | Specialized/rare domain |
| Temperature | Low (deterministic) | High (random) |
| Output length | Long sequences | Very short responses |

### Variants

| Variant | Approach |
|---------|----------|
| Standard | Separate draft model (e.g., 125M for 7B target) |
| Self-speculative | Use early exit from target model as draft |
| Medusa | Add extra prediction heads to target model |
| EAGLE | Feature-level prediction, not token-level |
| Lookahead | Jacobi iteration, no draft model |

### PyTorch Pseudocode

```python
def speculative_decode(target_model, draft_model, prompt, k=5, max_tokens=100):
    tokens = prompt.clone()
    kv_cache_target = None
    kv_cache_draft = None

    while len(tokens) < max_tokens:
        # Step 1: Draft k tokens with small model
        draft_tokens = []
        draft_probs = []
        x = tokens
        for _ in range(k):
            logits, kv_cache_draft = draft_model(x[:, -1:], kv_cache_draft)
            prob = F.softmax(logits[:, -1], dim=-1)
            token = torch.multinomial(prob, 1)
            draft_tokens.append(token)
            draft_probs.append(prob[0, token.item()])
            x = torch.cat([x, token], dim=1)

        # Step 2: Verify all k tokens with target model in ONE pass
        draft_sequence = torch.cat(draft_tokens, dim=1)
        all_logits, kv_cache_target = target_model(
            torch.cat([tokens[:, -1:], draft_sequence], dim=1),
            kv_cache_target
        )

        # Step 3: Accept/reject each token
        n_accepted = 0
        for i in range(k):
            target_prob = F.softmax(all_logits[:, i], dim=-1)
            p_target = target_prob[0, draft_tokens[i].item()]
            p_draft = draft_probs[i]

            # Accept with probability min(1, p_target/p_draft)
            if torch.rand(1) < (p_target / p_draft):
                n_accepted += 1
            else:
                # Reject: sample correction from adjusted distribution
                adjusted = torch.clamp(target_prob - draft_probs_full[i], min=0)
                adjusted = adjusted / adjusted.sum()
                correction = torch.multinomial(adjusted, 1)
                tokens = torch.cat([tokens, draft_tokens[:i], correction], dim=1)
                break

        if n_accepted == k:
            # All accepted — also sample one bonus token from target
            bonus_prob = F.softmax(all_logits[:, k], dim=-1)
            bonus = torch.multinomial(bonus_prob, 1)
            tokens = torch.cat([tokens, draft_sequence, bonus], dim=1)

    return tokens
```

---

## Constrained / Structured Generation

### Problem

LLMs generate free-form text, but many applications need structured output:
- Valid JSON for API responses
- SQL queries that parse correctly
- Code that compiles
- Specific formats (dates, emails, etc.)

### Approach: Logit Masking

At each step, compute which tokens are valid given the grammar, and mask invalid tokens:

```
Grammar: JSON object
Current partial output: '{"name": "Al'

Valid next tokens:    [a-z, A-Z, ", \, etc.]
Invalid next tokens:  [}, {, [, numbers at this position, etc.]

logits[invalid_tokens] = -inf
→ softmax guarantees only valid tokens are sampled
```

### Tools

| Tool | Approach | Use Case |
|------|----------|----------|
| Outlines | Finite state machine on regex/grammar | JSON, regex patterns |
| Guidance (Microsoft) | Template-based with constraints | Structured prompts |
| LMQL | Query language for LLMs | Complex constraints |
| llama.cpp grammars | GBNF grammar support | Local inference |
| SGLang | Regex/JSON constraints built-in | Serving |

### Example: JSON Constrained Generation

```python
# Using Outlines
from outlines import models, generate

model = models.transformers("meta-llama/Llama-2-7b")

# Generate valid JSON matching a schema
generator = generate.json(model, schema={
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"]
})

result = generator("Extract info: John is 25 years old")
# Always valid JSON: {"name": "John", "age": 25}
```

---

## Contrastive Decoding

Subtract the log-probabilities of a small "amateur" model from the large "expert" model:

```
score(t) = log P_expert(t) - β × log P_amateur(t)

Intuition: tokens that the expert likes BUT the amateur doesn't
           → more interesting, less generic text
```

Reduces repetitive, generic text while maintaining coherence.

---

## Decoding Strategy Comparison

| Strategy | Quality | Diversity | Speed | Memory | Best For |
|----------|---------|-----------|-------|--------|----------|
| Greedy | Medium | None | 1× | 1× | Deterministic tasks |
| Beam search (b=4) | High | Low | 0.25× | 4× | Translation, summarization |
| Top-p sampling | Good | High | 1× | 1× | Chat, creative writing |
| Speculative (k=5) | Same as target | Same | 2-3× | +draft model | Any (with good draft) |
| Constrained | N/A | Limited | ~0.8× | 1× | Structured output |
| Contrastive | Good | Medium | 0.5× | 2× (two models) | Reducing repetition |

---

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Speculative Decoding (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192) | Draft-verify paradigm |
| [Medusa (Cai et al., 2024)](https://arxiv.org/abs/2401.10774) | Multiple prediction heads |
| [EAGLE (Li et al., 2024)](https://arxiv.org/abs/2401.15077) | Feature-level speculation |
| [Contrastive Decoding (Li et al., 2023)](https://arxiv.org/abs/2210.15097) | Expert-amateur contrast |
| [Outlines (Willard & Louf, 2023)](https://arxiv.org/abs/2307.09702) | FSM-guided generation |
| [Lookahead Decoding (Fu et al., 2024)](https://arxiv.org/abs/2402.02057) | Jacobi iteration decoding |

## Related

- [Autoregressive Decoding](01_Autoregressive_Decoding.md) — Foundation that all these methods build on
- [Sampling Strategies](02_Sampling.md) — Basic sampling methods
- [Efficient Attention](../attention/05_Efficient_Attention.md) — FlashAttention for faster attention computation
