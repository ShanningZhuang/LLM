# Sampling Strategies

> Parent: [Generation](00_Generation.md)

## Overview

After the model produces a logit vector over the vocabulary, a **sampling strategy** decides which token to actually pick. The choice of strategy dramatically affects output quality: greedy decoding is deterministic but repetitive, while pure random sampling is diverse but incoherent. The practical sweet spot lies in **controlled randomness** -- temperature scaling combined with top-p or min-p filtering.

## From Logits to Tokens: The Sampling Pipeline

```
Model output: logits ∈ R^vocab_size   (raw, unnormalized scores)
                │
                ▼
┌───────────────────────────────┐
│  1. Repetition penalty        │  Reduce logits of recently seen tokens
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  2. Temperature scaling       │  logits = logits / T
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  3. Top-k filtering           │  Keep only top k logits, set rest to -inf
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  4. Top-p (nucleus) filtering │  Keep smallest set with cumprob >= p
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  5. Softmax                   │  probs = softmax(filtered_logits)
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  6. Multinomial sample        │  token = sample(probs)
└───────────────────────────────┘

Note: Not all steps are used simultaneously.
      Common combo: temperature + top-p.
      Order matters: penalties first, then temperature, then filtering.
```

---

## Greedy Decoding

The simplest strategy: always pick the highest-probability token.

```
token = argmax(logits)
```

**Properties:**
- Deterministic: same prompt always gives same output
- Fast: no sampling overhead
- Problem: leads to repetitive, generic text for open-ended generation
- Works well for: code completion, factual Q&A, math

```python
import torch

def greedy_decode(logits: torch.Tensor) -> int:
    """Select the token with the highest logit."""
    # logits: (vocab_size,)
    return logits.argmax(dim=-1).item()
```

### The Degeneration Problem

Greedy decoding (and high-probability sampling in general) tends to produce degenerate text:

```
Prompt: "The meaning of life is"

Greedy: "The meaning of life is to be happy. The meaning of life is to be
         happy. The meaning of life is to be happy. The meaning of ..."

         → falls into repetition loops
```

This happens because the highest-probability next token is often a "safe" continuation that leads the model into a cycle (Holtzman et al., 2020).

---

## Temperature Scaling

Temperature controls the sharpness of the probability distribution:

```
p_i = softmax(logits / T)_i = exp(logit_i / T) / Σ_j exp(logit_j / T)
```

```
Effect of temperature on distribution:

  T = 0.1 (very sharp)     T = 1.0 (default)      T = 2.0 (flat)
  prob                      prob                    prob
  ▲                         ▲                       ▲
  │█                        │                       │
  │█                        │█                      │▓▓
  │█                        │█▓                     │▓▓▓▓
  │█                        │█▓▓                    │▓▓▓▓▓▓
  │█░                       │█▓▓░░░                 │▓▓▓▓▓▓▓▓░░░░░
  └──────── tokens          └──────── tokens        └──────────── tokens
  Almost deterministic      Moderate diversity      Very random
```

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| T -> 0 | Equivalent to greedy (argmax) | N/A (degenerate) |
| T = 0.1-0.3 | Very focused, near-deterministic | Code, math, factual |
| T = 0.7-0.9 | Balanced diversity | General chat, writing |
| T = 1.0 | Model's original distribution | Default / baseline |
| T = 1.2-2.0 | High diversity, some incoherence | Brainstorming |
| T -> inf | Uniform random over vocabulary | N/A (gibberish) |

```python
import torch
import torch.nn.functional as F

def temperature_sample(logits: torch.Tensor, temperature: float = 1.0) -> int:
    """Sample a token with temperature scaling."""
    # logits: (vocab_size,)
    if temperature == 0:
        return logits.argmax(dim=-1).item()  # Greedy

    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
```

---

## Top-k Sampling

**Fan et al. (2018):** Keep only the top-k highest-probability tokens, set everything else to zero probability, then redistribute.

```
Algorithm:
  1. Sort tokens by logit value (descending)
  2. Keep top k tokens
  3. Set all other logits to -inf
  4. Apply softmax to get probabilities
  5. Sample from the k remaining tokens
```

```
Original distribution (10 tokens shown):

  Token:  A    B    C    D    E    F    G    H    I    J
  Prob:  0.35 0.20 0.15 0.10 0.08 0.05 0.03 0.02 0.01 0.01

After top-k (k=4):

  Token:  A    B    C    D    E    F    G    H    I    J
  Prob:  0.44 0.25 0.19 0.12  0    0    0    0    0    0
                              ↑
                        Redistributed to sum to 1.0
```

**Properties:**
- k=1: equivalent to greedy
- k=V (vocab size): equivalent to full sampling
- Problem: fixed k regardless of distribution shape

```
Why fixed k is a problem:

Confident prediction (most mass on few tokens):
  Prob: [0.90, 0.05, 0.03, 0.01, 0.005, ...]
  k=50 → includes many garbage tokens (0.001 or less)!

Uncertain prediction (mass spread over many tokens):
  Prob: [0.05, 0.04, 0.04, 0.03, 0.03, 0.03, ...]
  k=50 → might still miss reasonable tokens
```

```python
import torch
import torch.nn.functional as F

def top_k_sample(logits: torch.Tensor, k: int = 50, temperature: float = 1.0) -> int:
    """Sample from the top-k highest probability tokens."""
    # logits: (vocab_size,)
    scaled_logits = logits / temperature

    # Keep only top-k, set rest to -inf
    top_k_logits, top_k_indices = torch.topk(scaled_logits, k)
    # top_k_logits: (k,)  top_k_indices: (k,)

    # Create filtered logits (everything else is -inf)
    filtered_logits = torch.full_like(scaled_logits, float('-inf'))
    filtered_logits.scatter_(0, top_k_indices, top_k_logits)

    probs = F.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
```

---

## Top-p / Nucleus Sampling

**Holtzman et al. (2020):** Instead of a fixed number of tokens, keep the **smallest** set of tokens whose cumulative probability is at least p.

```
Algorithm:
  1. Sort tokens by probability (descending)
  2. Compute cumulative sum of probabilities
  3. Find the cutoff: smallest set where cumsum >= p
  4. Zero out everything below the cutoff
  5. Redistribute and sample
```

```
Example with p = 0.9:

  Token:  A     B     C     D     E     F     G    ...
  Prob:  0.35  0.20  0.15  0.10  0.08  0.05  0.03 ...
  Cum:   0.35  0.55  0.70  0.80  0.88  0.93  ...
                                        ↑
                                  0.93 >= 0.9  → cut here

  Keep: {A, B, C, D, E, F}  (6 tokens, cumprob = 0.93)
  Redistribute to sum to 1.0:
  A=0.376, B=0.215, C=0.161, D=0.108, E=0.086, F=0.054
```

### Why Top-p Adapts to the Distribution

```
Confident prediction: "The capital of France is ___"
  Prob: [Paris=0.92, Lyon=0.03, Marseille=0.02, ...]
  Top-p (p=0.9): keeps only {Paris} → 1 token
  Top-k (k=50):  keeps 50 tokens including nonsense!

Uncertain prediction: "I think the answer might be ___"
  Prob: [yes=0.08, no=0.07, maybe=0.06, probably=0.05, ...]
  Top-p (p=0.9): keeps ~20-30 tokens → broad sampling
  Top-k (k=5):   keeps only 5 tokens → too restrictive!
```

Top-p automatically uses fewer tokens when the model is confident and more tokens when it is uncertain.

```python
import torch
import torch.nn.functional as F

def top_p_sample(
    logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0
) -> int:
    """Nucleus sampling: keep smallest token set with cumulative prob >= p."""
    # logits: (vocab_size,)
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    # Sort in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find cutoff: tokens where cumulative prob EXCEEDS p
    # Shift right by 1 so the token that crosses p is kept
    sorted_mask = cumulative_probs - sorted_probs >= p  # True = remove
    sorted_probs[sorted_mask] = 0.0

    # Redistribute
    sorted_probs /= sorted_probs.sum()

    # Sample from filtered distribution
    token_idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices[token_idx].item()
```

---

## Min-p Sampling

A simpler adaptive method: keep all tokens whose probability is at least `min_p` times the maximum token probability.

```
Algorithm:
  1. Find p_max = max(probs)
  2. Threshold = p_max * min_p
  3. Keep all tokens with prob >= threshold
  4. Redistribute and sample
```

```
Example with min_p = 0.1:

  Token:  A     B     C     D     E     F     G    ...
  Prob:  0.35  0.20  0.15  0.10  0.08  0.05  0.03 ...
         ↑
       p_max = 0.35
       threshold = 0.35 * 0.1 = 0.035

  Keep: {A, B, C, D, E, F}  (all with prob >= 0.035)
  Remove: {G, ...}           (prob < 0.035)
```

**Advantages over top-p:**
- Simpler to implement and understand
- Also adaptive: high-confidence predictions keep fewer tokens
- No need to sort and compute cumulative sums
- Behavior is more intuitive: "keep tokens at least 10% as likely as the best"

```python
import torch
import torch.nn.functional as F

def min_p_sample(
    logits: torch.Tensor, min_p: float = 0.1, temperature: float = 1.0
) -> int:
    """Min-p sampling: keep tokens with prob >= min_p * max_prob."""
    # logits: (vocab_size,)
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    # Compute threshold
    p_max = probs.max()
    threshold = p_max * min_p

    # Zero out tokens below threshold
    filtered_probs = probs.clone()
    filtered_probs[probs < threshold] = 0.0

    # Redistribute
    filtered_probs /= filtered_probs.sum()

    return torch.multinomial(filtered_probs, num_samples=1).item()
```

---

## Repetition Penalty

Prevents the model from falling into degenerate repetition loops by reducing the probability of recently generated tokens.

```
For each token i that appears in the recent context:
  if logit_i > 0:
      logit_i = logit_i / alpha      (reduce positive logits)
  else:
      logit_i = logit_i * alpha      (make negative logits more negative)

where alpha > 1.0 is the penalty factor (typical: 1.1 - 1.3)
```

**Variants:**
- **Repetition penalty** (Keskar et al., 2019): penalize all tokens that appeared anywhere in the context
- **Frequency penalty**: scale penalty by how many times the token appeared
- **Presence penalty**: flat penalty if the token appeared at all (binary)

```python
import torch

def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list[int],
    penalty: float = 1.2,
) -> torch.Tensor:
    """
    Apply repetition penalty to logits for tokens that appear
    in the generated sequence.

    Args:
        logits: (vocab_size,) raw logits
        generated_ids: list of previously generated token IDs
        penalty: multiplicative penalty factor (> 1.0)

    Returns:
        Modified logits
    """
    logits = logits.clone()

    # Get unique token IDs that have been generated
    penalized_ids = torch.tensor(list(set(generated_ids)), dtype=torch.long)

    # Penalize: divide positive logits, multiply negative logits
    penalized_logits = logits[penalized_ids]
    logits[penalized_ids] = torch.where(
        penalized_logits > 0,
        penalized_logits / penalty,
        penalized_logits * penalty,
    )

    return logits
```

---

## Combining Strategies

In practice, multiple strategies are combined. The most common combination is **temperature + top-p**:

```python
import torch
import torch.nn.functional as F

def sample_token(
    logits: torch.Tensor,
    generated_ids: list[int] = None,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 0,        # 0 = disabled
    min_p: float = 0.0,    # 0 = disabled
    repetition_penalty: float = 1.0,
) -> int:
    """
    Full sampling pipeline combining multiple strategies.

    Typical configurations:
      Creative writing: temperature=0.9, top_p=0.95
      Code generation:  temperature=0.2, top_p=0.95
      Factual Q&A:      temperature=0.0 (greedy)
      Chat:             temperature=0.7, top_p=0.9
    """
    # logits: (vocab_size,)

    # Step 1: Repetition penalty
    if repetition_penalty != 1.0 and generated_ids:
        logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    # Step 2: Temperature
    if temperature == 0:
        return logits.argmax(dim=-1).item()
    logits = logits / temperature

    # Step 3: Top-k filtering (if enabled)
    if top_k > 0:
        top_k_values, _ = torch.topk(logits, top_k)
        min_top_k = top_k_values[-1]
        logits[logits < min_top_k] = float('-inf')

    # Step 4: Top-p filtering (if enabled)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens above the threshold
        sorted_mask = (cumulative_probs - F.softmax(sorted_logits, dim=-1)) >= top_p
        sorted_logits[sorted_mask] = float('-inf')

        # Unsort
        logits = sorted_logits.gather(0, sorted_indices.argsort())

    # Step 5: Min-p filtering (if enabled)
    if min_p > 0.0:
        probs = F.softmax(logits, dim=-1)
        threshold = probs.max() * min_p
        logits[probs < threshold] = float('-inf')

    # Step 6: Sample
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
```

---

## Probability Distribution Visualization

```
Logits → Softmax → Probability distribution over vocabulary:

Original distribution (vocab_size tokens, sorted):
prob
0.25 │█
0.20 │█ █
0.15 │█ █ █
0.10 │█ █ █ █ █
0.05 │█ █ █ █ █ █ █ █ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ...
0.00 └──────────────────────────────────────────────────────
     Tokens (sorted by probability, left = highest)

After top-k (k=5):                After top-p (p=0.9):
prob                              prob
0.30 │█                           0.28 │█
0.24 │█ █                         0.22 │█ █
0.18 │█ █ █                       0.17 │█ █ █
0.15 │█ █ █ █                     0.12 │█ █ █ █ █
0.13 │█ █ █ █ █ 0  0  0  0       0.06 │█ █ █ █ █ █ █ █  0  0  0  0 ...
0.00 └─────────────────────       0.00 └──────────────────────────────
     Fixed: always 5 tokens            Adaptive: 8 tokens here
                                       (fewer when confident, more when not)
```

---

## Strategy Comparison

| Strategy | Deterministic? | Adaptive? | Key Parameter | Failure Mode | Best For |
|----------|---------------|-----------|---------------|--------------|----------|
| Greedy | Yes | N/A | None | Repetition loops | Math, code, factual |
| Temperature | No | No | T (0.0-2.0) | Incoherent at high T | Always (combined) |
| Top-k | No | No | k (1-100) | Fixed cutoff misses context | Simple applications |
| Top-p (nucleus) | No | Yes | p (0.9-0.99) | None significant | General default |
| Min-p | No | Yes | min_p (0.05-0.2) | None significant | Alternative to top-p |
| Repetition penalty | N/A (modifier) | N/A | alpha (1.0-1.5) | Unnatural avoidance | Long-form text |

### Common Configurations

| Use Case | Temperature | Top-p | Top-k | Repetition Penalty |
|----------|-------------|-------|-------|-------------------|
| Code generation | 0.0-0.2 | 0.95 | -- | 1.0 |
| Factual Q&A | 0.0 | -- | -- | 1.0 |
| Creative writing | 0.8-1.0 | 0.95 | -- | 1.1-1.2 |
| Chatbot | 0.7 | 0.9 | -- | 1.0-1.1 |
| Brainstorming | 1.0-1.2 | 0.99 | -- | 1.0 |
| Translation | 0.3 | 0.9 | -- | 1.0 |

---

## Key Takeaways

1. **Greedy** is deterministic and works for constrained tasks but causes repetition in open-ended generation
2. **Temperature** controls randomness: lower is more focused, higher is more diverse
3. **Top-k** is simple but uses a fixed cutoff that does not adapt to the distribution shape
4. **Top-p (nucleus)** adapts to distribution confidence -- the standard choice in practice
5. **Min-p** is a simpler adaptive alternative to top-p with similar behavior
6. **Repetition penalty** prevents degenerate loops in long-form generation
7. **Combine strategies:** temperature + top-p is the most common production configuration

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Fan et al. (2018)](https://arxiv.org/abs/1805.04833) | Top-k sampling for open-ended text generation |
| [Holtzman et al. (2020)](https://arxiv.org/abs/1904.09751) | Nucleus (top-p) sampling, analysis of degeneration |
| [Keskar et al. (2019)](https://arxiv.org/abs/1909.05858) | CTRL: repetition penalty and control codes |

## Related

- [Autoregressive Decoding](01_Autoregressive_Decoding.md) -- the generation loop that calls the sampler
- [Advanced Decoding](03_Advanced_Decoding.md) -- beam search, speculative decoding, constrained generation
