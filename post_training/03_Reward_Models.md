# Reward Models

> Parent: [Post-Training & Alignment](00_Post_Training.md)

## Overview

The reward model (RM) is the bridge between human preferences and the RL training loop.
It takes a **(prompt, response)** pair and outputs a **scalar score**: higher means better.
Training a good reward model is arguably the most important step in RLHF -- a bad RM
teaches the policy bad behavior ("garbage in, garbage out").

Concretely, the reward model answers one question:

> "Given this prompt and this response, how good is the response on a single-number scale?"

The RL algorithm (e.g., PPO) then uses that score as the reward signal to update the
language model's weights. Every flaw in the reward model is amplified by the policy
optimizer, so RM quality upper-bounds alignment quality.

The typical training recipe is:

1. Collect **human preference data** (comparisons between responses).
2. Train a reward model on those comparisons using the **Bradley-Terry** loss.
3. Use the trained RM to provide rewards during **PPO** (or another RL algorithm).

---

## Bradley-Terry Model

### The Core Question

Given a prompt **x** and two candidate responses **y_A** and **y_B**, what is the
probability that a human prefers A over B?

The **Bradley-Terry** model answers this with a simple formula:

```
P(A > B | x) = sigma( r(x, y_A) - r(x, y_B) )
```

Breaking down every symbol:

| Symbol | Meaning |
|--------|---------|
| `P(A > B \| x)` | Probability that response A is preferred over B, given prompt x |
| `sigma(z)` | The sigmoid function: `sigma(z) = 1 / (1 + e^(-z))` |
| `r(x, y_A)` | The reward model's scalar score for response A given prompt x |
| `r(x, y_B)` | The reward model's scalar score for response B given prompt x |

The key insight: the preference probability depends **only on the difference** in scores,
not their absolute values. `r(A) = 5.0, r(B) = 4.0` gives the same probability as
`r(A) = 100.0, r(B) = 99.0`.

### Numerical Examples

Recall that `sigma(z) = 1 / (1 + e^(-z))`.

| r(A) | r(B) | Difference | sigma(diff) | Interpretation |
|------|------|-----------|-------------|----------------|
| 2.0 | 1.0 | 1.0 | **0.73** | A is likely preferred |
| 2.0 | 2.0 | 0.0 | **0.50** | Coin flip -- no preference |
| 1.0 | 3.0 | -2.0 | **0.12** | B is strongly preferred |
| 5.0 | 0.0 | 5.0 | **0.99** | A is almost certainly preferred |

### Deriving the Loss Function

We have a dataset of human preferences: triples of `(x, y_chosen, y_rejected)`.
We want the reward model (parameterized by phi) to assign a **higher** score to the
chosen response. The Bradley-Terry loss is:

```
L(phi) = -E[ log sigma( r_phi(x, y_chosen) - r_phi(x, y_rejected) ) ]
```

Step-by-step:

1. Compute `r_phi(x, y_chosen)` -- the reward for the chosen response.
2. Compute `r_phi(x, y_rejected)` -- the reward for the rejected response.
3. Take their difference: `delta = r_chosen - r_rejected`.
4. Pass through sigmoid: `sigma(delta)` -- this is the model's predicted probability
   that the chosen response is preferred.
5. Take `log` -- log-probability.
6. Negate and average over the dataset -- standard negative log-likelihood.

In plain English: "Maximize the log-probability that the model ranks chosen above
rejected for every pair in the dataset."

---

## Reward Model Architecture

A reward model is typically a pretrained LLM with one modification: the language model
head (which predicts next tokens) is replaced by a **scalar projection head**.

```
Input: [prompt tokens] [response tokens] [EOS]
         |
         v
+-----------------------------------+
|   LLM Backbone (e.g., LLaMA-7B)  |
|   (can be frozen or fine-tuned)   |
+-----------------------------------+
         |
         v
  Last hidden state at [EOS] token
  (contains info about entire sequence)
         |
         v
  Linear(hidden_dim -> 1)
         |
         v
  Scalar reward: 0.73
```

**Why the EOS token?** In causal (autoregressive) attention, each token only attends to
tokens that came before it. The `[EOS]` token is the last token, so its hidden state has
"seen" the entire prompt + response through the causal attention chain. It acts as a
natural summary of the full sequence.

### Minimal PyTorch Implementation

```python
class RewardModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone          # e.g., LlamaModel (no LM head)
        self.reward_head = nn.Linear(
            backbone.config.hidden_size, 1 # project hidden dim -> scalar
        )

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(
            input_ids, attention_mask
        ).last_hidden_state               # (batch, seq_len, hidden_dim)
        # Find the position of the last real token (EOS) per example
        eos_idx = attention_mask.sum(dim=1) - 1
        eos_hidden = hidden[range(len(eos_idx)), eos_idx]  # (batch, hidden_dim)
        return self.reward_head(eos_hidden).squeeze(-1)     # (batch,)
```

Common backbone choices: start from the same pretrained model as the policy (e.g.,
LLaMA-7B), or use a smaller model (e.g., LLaMA-3B) for faster reward inference.
The backbone can be fully fine-tuned or kept frozen with only the reward head trained
(less expressive but faster and more stable).

---

## Human Feedback Pipeline

### Step-by-Step Process

1. **Sample prompts** from a prompt distribution (user queries, red-team prompts, etc.).
2. **Generate K responses** per prompt using the current policy (K = 4 is common).
3. **Human annotators rank** the K responses, e.g.: `C > A > D > B`.
4. **Convert ranking to pairwise comparisons**:
   - From the ranking `C > A > D > B`, extract all ordered pairs:
   - `(C, A)`, `(C, D)`, `(C, B)`, `(A, D)`, `(A, B)`, `(D, B)`
   - One ranking of K items yields **K(K-1)/2** pairs.
   - For K = 4: `4 * 3 / 2 = 6` pairs from a single ranking.
5. **Train the RM** on all pairs using the Bradley-Terry loss.

### Why Rankings Instead of Scores?

Asking humans "rate this response 1-10" produces **noisy, inconsistent** scores.
Different annotators have different scales. Asking "which is better, A or B?" is a much
easier question for humans, and **pairwise comparisons are far more consistent**
across annotators. The Bradley-Terry model converts these relative comparisons into
absolute scores automatically.

### Scale

InstructGPT used ~33K prompts with 4-9 responses each, producing hundreds of thousands
of comparison pairs. Modern systems use even larger datasets, sometimes augmented with
AI-generated comparisons (see Constitutional AI).

---

## Process vs Outcome Reward Models

### ORM (Outcome Reward Model)

Score the **final answer only**. The entire response gets one reward at the end.

```
Prompt:  "What is 15 * 7?"
Response A: "15 * 7 = 105"   -> reward = 1.0  (correct)
Response B: "15 * 7 = 106"   -> reward = 0.0  (incorrect)
```

The reward is **sparse**: if the response has 50 tokens, only the final position gets a
meaningful signal. The model has to figure out on its own which intermediate step caused
the error.

### PRM (Process Reward Model)

Score **each reasoning step** individually.

```
Prompt: "What is 15 * 7?"
  Step 1: "15 * 7 = 15 * (5 + 2)"   -> reward = 0.9 (correct decomposition)
  Step 2: "= 75 + 30"               -> reward = 0.9 (correct arithmetic)
  Step 3: "= 105"                    -> reward = 1.0 (correct final answer)
```

The reward is **dense**: every step gets feedback. Much better for math and reasoning
tasks because the model learns *which step* went wrong.

### Comparison

| Feature | ORM | PRM |
|---------|-----|-----|
| Annotation cost | Low (just final answer) | High (per-step labels) |
| Signal density | Sparse (end only) | Dense (every step) |
| Best for | General chat, simple tasks | Math, code, reasoning |
| Credit assignment | Poor (which token caused error?) | Good (which step was wrong?) |
| Training data | Easy to collect at scale | Requires expert annotators |

Lightman et al. (2023) showed that PRMs significantly outperform ORMs on math
benchmarks, because dense step-level feedback enables much better credit assignment.

---

## Reward Hacking

**Reward hacking** occurs when the policy finds ways to achieve high reward without
actually producing better responses. It exploits weaknesses in the reward model rather
than genuinely improving.

### Common Examples

- **Verbosity**: RM gives slightly higher scores to longer responses. The policy learns
  to pad answers with unnecessary filler until responses are absurdly long.
- **Repetition**: Certain phrases (e.g., "I'd be happy to help!") score high. The model
  repeats them excessively.
- **Sycophancy**: RM prefers agreeable responses. The model learns to agree with the
  user even when the user is factually wrong.
- **Formatting tricks**: Bullet points, headers, and bold text score higher. The model
  over-formats simple answers that don't need structure.

### Mitigations

| Strategy | How It Helps |
|----------|-------------|
| KL penalty (beta) | Prevents the policy from diverging too far from the reference model |
| RM ensembles | Train multiple RMs; harder to hack all simultaneously |
| Reward normalization | Prevent reward scale drift during training |
| Diverse training data | RM learns robust preferences, not surface-level shortcuts |
| Length penalty | Directly penalize excessive verbosity in the reward |
| Periodic RM retraining | Update the RM as the policy distribution shifts |

Gao et al. (2023) showed that reward hacking follows predictable **scaling laws**:
as you optimize more against the RM (higher KL divergence), the true reward first
increases, then **decreases** -- the policy over-optimizes past the point of genuine
improvement. This means there is an optimal amount of RL training, and more is not
always better.

---

## Key Papers

- [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) -- Full RLHF pipeline with reward model training on human comparisons
- [Let's Verify Step by Step (Lightman et al., 2023)](https://arxiv.org/abs/2305.20050) -- Process Reward Models for mathematical reasoning
- [Scaling Laws for Reward Model Overoptimization (Gao et al., 2023)](https://arxiv.org/abs/2210.10760) -- Quantitative analysis of reward hacking dynamics
- [Constitutional AI (Bai et al., 2022)](https://arxiv.org/abs/2212.08073) -- Using AI feedback instead of human feedback to train reward models

## Related

- [PPO for LLMs](02_PPO.md) -- How the reward model feeds into PPO training
- [DPO & Variants](04_DPO_and_Variants.md) -- DPO eliminates the reward model entirely
- [AI_Infra RL Notes](../../AI_Infra/rl/04_Note.md) -- Extended reward model architecture analysis
