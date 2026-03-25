# GRPO & Modern Methods

> Parent: [Post-Training & Alignment](00_Post_Training.md)

## Overview

GRPO (Group Relative Policy Optimization) is the algorithm behind DeepSeek's reasoning models. The key insight is deceptively simple: **replace the expensive value model (V_ψ) with group statistics**. Instead of training a separate neural network to estimate how good a state is, GRPO generates multiple responses per prompt, computes the mean and standard deviation of their rewards within the group, and uses that as the baseline.

This means:
- No value model to train or store in memory
- Fewer models running during training (just policy + reference + reward, not policy + reference + reward + value)
- Simpler implementation with fewer moving parts
- Comparable or better performance, especially for reasoning tasks

DeepSeek-R1 demonstrated that GRPO, combined with rule-based rewards, can produce chain-of-thought reasoning that **emerges spontaneously** — without ever being explicitly taught.

## The Value Function Problem

In standard PPO for LLMs, the advantage function measures "how much better was this action than expected":

```
A(s, a) = Q(s, a) - V(s)
```

The value function V_ψ(s) is a separate neural network (often a full LLM-sized model) that estimates the expected total reward from a given state. It answers: "On average, how much reward will we get from here?"

The problems with learning V_ψ:

1. **Memory**: V_ψ is often the same size as the policy — so you need ~2x the GPU memory just for this extra model
2. **Training instability**: V_ψ has its own loss function and its own optimization dynamics. If V_ψ learns poorly, the advantages are noisy, and the policy update suffers
3. **Bootstrapping errors**: V_ψ estimates are used to compute targets for V_ψ itself — circular reasoning that can compound errors
4. **Complexity**: You now have two models to tune, two learning rates, two sets of hyperparameters

The question GRPO asks: **Can we compute advantages WITHOUT learning V?**

The answer is yes — if we're willing to generate multiple responses per prompt.

## GRPO Key Insight

For each prompt x, generate G responses (typically G = 4 to 8). Score each response with the reward function. Then compute advantages using group statistics:

```
A_i = (r_i - mean(r_1, ..., r_G)) / std(r_1, ..., r_G)
```

Breaking this down piece by piece:

- **r_i**: the reward score for the i-th response in the group
- **mean(r_1, ..., r_G)**: the average reward across all G responses to this prompt — this is the **baseline**
- **std(r_1, ..., r_G)**: the standard deviation of rewards in the group — this **normalizes** the advantages

The subtraction (r_i - mean) centers the advantages: above-average responses get positive advantages, below-average get negative. The division by std ensures advantages are on a consistent scale regardless of whether rewards range from 0-1 or 0-100.

### Worked Example

```
Prompt: "What is the capital of France?"

Response 1: "Paris"                                → reward = 1.0
Response 2: "The capital is Paris, a beautiful..." → reward = 0.8
Response 3: "Lyon"                                 → reward = 0.0
Response 4: "Paris is the capital"                 → reward = 0.9

Group mean = (1.0 + 0.8 + 0.0 + 0.9) / 4 = 0.675
Group std  = sqrt(((1.0-0.675)² + (0.8-0.675)² + (0.0-0.675)² + (0.9-0.675)²) / 4)
           = sqrt((0.1056 + 0.0156 + 0.4556 + 0.0506) / 4)
           = sqrt(0.1569)
           = 0.396

Advantages:
A_1 = (1.0 - 0.675) / 0.396 = +0.82  ← reinforce strongly (best response)
A_2 = (0.8 - 0.675) / 0.396 = +0.32  ← reinforce mildly
A_3 = (0.0 - 0.675) / 0.396 = -1.70  ← suppress strongly (wrong answer)
A_4 = (0.9 - 0.675) / 0.396 = +0.57  ← reinforce moderately
```

Notice: responses are ranked **relative to each other**, not against some absolute standard. If all four responses were mediocre, the best mediocre one would still get a positive advantage. This is the "relative" in Group Relative Policy Optimization.

## GRPO Objective

The full GRPO loss function:

```
L_GRPO = E_{x~D} [ (1/G) Σ_{i=1}^{G} min(
    r_t(θ) · A_i,
    clip(r_t(θ), 1-ε, 1+ε) · A_i
) ] - β · KL(π_θ || π_ref)
```

Where:
- **r_t(θ) = π_θ(y_i|x) / π_old(y_i|x)**: the probability ratio (same as PPO)
- **clip(r_t(θ), 1-ε, 1+ε)**: bounds the ratio to [1-ε, 1+ε], typically [0.8, 1.2]
- **A_i**: the group-normalized advantage (computed as above)
- **β · KL(π_θ || π_ref)**: KL penalty to stay close to the reference model

The critical insight: **GRPO still uses PPO-style clipping!** The only difference from PPO is how advantages are computed:

```
PPO:  A_t = GAE(rewards, V_ψ)            ← learned value function (another neural net)
GRPO: A_i = (r_i - mean(r)) / std(r)     ← group statistics (just arithmetic)
```

Everything else — the clipped surrogate objective, the KL penalty, the importance sampling ratio — is identical to PPO.

## GRPO vs PPO

| Aspect | PPO | GRPO |
|--------|-----|------|
| Models in memory | Policy + Reference + Reward + Value (4) | Policy + Reference + Reward (3) |
| Advantage computation | GAE with learned V_ψ | Group mean/std normalization |
| Extra model training | Must train V_ψ alongside policy | None |
| GPU memory | ~25-50% more (for V_ψ) | Lower |
| Implementation complexity | High (GAE, value loss, dual optimization) | Moderate |
| Samples per prompt | 1 | G (typically 4-8) |
| Best for | General RLHF with learned rewards | Tasks with verifiable rewards (math, code) |
| Bias | Depends on V_ψ accuracy | Slightly biased (sample includes itself in mean) |

**When to use GRPO**: When you have clear reward signals (correctness checks, unit tests), limited GPU memory, or want a simpler training pipeline. GRPO shines for reasoning tasks where answers are verifiable.

**When to use PPO**: When you have a learned reward model, need sample efficiency (PPO needs fewer generations per prompt), or are working on open-ended generation where group comparisons are less meaningful.

## RLOO (REINFORCE Leave-One-Out)

RLOO uses a similar idea to GRPO but with a **leave-one-out** baseline:

```
A_i = r_i - (Σ_{j≠i} r_j) / (G - 1)
```

In plain English: "For each response, the baseline is the average reward of ALL OTHER responses in the group." The response being evaluated is excluded from its own baseline.

Using our earlier example (rewards: 1.0, 0.8, 0.0, 0.9):

```
A_1 = 1.0 - (0.8 + 0.0 + 0.9)/3 = 1.0 - 0.567 = +0.433
A_2 = 0.8 - (1.0 + 0.0 + 0.9)/3 = 0.8 - 0.633 = +0.167
A_3 = 0.0 - (1.0 + 0.8 + 0.9)/3 = 0.0 - 0.900 = -0.900
A_4 = 0.9 - (1.0 + 0.8 + 0.0)/3 = 0.9 - 0.600 = +0.300
```

Why leave-one-out? Because including the sample in its own baseline introduces a small bias. RLOO produces an **unbiased** estimator of the advantage, which can matter for convergence guarantees.

| Aspect | GRPO | RLOO |
|--------|------|------|
| Baseline | Group mean (includes self) | Leave-one-out mean (excludes self) |
| Normalization | Divides by group std | No std normalization |
| Bias | Slightly biased | Unbiased |
| Variance | Lower (due to normalization) | Slightly higher |
| In practice | Used by DeepSeek | Strong results in Hugging Face TRL |

## DeepSeek-R1: Pure RL for Reasoning

DeepSeek-R1 demonstrated a radical simplification of the alignment pipeline. Key findings:

**1. Skip SFT entirely.** Go directly from the pre-trained base model to GRPO. No supervised fine-tuning on human demonstrations. The model learns to reason purely from reinforcement signals.

**2. Use rule-based rewards instead of learned reward models.**
- Math: check if the final answer matches the ground truth
- Code: run the code and check if tests pass
- Format: verify the response follows `<think>...</think><answer>...</answer>` structure

**3. Chain-of-thought reasoning EMERGES from RL.** Nobody taught the model to "think step by step." Given only a correctness reward, the model spontaneously developed:
- Breaking problems into sub-steps
- Trying multiple approaches
- Self-checking and backtracking
- Explicit "wait, let me reconsider" moments

**4. Self-reflection appears spontaneously.** The model learns to say things like "Hmm, that doesn't seem right" and correct itself — purely because this strategy earns higher rewards.

The pipeline is remarkably simple:

```
Pre-trained model → GRPO with correctness reward → Reasoning model
(No SFT step!)      (No learned reward model!)
```

**Why it matters**: If your rewards are verifiable (math, code, formal logic), you may not need human preferences at all. This dramatically simplifies alignment — no preference data collection, no reward model training, no SFT data curation. Just a base model, a correctness checker, and GRPO.

## Algorithm Selection Guide

When choosing an RL/alignment algorithm, use this decision tree:

```
Do you have preference pairs (chosen vs rejected)?
├── No  → Do you have only positive examples?
│        ├── Yes → KTO (Kahneman-Tversky Optimization)
│        └── No  → Collect preferences or use rule-based rewards
│
└── Yes → Are you memory constrained?
         ├── Yes → ORPO or SimPO (no reference model needed)
         └── No  → Do you need online generation during training?
                  ├── No  → DPO (simple, stable, offline)
                  └── Yes → What kind of reward signal?
                           ├── Rule-based (math, code) → GRPO
                           └── Learned reward model    → PPO
```

Rules of thumb:
- **Start with DPO** if you have preference data — it's the simplest online-free method
- **Use GRPO** if you have verifiable rewards and want emergent reasoning
- **Use PPO** if you need maximum control and have the compute budget
- **Use KTO** if you only have thumbs-up/thumbs-down signals (no pairs)

## Key Papers

- [DeepSeekMath: GRPO (Shao et al., 2024)](https://arxiv.org/abs/2402.03300) — Introduced GRPO for math reasoning
- [DeepSeek-R1 (DeepSeek, 2025)](https://arxiv.org/abs/2501.12948) — Pure RL reasoning without SFT
- [RLOO / Back to Basics (Ahmadian et al., 2024)](https://arxiv.org/abs/2402.14740) — Leave-one-out baselines for LLM RL

## Related

- [PPO for LLMs](02_PPO.md) — GRPO simplifies PPO by removing the value model
- [DPO & Variants](04_DPO_and_Variants.md) — Offline alternative that avoids RL entirely
- [AI_Infra Algorithms](../../AI_Infra/rl/01_Algorithms.md) — Implementation details for RL training
