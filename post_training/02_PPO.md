# PPO for LLMs

> Parent: [Post-Training & Alignment](00_Post_Training.md)

## Overview

PPO (Proximal Policy Optimization) is the algorithm that powered InstructGPT, ChatGPT, and GPT-4's alignment with human preferences. It is the workhorse of RLHF (Reinforcement Learning from Human Feedback) and the single most important algorithm to understand in the post-training stack.

PPO is a **policy gradient method** that uses a **clipped surrogate objective** to make stable, incremental updates to the language model. In the LLM context, "policy" simply means the language model itself -- the function that maps a prompt to a probability distribution over next tokens. PPO maximizes reward model scores (a proxy for human preferences) while staying close to a reference (SFT) model via a KL divergence penalty. This prevents the model from drifting into degenerate outputs that exploit the reward model.

The algorithm was introduced by John Schulman et al. at OpenAI in 2017 as a simpler, more practical alternative to TRPO (Trust Region Policy Optimization). It became the default RL algorithm for LLM alignment because it balances stability, sample efficiency, and implementation simplicity.

**One-sentence summary for interviews:**

> "PPO generates responses, scores them with a reward model, computes per-token advantages using a value function and GAE, then updates the policy with a clipped objective that prevents destructively large steps -- all while a KL penalty keeps the model close to the original SFT checkpoint."

If you remember nothing else, remember that sentence.

---

## The RLHF Objective

Before diving into PPO's mechanics, we need to understand what we are optimizing. The full RLHF objective is:

```
max_θ  E_{x~D, y~π_θ(·|x)} [ r_φ(x, y) ]  -  β · KL(π_θ || π_ref)
```

Let's break down **every single symbol**:

| Symbol | Meaning |
|--------|---------|
| `max_θ` | Adjust the model weights θ to maximize this entire expression |
| `E` | Expectation (average) over many samples |
| `x ~ D` | Prompts x are sampled from a dataset D of training prompts |
| `y ~ π_θ(·\|x)` | Responses y are generated (sampled) from the current policy π_θ given prompt x |
| `π_θ` | The policy -- this IS the language model, parameterized by weights θ |
| `r_φ(x, y)` | The reward model (with its own frozen parameters φ) assigns a scalar score to the (prompt, response) pair |
| `β` | A hyperparameter (typically 0.01 to 0.2) controlling how much we penalize divergence from the reference model |
| `KL(π_θ \|\| π_ref)` | KL divergence measuring how different the current policy is from the reference (frozen SFT) model |
| `π_ref` | The reference policy -- a frozen copy of the SFT model, never updated |

**Plain English:** "Generate responses. Get reward scores. But don't change too much from the original SFT model."

### Why the KL Term Matters

The KL penalty is not optional decoration -- it is critical to training stability:

| Scenario | What Happens |
|----------|-------------|
| **No KL penalty (β = 0)** | The model finds reward hacks: repetitive phrases, adversarial patterns that score high on the reward model but are gibberish to humans. The model "overfits" to the reward model's weaknesses. |
| **β too small (e.g., 0.001)** | Reward hacking still dominates. Model outputs become degenerate. |
| **β just right (e.g., 0.02-0.1)** | Model improves quality and alignment while remaining fluent and capable. Sweet spot. |
| **β too large (e.g., 1.0)** | Model barely changes from the SFT checkpoint. Alignment is weak because the KL cost overwhelms the reward signal. |

Think of β as a leash. Too loose and the dog runs into traffic. Too tight and the dog can't move at all.

---

## From REINFORCE to PPO

PPO didn't appear out of nowhere. It is the result of a clear evolutionary path in policy gradient methods. Understanding this path explains *why* PPO is designed the way it is.

### The Evolution

```
REINFORCE (1992)     ──>    TRPO (2015)         ──>    PPO (2017)
─────────────────          ─────────────────          ─────────────────
∇J = E[∇log π · R]        Constrained update          Clipped update
                           KL(π_new||π_old) ≤ δ        clip(ratio, 1±ε)

High variance              Low variance                Low variance
Unstable updates           Stable updates              Stable updates
Simple to implement        Expensive (conjugate        Simple to implement
                           gradients, line search)
```

### Why REINFORCE Falls Short

The basic REINFORCE gradient is:

```
∇J(θ) = E_t [ ∇ log π_θ(a_t | s_t) · R_t ]
```

Two problems make this impractical for LLMs:

1. **High variance.** R_t is a noisy, high-variance signal. The gradient estimate bounces wildly between updates. Solution: subtract a baseline (the value function) to get an **advantage** A_t instead of raw return R_t. This is where the value model enters the picture.

2. **Large updates can be catastrophic.** If one gradient step happens to be large, the policy can change drastically, generating terrible outputs on the next batch, leading to another wild gradient step -- a death spiral. Solution: constrain how much the policy can change per update.

### TRPO's Approach (and Why It's Too Expensive)

TRPO (Trust Region Policy Optimization) solves the instability problem elegantly: it adds a hard constraint that the KL divergence between the new and old policy must stay below a threshold δ:

```
maximize   E_t [ (π_θ(a_t|s_t) / π_old(a_t|s_t)) · A_t ]
subject to KL(π_old || π_θ) ≤ δ
```

This works beautifully in theory, but solving this constrained optimization requires computing second-order derivatives (the Fisher information matrix) and using conjugate gradients. For a 7B-parameter LLM, this is prohibitively expensive.

### PPO's Key Insight

PPO approximates the trust region constraint using a simple **clipping mechanism** on the probability ratio. No second-order derivatives. No conjugate gradients. Just a `min` and a `clip` -- operations that are trivially cheap. The result is nearly as stable as TRPO but as simple to implement as REINFORCE.

---

## The Clipped Surrogate Objective

This is the heart of PPO. We will build it from scratch, one piece at a time.

### Step 1: The Importance Sampling Ratio

When we collect a batch of data using the current policy π_old, then update the policy to π_θ, we need to account for the fact that the data was generated by a *different* policy. The importance sampling ratio does this:

```
r_t(θ) = π_θ(a_t | s_t) / π_old(a_t | s_t)
```

| Symbol | Meaning |
|--------|---------|
| `r_t(θ)` | The probability ratio for action a_t at state s_t |
| `π_θ(a_t \| s_t)` | Probability the **current** (being-updated) policy assigns to action a_t |
| `π_old(a_t \| s_t)` | Probability the **old** (data-generating) policy assigned to action a_t |

Intuition for the ratio's value:

- `r_t = 1.0` -- current policy assigns the **same** probability as old policy (no change)
- `r_t = 2.0` -- current policy is **twice as likely** to take this action
- `r_t = 0.5` -- current policy is **half as likely** to take this action
- `r_t = 0.0` -- current policy would **never** take this action

**For LLMs:** s_t is the prompt plus all tokens generated so far, and a_t is the next token. So r_t is the ratio of the probability the updated model assigns to that specific token versus what the old model assigned.

### Step 2: The CPI Surrogate

The Conservative Policy Iteration (CPI) surrogate objective is:

```
L^CPI(θ) = E_t [ r_t(θ) · A_t ]
```

| Symbol | Meaning |
|--------|---------|
| `L^CPI(θ)` | The surrogate loss we want to maximize |
| `r_t(θ)` | Importance sampling ratio from Step 1 |
| `A_t` | **Advantage** -- how much better this action was compared to the average action from this state |

Why this works:

- If `A_t > 0` (good action) and we increase π_θ for that action, r_t goes up, loss goes up. Good -- we're reinforcing good behavior.
- If `A_t < 0` (bad action) and we decrease π_θ for that action, r_t goes down, making `r_t · A_t` less negative. Good -- we're discouraging bad behavior.

### Step 3: The Problem with Unconstrained Optimization

Without any constraint, the optimizer can push r_t(θ) to extreme values. Suppose A_t = +5.0 for some action. The optimizer wants to maximize `r_t · 5.0`, so it makes r_t as large as possible -- maybe 10, 50, 100. This means the policy has changed *dramatically* in a single update, leading to:

- **Performance collapse** -- the new policy is so different it generates nonsense
- **Irreversible damage** -- you can't easily recover from a catastrophically bad policy
- **Training instability** -- each update overcorrects, leading to oscillation or divergence

### Step 4: PPO's Clipping Solution

PPO constrains the objective by clipping the ratio:

```
L^CLIP(θ) = E_t [ min(
    r_t(θ) · A_t,                              <-- unclipped term
    clip(r_t(θ), 1 - ε, 1 + ε) · A_t           <-- clipped term
)]
```

Where ε (epsilon) is typically **0.2**, meaning the ratio is clipped to the range [0.8, 1.2].

The `min` operator takes the **more pessimistic** (conservative) estimate:

- If the unclipped objective suggests a big improvement, the clipped version says "hold on, let's be cautious" -- and `min` picks the cautious estimate.
- If the unclipped objective suggests things are getting worse, `min` lets the bad signal through unclipped, so the optimizer gets the full gradient to fix the problem.

This is the genius of PPO: it is **pessimistic about improvements** but **honest about problems**.

### Step 5: Worked Examples with Actual Numbers

Let's trace through four scenarios with ε = 0.2 (clip range [0.8, 1.2]):

---

**Scenario 1: Good action, policy wants to increase it too much**

- `A_t = +2.0` (this was a good action)
- `r_t = 1.5` (current policy gives 50% more probability than old)
- Unclipped: `1.5 x 2.0 = 3.0`
- Clipped: `clip(1.5, 0.8, 1.2) x 2.0 = 1.2 x 2.0 = 2.4`
- `min(3.0, 2.4) = 2.4` -- **CLIPPED!** Prevents overconfident update. The policy already moved enough.

---

**Scenario 2: Good action, policy barely changed**

- `A_t = +2.0` (this was a good action)
- `r_t = 1.05` (policy only changed 5%)
- Unclipped: `1.05 x 2.0 = 2.1`
- Clipped: `clip(1.05, 0.8, 1.2) x 2.0 = 1.05 x 2.0 = 2.1`
- `min(2.1, 2.1) = 2.1` -- **NOT clipped.** Small update is fine, gradient flows normally.

---

**Scenario 3: Bad action, policy is correctly reducing it**

- `A_t = -1.0` (this was a bad action)
- `r_t = 0.7` (policy already reduced probability by 30%)
- Unclipped: `0.7 x (-1.0) = -0.7`
- Clipped: `clip(0.7, 0.8, 1.2) x (-1.0) = 0.8 x (-1.0) = -0.8`
- `min(-0.7, -0.8) = -0.8` -- Takes the **more negative** value! The clipped term is more pessimistic, so it is selected. The gradient *still flows* to continue correcting the bad action.

---

**Scenario 4: Bad action, policy wrongly increasing it**

- `A_t = -1.0` (this was a bad action)
- `r_t = 1.3` (policy is INCREASING probability -- going the wrong direction!)
- Unclipped: `1.3 x (-1.0) = -1.3`
- Clipped: `clip(1.3, 0.8, 1.2) x (-1.0) = 1.2 x (-1.0) = -1.2`
- `min(-1.3, -1.2) = -1.3` -- Takes the unclipped value! The full negative signal passes through so the optimizer corrects this mistake.

---

### Visualizing the Clipping

```
L^CLIP
  |
  |              / (unclipped, A > 0)
  |            /
  |──────────/─────────────── clipped ceiling when A > 0
  |        /
  |      /
  |    /
  |  /
  |/
──+──────────────────────────── r_t(θ)
  |        1-ε   1   1+ε
  |\
  |  \
  |    \
  |      \
  |        \
  |──────────\─────────────── clipped floor when A < 0
  |            \
  |              \ (unclipped, A < 0)
  |
```

**Key insight from the diagram:** Inside the range [1-ε, 1+ε], gradients flow normally. Outside that range, gradients are zeroed out for the "good news" direction (preventing overly optimistic updates) but still flow for the "bad news" direction (allowing the optimizer to correct mistakes).

---

## GAE (Generalized Advantage Estimation)

The advantage A_t measures "how much better was this action than what we expected?" Computing it well is critical for stable training. GAE provides a principled way to do this.

### The TD Residual (One-Step Advantage)

The simplest advantage estimate uses the temporal-difference (TD) residual:

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)
```

| Symbol | Meaning |
|--------|---------|
| `δ_t` | TD residual at time step t -- a one-step advantage estimate |
| `r_t` | Immediate reward received at step t |
| `γ` | Discount factor (0 to 1). For LLMs, typically γ = 1 since episodes are finite. |
| `V(s_{t+1})` | Value function's estimate of expected future reward from the next state |
| `V(s_t)` | Value function's estimate of expected future reward from the current state |

Intuition:

- `δ_t > 0`: "Things went **better** than expected" -- the reward plus future value exceeded our prediction
- `δ_t < 0`: "Things went **worse** than expected" -- we overestimated how good this state was

### The GAE Formula

GAE combines multiple TD residuals with exponential weighting:

```
A_t^GAE(γ,λ) = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
             = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}
```

| Symbol | Meaning |
|--------|---------|
| `λ` | GAE parameter (0 to 1) controlling the bias-variance trade-off |
| `(γλ)^l` | Exponentially decaying weight -- later TD residuals contribute less |

The λ parameter is the key knob:

| λ Value | Behavior | Bias | Variance | Name |
|---------|----------|------|----------|------|
| λ = 0 | `A_t = δ_t` (just the one-step TD residual) | High bias | Low variance | TD(0) |
| λ = 0.95 | Balanced mix of short and long-horizon estimates | Moderate | Moderate | **Practical sweet spot** |
| λ = 1 | `A_t = R_t - V(s_t)` (full Monte Carlo return minus baseline) | Low bias | High variance | Monte Carlo |

### GAE for LLMs Specifically

LLM RLHF has some unique characteristics that affect GAE:

- **γ = 1:** Episodes are finite (one response), so no discounting needed.
- **Sparse reward:** The reward model typically gives ONE scalar score for the entire response, applied at the final token. Most intermediate r_t values are zero (or just the KL penalty).
- **Dense KL signal:** The per-token KL penalty `-β · (log π_θ(token) - log π_ref(token))` provides a reward signal at every token position, partially compensating for sparse reward.
- **λ = 0.95:** Standard choice. Balances the sparse end-of-sequence reward with the dense per-token KL signals.

---

## The 4-Model Setup

PPO for LLMs requires four models in memory simultaneously. This is its biggest practical drawback.

```
┌───────────────────────────────────────────────────────────┐
│                    PPO for LLMs: 4-Model Setup             │
├───────────────────────────────────────────────────────────┤
│                                                            │
│  1. Policy Model π_θ          [TRAINABLE]                  │
│     ─────────────────────────────────────                  │
│     The language model being optimized.                    │
│     Generates responses; updated by PPO gradients.         │
│     Size: full LLM (e.g., 7B, 13B, 70B parameters)        │
│                                                            │
│  2. Reference Model π_ref     [FROZEN]                     │
│     ─────────────────────────────────────                  │
│     A frozen copy of the SFT checkpoint.                   │
│     Provides the KL anchor: KL(π_θ || π_ref).              │
│     Same architecture and size as the policy model.        │
│     Never receives gradient updates.                       │
│                                                            │
│  3. Reward Model r_φ          [FROZEN]                     │
│     ─────────────────────────────────────                  │
│     Trained on human preference data.                      │
│     Input: (prompt, response) → Output: scalar score.      │
│     Often smaller than the policy (e.g., 1B-3B).           │
│     Frozen during PPO training.                            │
│                                                            │
│  4. Value Model V_ψ           [TRAINABLE]                  │
│     ─────────────────────────────────────                  │
│     Estimates expected future reward at each token.         │
│     Used by GAE to compute advantages.                     │
│     Often initialized from the reward model or shares      │
│     the policy backbone with a separate value head.        │
│                                                            │
│  Memory Requirement: ~4x a single model's footprint        │
│  Example: For a 7B policy, ~56 GB in FP16 for weights      │
│  alone (7B x 2 bytes x 4 models), plus optimizer states.   │
│                                                            │
└───────────────────────────────────────────────────────────┘
```

This 4-model cost is the primary motivation for PPO alternatives like DPO (no RL at all) and GRPO (removes the value model, bringing it down to 3 models).

---

## PPO Training Loop

Here is the full algorithm, step by step:

```
PPO Training Loop (one iteration):
═══════════════════════════════════

1. SAMPLE PROMPTS
   x_1, x_2, ..., x_B  ~  D
   (Sample a batch of B prompts from the training dataset)

2. GENERATE RESPONSES
   y_i  ~  π_θ(·|x_i)    for each prompt x_i
   (Standard autoregressive decoding, same as normal LLM inference)
   Store the per-token log-probabilities: log π_old(token_t | x_i, y_i[:<t])

3. SCORE RESPONSES
   reward_i  =  r_φ(x_i, y_i)                            ← reward model score
   kl_t      =  log π_θ(token_t|...) - log π_ref(token_t|...)  ← per-token KL
   modified_reward_t  =  -β · kl_t                        ← dense KL penalty signal
   modified_reward_T  += reward_i                          ← add RM score at final token

4. COMPUTE VALUES AND ADVANTAGES
   v_t  =  V_ψ(x_i, y_i[:t])      for each token position t    ← value estimates
   A_t  =  GAE(modified_rewards, values, γ=1, λ=0.95)           ← advantages

5. STORE OLD LOG-PROBS
   log π_old(y_i | x_i)  ← snapshot of the policy BEFORE updates
   (These were already computed in step 2)

6. PPO UPDATE (K epochs over the same batch, typically K = 2-4)
   For each mini-batch in the collected data:

     a. Compute current log-probs: log π_θ(token_t | ...)
     b. Compute ratio: r_t = exp(log π_θ - log π_old)
     c. Clipped policy loss:
        L^CLIP = E_t[ min(r_t · A_t,  clip(r_t, 1-ε, 1+ε) · A_t) ]
     d. Value function loss:
        L^VF = E_t[ (V_ψ(s_t) - returns_t)² ]
     e. Total loss:
        L = -L^CLIP  +  c_v · L^VF
        (c_v ≈ 0.5 typically; we MINIMIZE this, hence the negation of L^CLIP)
     f. Compute gradients, clip gradient norms, take optimizer step on θ and ψ

7. REPEAT from step 1 with next batch of prompts
```

### Typical Hyperparameters

| Hyperparameter | Typical Value | Notes |
|----------------|---------------|-------|
| Batch size B | 64-512 prompts | Larger = more stable, slower |
| PPO epochs K | 2-4 | More epochs = more reuse, but ratio drifts |
| Clip ε | 0.2 | Ratio allowed in [0.8, 1.2] |
| KL coefficient β | 0.02-0.1 | Often adapted dynamically |
| GAE λ | 0.95 | Bias-variance trade-off |
| Learning rate | 1e-6 to 5e-6 | Much smaller than pretraining |
| Value coefficient c_v | 0.5-1.0 | Weight of value loss |
| Max gradient norm | 0.5-1.0 | Gradient clipping for stability |

---

## Token-Level vs Response-Level

Understanding the two "levels" of PPO for LLMs resolves a common source of confusion.

**Response level (what the reward model sees):**
- The reward model receives a complete (prompt, response) pair
- It outputs ONE scalar score for the entire response
- This score is sparse -- it only appears at the very last token position

**Token level (what PPO optimizes):**
- The policy π_θ is a token-level model: it outputs `P(next_token | context)` at every position
- The importance sampling ratio `r_t = π_θ(token_t) / π_old(token_t)` is computed per token
- The KL penalty is computed per token: `kl_t = log π_θ(token_t) - log π_ref(token_t)`
- The value function estimates expected future reward at each token position
- GAE computes a per-token advantage A_t

**How they connect:**
- The single response-level reward gets placed at the final token position
- The per-token KL penalties provide dense reward at every position
- GAE propagates the final reward backward through time (via the value function), assigning credit to earlier tokens
- The PPO clipped objective is then applied at each token independently

This is how PPO answers the **credit assignment problem**: "Which tokens in a 200-token response were responsible for the high reward score?" GAE and the value function distribute the credit across all token positions.

---

## Practical Issues and Solutions

| Issue | What Happens | Solution |
|-------|-------------|----------|
| **Reward hacking** | Model finds reward model exploits: repetition, excessive hedging, verbosity that scores high but is low quality | Increase β, use reward model ensembles, reward normalization/clipping, periodically retrain RM |
| **KL explosion** | Policy diverges too far from reference; outputs degrade | Lower learning rate, increase β, dynamic β adaptation (target a specific KL budget), early stopping |
| **Training collapse** | Outputs suddenly become gibberish mid-training | Gradient clipping (max norm 0.5-1.0), lower learning rate, learning rate warmup, monitor KL and reward curves |
| **Value function lag** | V_ψ doesn't keep up with rapid policy changes, giving stale advantage estimates | More value function update epochs, larger value learning rate, initialize V_ψ from reward model |
| **Mode collapse** | Model produces the same or very similar response for different prompts; diversity disappears | Entropy bonus in the loss, higher temperature during generation, diverse prompt sampling |
| **Catastrophic forgetting** | Model loses general capabilities while optimizing for reward | Stronger KL penalty, mix in pretraining data, use LoRA/adapter-based fine-tuning |
| **Memory pressure** | 4 models don't fit in GPU memory | LoRA on policy (smaller optimizer states), offload reference model to CPU, share backbone between policy and value model |

---

## Interview Q&A

### Q1: "Explain PPO for LLMs in 2 minutes"

**Answer:** PPO is the RL algorithm used to align LLMs with human preferences. The process starts after supervised fine-tuning (SFT). We have four models: the policy (the LLM being trained), a frozen reference model (the SFT checkpoint), a reward model trained on human preferences, and a value model for variance reduction.

In each iteration, we sample prompts, generate responses with the current policy, and score them with the reward model. We compute per-token advantages using GAE, which tells us which tokens were better or worse than expected. Then we update the policy using a clipped surrogate objective -- the key innovation of PPO. The clipping ensures we never change the policy too drastically in one step: the probability ratio between new and old policy is clipped to [0.8, 1.2]. A KL divergence penalty against the reference model prevents the policy from drifting into reward-hacking territory.

The result: stable, incremental improvements in alignment without catastrophic forgetting or mode collapse.

### Q2: "Why use clipping instead of a hard KL constraint like TRPO?"

**Answer:** TRPO enforces `KL(π_old || π_new) <= δ` as a hard constraint, which requires computing the Fisher information matrix and solving a constrained optimization problem using conjugate gradients and line search. This is computationally expensive, especially for models with billions of parameters.

PPO's clipping approximates the same goal -- preventing large policy changes -- but through a much simpler mechanism. By clipping the probability ratio to [1-ε, 1+ε], we effectively zero out the gradient when the policy has already moved "far enough" from the old policy. It's a first-order method (just standard gradient descent) that achieves similar stability to TRPO in practice. The empirical performance is nearly identical, but PPO is dramatically simpler to implement and far cheaper to compute.

### Q3: "What's the value model for? Can we remove it?"

**Answer:** The value model V_ψ estimates the expected future reward at each token position. It serves as a **baseline** for advantage estimation via GAE. Without it, we'd have to use the raw return as the advantage, which has extremely high variance -- especially in LLMs where a single reward signal at the end must be attributed across hundreds of tokens.

Yes, you can remove it -- that's exactly what GRPO (Group Relative Policy Optimization) does. GRPO generates multiple responses per prompt and uses the group's mean reward as the baseline instead of a learned value function. This eliminates one of the four models, reducing memory by ~25% and simplifying the system. DeepSeek-R1 uses GRPO. The trade-off is that you need more generations per prompt to get a stable baseline.

### Q4: "PPO vs DPO -- when to use which?"

**Answer:** PPO is an online RL method: it generates new responses, scores them, and updates iteratively. DPO (Direct Preference Optimization) is an offline method: it directly optimizes on a fixed dataset of preference pairs without any RL loop.

**Use PPO when:**
- You have a strong reward model and can afford the compute (4 models in memory)
- You want the model to explore and find novel good responses
- You're doing iterative, online training at scale (like OpenAI's approach)
- The task benefits from on-policy exploration

**Use DPO when:**
- You want simplicity -- DPO is just supervised learning on preference pairs
- Compute is limited (only 2 models needed: policy and reference)
- You have high-quality preference data and don't need exploration
- You want faster iteration cycles

In practice, PPO tends to produce stronger results when done well (it can explore beyond the preference dataset), but DPO is much easier to implement and debug. Many teams start with DPO and graduate to PPO if they need more performance.

### Q5: "What is reward hacking and how do you prevent it?"

**Answer:** Reward hacking is when the policy finds inputs that score high on the reward model but are not actually high-quality by human standards. The reward model is an imperfect proxy for human preferences, and the policy is essentially an adversarial optimizer against it.

Examples: The model might learn to be excessively verbose (because longer responses tend to score higher), repeat certain phrases the RM was trained to like, or generate adversarial patterns that exploit specific RM weaknesses.

Prevention strategies:
1. **KL penalty (β):** Keep the policy close to the reference model, limiting how far it can deviate.
2. **Reward model ensembles:** Use multiple RMs and take the minimum or average score, making it harder to exploit any single model.
3. **Reward normalization/clipping:** Normalize rewards to have zero mean and unit variance, clipping outliers.
4. **Iterative RM retraining:** Periodically collect new preference data on the current policy's outputs and retrain the RM to close the gaps the policy found.
5. **KL budget monitoring:** Track KL divergence during training and stop or reduce LR if it spikes.

---

## Key Papers

- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) -- The original PPO paper. Introduced clipped surrogate objective.
- [Training language models to follow instructions with human feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) -- InstructGPT. First large-scale application of PPO to LLM alignment.
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2016)](https://arxiv.org/abs/1506.02438) -- GAE paper. The advantage estimation method used inside PPO.
- [Trust Region Policy Optimization (Schulman et al., 2015)](https://arxiv.org/abs/1502.05477) -- TRPO. PPO's predecessor with hard KL constraints.
- [Secrets of RLHF in Large Language Models Part I (Zheng et al., 2023)](https://arxiv.org/abs/2307.04964) -- Practical insights on making PPO work for LLMs.

## Related

- [RL Foundations](01_RL_Foundations_for_LLMs.md) -- Policy gradient derivation this builds on
- [Reward Models](03_Reward_Models.md) -- How r_φ is trained
- [DPO & Variants](04_DPO_and_Variants.md) -- The main alternative to PPO
- [GRPO](05_GRPO_and_Modern_Methods.md) -- Removes the value model
- [AI_Infra RL Algorithms](../../AI_Infra/rl/01_Algorithms.md) -- Pseudocode and industry usage
