# RL Foundations for LLMs

> Parent: [Post-Training & Alignment](00_Post_Training.md)

## Overview

If you have done RL for robotics -- SAC, continuous control, sim-to-real -- LLM RL will feel both familiar and strange. The core loop is identical: **maximize expected reward by updating a policy**. But the "environment" is text generation, not physics.

```
Classical RL                          LLM RL
┌──────────┐   torque   ┌────────┐   ┌──────────┐   token    ┌────────────┐
│  Policy   │ ────────▶ │ Physics│   │  Policy   │ ────────▶ │ Deterministic│
│  π(a|s)   │           │ Engine │   │  π(tok|x) │           │ Append tok  │
└──────────┘ ◀──────── └────────┘   └──────────┘ ◀──────── └────────────┘
          reward (dense)                       reward (sparse, end of seq)
```

The differences that trip people up:

| What changes | Robotics | LLM RL |
|---|---|---|
| Action space | Continuous (joint torques) | Discrete (32K-128K tokens) |
| Reward density | Dense (every timestep) | Sparse (one score after full response) |
| Transition dynamics | Stochastic (physics noise) | Deterministic (append token) |
| Episode length | 100s-1000s of steps | 10s-1000s of tokens |
| Exploration goal | Visit new states | Stay close to the SFT model |
| Off-policy reuse | Easy (replay buffer) | Hard (distribution shift) |

This document maps every classical RL concept to its LLM counterpart and derives the policy gradient from scratch. By the end you should be able to read any RLHF paper and know exactly where each equation comes from.

## MDP Mapping: Robotics to LLM

The Markov Decision Process (MDP) is the formal framework for both domains. Here is the element-by-element mapping.

| MDP Element | Classical RL (Robotics) | LLM RL (RLHF) |
|---|---|---|
| **State s** | Joint angles, velocities, object poses | Prompt **x** (or prompt + tokens generated so far) |
| **Action a** | Continuous torques (e.g., 7-dim for a Franka arm) | Next token from vocabulary (~32K for LLaMA, ~128K for GPT-4) |
| **Policy pi(a\|s)** | Gaussian over torques: mu(s), sigma(s) | Softmax over vocabulary: pi_theta(token \| prompt + previous tokens) |
| **Transition P(s'\|s,a)** | Stochastic physics simulation | **Deterministic**: new state = old state + chosen token |
| **Reward r(s,a)** | Dense: reward every 20ms control step | **Sparse**: reward model score at end of full response |
| **Episode** | Robot attempt (reset when it falls, times out) | One prompt -> complete response (until EOS token) |
| **Discount gamma** | 0.99 (future rewards worth slightly less) | Often gamma=1 (finite episodes) or not used (bandit view) |
| **Value V(s)** | Expected return from current robot state | Expected reward from this partial response onward |
| **Horizon** | Fixed (e.g., 1000 steps) or until termination | Variable (until EOS token or max length) |

### Two Views of LLM RL

There are two ways to frame text generation as an RL problem. Most papers use a mix of both.

**Token-level MDP (fine-grained view)**

```
State:   [prompt, tok_1, tok_2, ..., tok_t]
Action:  tok_{t+1}  (one of ~32K-128K choices)
Reward:  0 for every step EXCEPT the last token, where r = RM(full_response)
Horizon: T tokens (variable, maybe 50-2000)
```

This is the "true" MDP. It is what PPO with GAE actually operates on. The problem is that the reward is extremely sparse -- hundreds of actions before any signal.

**Response-level bandit (coarse view)**

```
State:   prompt x
Action:  entire response y  (a single "mega-action")
Reward:  RM(x, y)
Horizon: 1 step
```

This is simpler. REINFORCE, DPO, and GRPO all effectively use this view. The downside is that you lose credit assignment within the response -- you cannot tell which sentence was good or bad.

In practice, algorithms like PPO bridge both views: they treat generation as a token-level MDP (to get per-token gradients) but assign rewards at the response level (from the reward model).

## Why Not SAC for LLMs?

If you come from robotics, SAC is probably your go-to algorithm. It is state-of-the-art for continuous control, sample-efficient, and stable. So why does every LLM paper use PPO (or REINFORCE variants) instead?

| Dimension | SAC (Robotics) | PPO (LLMs) | Why It Matters |
|---|---|---|---|
| **Action space** | Continuous (torques in R^n) | Discrete (token from vocab of 32K-128K) | SAC parameterizes a Gaussian over continuous actions. You cannot put a Gaussian over a discrete vocabulary. |
| **Off-policy vs on-policy** | Off-policy (replay buffer) | On-policy (fresh rollouts each iteration) | SAC reuses old experience because physics does not change. In LLM RL, the policy changes rapidly -- old responses from pi_old are misleading for pi_theta. |
| **Entropy treatment** | Maximize entropy H[pi] -- explore broadly | Minimize KL(pi \|\| pi_ref) -- stay close to SFT model | SAC *wants* the robot to try diverse actions. LLM RL *wants* the model to stay coherent and close to the supervised fine-tuned model. These are opposite goals. |
| **Q-function** | Q(s, a) for continuous a -- one forward pass | Q(s, a) for discrete a -- would need Q-value for each of 128K tokens | Technically possible (just predict a 128K-dim vector), but expensive and hard to train well. |
| **Critic architecture** | Small MLP (state dim ~50-200) | Would need to process full token sequences | The "state" in LLM RL is the entire prompt + generated text so far. The critic would be as large as the LLM itself. |

### The Real Reason PPO Won

1. **Simplicity**: PPO needs a policy model and a value head. SAC needs a policy, two Q-networks, a value network, and a target network. For 70B-parameter LLMs, each extra model doubles your GPU cost.

2. **KL penalty is natural**: LLM RL has a built-in reference policy (the SFT model). The objective is `R(y) - beta * KL(pi || pi_ref)`. This is fundamentally different from SAC's entropy maximization.

3. **On-policy is safer**: When you update a 70B model, you want the training signal to come from the model's *current* behavior, not stale data. Off-policy methods risk catastrophic updates from distribution mismatch.

4. **It just worked**: InstructGPT (2022) showed PPO works for RLHF. The community built on that. Sometimes engineering momentum matters more than theoretical elegance.

Could you adapt SAC for LLMs? Researchers have tried (e.g., replacing the Gaussian with a categorical distribution, using a discrete Q-function). But the results have not been compelling enough to displace PPO.

## Policy Gradient Theorem (REINFORCE)

This is the foundation that PPO, GRPO, and every other LLM RL algorithm builds on. If you understand this derivation, you understand the core of RLHF.

### The Objective

We want to find policy parameters theta that maximize expected reward:

```
J(theta) = E_{y ~ pi_theta}[ R(y) ]
```

Reading this aloud:
- `J(theta)` -- the "score" of our policy, as a function of its parameters theta
- `E_{y ~ pi_theta}` -- the average, when we sample responses y from our policy pi_theta
- `R(y)` -- the reward for response y (from a reward model or human rating)

### Step 1: Write as a Sum

Since y is a discrete sequence (a string of tokens), the expectation is a sum over all possible responses:

```
J(theta) = SUM_y  pi_theta(y) * R(y)
```

- `pi_theta(y)` -- the probability that our model generates response y
- `R(y)` -- the reward for that response
- We sum over every possible response y (astronomically many, but bear with the math)

### Step 2: Take the Gradient

We want to know how to change theta to increase J:

```
grad_theta J(theta) = SUM_y  grad_theta[ pi_theta(y) ] * R(y)
```

Problem: `grad_theta pi_theta(y)` is hard to compute directly and impossible to estimate by sampling (we can sample from pi_theta, but not from its gradient).

### Step 3: The Log Trick

This is the key insight. From calculus:

```
grad log f(x) = grad f(x) / f(x)
```

Rearranging:

```
grad f(x) = f(x) * grad log f(x)
```

Apply this to pi_theta(y):

```
grad_theta pi_theta(y) = pi_theta(y) * grad_theta log pi_theta(y)
```

Why does this help? Because now we have `pi_theta(y)` as a multiplier, which turns the sum back into an expectation we can estimate by sampling.

### Step 4: Substitute Back

```
grad_theta J(theta) = SUM_y  pi_theta(y) * grad_theta log pi_theta(y) * R(y)
                    = E_{y ~ pi_theta}[ grad_theta log pi_theta(y) * R(y) ]
```

This is the **REINFORCE** estimator (Williams, 1992). In plain English:

> Sample responses from the current policy. For each response, compute how to make it more likely (grad log pi) and scale by how good it was (R). Good responses get reinforced, bad ones get suppressed.

### Step 5: Estimate with Samples

In practice, we cannot sum over all possible responses. We sample N of them:

```
grad_theta J(theta)  ≈  (1/N) * SUM_{i=1}^{N}  grad_theta log pi_theta(y_i) * R(y_i)
```

**Concrete numerical example** (N=3 responses to "Explain gravity"):

| Response y_i | R(y_i) | log pi_theta(y_i) | Effect |
|---|---|---|---|
| "Gravity is the force..." (good) | 0.9 | -2.1 | Push theta to increase probability by 0.9 units |
| "Gravity is like magnets..." (ok) | 0.4 | -3.5 | Smaller push, 0.4 units |
| "I don't know" (bad) | -0.2 | -1.8 | Push theta to *decrease* probability by 0.2 units |

The gradient is the average of these pushes. Over many batches, the policy learns to produce responses like the first one.

### What is log pi_theta(y)?

For an autoregressive LLM, a response y = [tok_1, tok_2, ..., tok_T]. The log probability decomposes as:

```
log pi_theta(y) = SUM_{t=1}^{T}  log pi_theta(tok_t | prompt, tok_1, ..., tok_{t-1})
```

Each term is just the log-softmax output at position t. This is the same quantity you compute during standard language model training -- it is the cross-entropy loss, but now weighted by reward instead of supervised labels.

## Baselines and Advantage

### The Variance Problem

Raw REINFORCE has **high variance**. Consider this scenario: a reward model that outputs scores in [0, 1]. All rewards are positive.

```
Response A: R = 0.9   → reinforce (a lot)
Response B: R = 0.7   → reinforce (medium)
Response C: R = 0.5   → reinforce (a little)
```

Every response gets reinforced -- just some more than others. The gradient direction depends on random fluctuations in which responses happened to be sampled. You need enormous batch sizes to get a stable signal.

### The Baseline Fix

Subtract a constant b from all rewards:

```
grad_theta J(theta) = E[ grad_theta log pi_theta(y) * (R(y) - b) ]
```

**Key fact**: this does NOT change the expected gradient. Proof sketch:

```
E[ grad_theta log pi_theta(y) * b ] = b * E[ grad_theta log pi_theta(y) ]
                                     = b * grad_theta SUM_y pi_theta(y)
                                     = b * grad_theta 1
                                     = 0
```

The sum of all probabilities is always 1, so its gradient is always 0. Subtracting any constant from the reward does not bias the gradient, but it *does* reduce variance.

### The Value Function as Baseline

The best baseline (in terms of variance reduction) is the **value function**:

```
V(s) = E[ R | starting from state s ]
     = "the average reward you expect from this point onward"
```

This gives us the **advantage function**:

```
A(s, a) = Q(s, a) - V(s)
```

Where:
- `Q(s, a)` = expected reward if you take action a in state s, then follow the policy
- `V(s)` = expected reward if you just follow the policy from state s
- `A(s, a)` = "how much better is action a than the average action in state s?"

### Concrete Example

Prompt: "Explain gravity in simple terms"

| | Response A | Response B |
|---|---|---|
| Content | "Gravity pulls objects toward each other..." | "Gravity is complicated..." |
| Reward R | 0.8 | 0.3 |

The value function estimates V(prompt) = 0.55 (average expected reward for this prompt).

```
Advantage_A = 0.8 - 0.55 = +0.25   → above average → reinforce
Advantage_B = 0.3 - 0.55 = -0.25   → below average → suppress
```

Now the gradient has clear positive and negative signals. Much less variance. This is why PPO trains a separate **value model** (the critic) alongside the policy -- it provides the baseline that makes training stable.

## Why On-Policy Matters

In robotics, you can store (s, a, r, s') tuples in a replay buffer and reuse them thousands of times. This is what makes SAC so sample-efficient. Why can't we do this for LLMs?

### The Distribution Mismatch Problem

```
Iteration 1:  pi_old generates response y    → gets reward R(y)
Iteration 2:  pi_theta has been updated       → pi_theta(y) ≠ pi_old(y)
```

The response y was generated by pi_old. Using it to update pi_theta is like learning to drive by watching someone else's dashcam -- you see their decisions, not the decisions you would have made. The gradient `grad log pi_theta(y) * R(y)` is biased because y was sampled from the wrong distribution.

In robotics this matters less because:
- The policy changes slowly (small updates)
- Physics provides strong supervision (the reward signal is dense)
- The state space is low-dimensional (generalization is easier)

In LLM RL, the policy can change dramatically in a single update (high-dimensional parameter space, sharp softmax distributions), so old data goes stale fast.

### Importance Sampling to the Rescue

We can correct for the mismatch by reweighting samples:

```
E_{y ~ pi_theta}[ f(y) ] = E_{y ~ pi_old}[ (pi_theta(y) / pi_old(y)) * f(y) ]
```

The ratio `pi_theta(y) / pi_old(y)` is the **importance weight**. It upweights responses that the current policy finds more likely (and downweights ones it finds less likely).

Problem: if pi_theta and pi_old diverge, the ratio can explode:

```
pi_theta(y) = 0.5,  pi_old(y) = 0.001   →  ratio = 500
```

One sample now has 500x the influence of others. Training becomes unstable.

**PPO's solution**: clip the ratio to [1 - epsilon, 1 + epsilon] (typically epsilon = 0.2). This limits how much any single update can change the policy. This is the "proximal" in Proximal Policy Optimization.

For the full PPO derivation with clipping, see [PPO for LLMs](02_PPO.md).

## Key Papers

| Paper | Year | Key Contribution |
|---|---|---|
| [Policy Gradient Methods for RL (Sutton et al., 2000)](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) | 2000 | The policy gradient theorem -- the mathematical foundation for everything here |
| [REINFORCE (Williams, 1992)](https://link.springer.com/article/10.1007/BF00992696) | 1992 | Original Monte Carlo policy gradient algorithm |
| [GAE (Schulman et al., 2016)](https://arxiv.org/abs/1506.02438) | 2016 | Generalized Advantage Estimation -- smooth tradeoff between bias and variance |
| [PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) | 2017 | Practical policy gradient with clipped surrogate objective |
| [SAC (Haarnoja et al., 2018)](https://arxiv.org/abs/1801.01290) | 2018 | Soft Actor-Critic -- maximum entropy RL for continuous control |
| [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) | 2022 | First large-scale RLHF -- SFT + reward model + PPO on human preferences |

## Related

- [PPO for LLMs](02_PPO.md) -- Builds directly on policy gradients with clipping and the 4-model setup
- [Reading RL Math](06_Reading_RL_Math.md) -- Notation guide for the symbols used throughout this section
- [AI_Infra RL Overview](../../AI_Infra/rl/01_Algorithms.md) -- Pseudocode and comparison tables for PPO, DPO, GRPO
