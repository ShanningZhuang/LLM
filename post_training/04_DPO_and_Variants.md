# DPO & Variants

> Parent: [Post-Training & Alignment](00_Post_Training.md)

## Overview

Direct Preference Optimization (DPO) is the single most important simplification in LLM alignment. It eliminates the reward model and the entire RL loop, replacing them with a **supervised loss on preference pairs**. The key insight: the optimal RLHF policy has a closed-form solution, so you can optimize it directly without PPO.

**DPO in one sentence (for interviews):**
"DPO rearranges the RLHF objective to show that the implicit reward equals `beta * log(pi_theta / pi_ref)`, then trains directly on preference pairs with a cross-entropy-like loss."

Before DPO, alignment required four models in memory (policy, reference, reward, value) and a complex online RL loop. DPO needs just two models (policy + reference) and a static preference dataset. Training looks like ordinary supervised learning: forward pass, compute loss, backward pass, update. No generation step, no reward scoring, no advantage estimation.

DPO was published by Rafailov et al. in May 2023 and rapidly became the default alignment method for open-source LLMs due to its simplicity and competitive performance. Most Llama, Mistral, and Zephyr fine-tunes use some form of DPO.

---

## The DPO Derivation

This is the core of DPO. Five steps, no skipping.

### Step 1: The RLHF Optimal Policy

The standard RLHF objective (what PPO maximizes) is:

```
max_theta  E_{x~D, y~pi_theta}[ r(x, y) ] - beta * KL( pi_theta || pi_ref )
```

In words: "Maximize expected reward while staying close to the reference policy." Expanding the KL term and writing it as a single expectation over `y`:

```
max_pi  E_{y~pi}[ r(x, y) ] - beta * SUM_y pi(y|x) * log( pi(y|x) / pi_ref(y|x) )
```

To find the optimal policy `pi*`, take the functional derivative with respect to `pi(y|x)`, add a Lagrange multiplier for the normalization constraint `SUM_y pi(y|x) = 1`, and set to zero:

```
d/d(pi(y|x)) [ r(x,y) * pi(y|x) - beta * pi(y|x) * log(pi(y|x) / pi_ref(y|x)) - lambda * pi(y|x) ] = 0

r(x,y) - beta * log(pi(y|x) / pi_ref(y|x)) - beta - lambda = 0

pi(y|x) = pi_ref(y|x) * exp( r(x,y) / beta ) * exp( -(lambda + beta) / beta )
```

Enforcing normalization absorbs the constants into a partition function `Z(x)`:

```
pi*(y|x) = (1 / Z(x)) * pi_ref(y|x) * exp( r(x,y) / beta )
```

where:

```
Z(x) = SUM_y  pi_ref(y|x) * exp( r(x,y) / beta )
```

**Breaking it down:**

| Symbol | Meaning |
|--------|---------|
| `pi*(y\|x)` | The best possible policy (what RLHF converges to) |
| `pi_ref(y\|x)` | Reference (SFT) model probability of response `y` |
| `exp(r(x,y)/beta)` | Exponential boost for high-reward responses |
| `Z(x)` | Normalization constant (partition function, prompt-dependent) |

**Intuition:** The optimal policy takes the reference distribution and re-weights it -- responses with high reward get exponentially more probability mass. `beta` controls how aggressive the re-weighting is. Small `beta` means sharp re-weighting (concentrate on highest-reward responses). Large `beta` means gentle re-weighting (stay close to reference).

### Step 2: Rearrange for the Implicit Reward

Starting from the optimal policy equation, solve for `r(x, y)`:

```
pi*(y|x) = (1 / Z(x)) * pi_ref(y|x) * exp( r(x,y) / beta )

=> pi*(y|x) / pi_ref(y|x) = (1 / Z(x)) * exp( r(x,y) / beta )

=> log( pi*(y|x) / pi_ref(y|x) ) = r(x,y) / beta  -  log Z(x)

=> r(x, y) = beta * log( pi*(y|x) / pi_ref(y|x) ) + beta * log Z(x)
```

**In plain English:** "The implicit reward of response `y` equals `beta` times the log-ratio of optimal-to-reference probability, plus a prompt-dependent constant."

The `beta * log Z(x)` term depends only on the prompt `x`, not on the response `y`. This will be critical in the next step.

### Step 3: Substitute into Bradley-Terry

Reward models are trained with the Bradley-Terry preference model:

```
P(y+ > y- | x) = sigma( r(x, y+) - r(x, y-) )
```

where `sigma` is the sigmoid function. Substitute the implicit reward:

```
P(y+ > y- | x) = sigma(
    beta * log(pi*(y+|x) / pi_ref(y+|x)) + beta * log Z(x)
  - beta * log(pi*(y-|x) / pi_ref(y-|x)) - beta * log Z(x)
)
```

### Step 4: Z(x) Cancels!

The `beta * log Z(x)` terms appear with opposite signs and cancel exactly:

```
P(y+ > y- | x) = sigma(
    beta * [ log(pi*(y+|x) / pi_ref(y+|x))  -  log(pi*(y-|x) / pi_ref(y-|x)) ]
)
```

This is the crucial insight: **we never need to compute `Z(x)`**, the intractable partition function. It drops out of the preference probability because preferences are relative (comparing two responses), not absolute.

### Step 5: The DPO Loss

Replace the theoretical optimal policy `pi*` with our trainable model `pi_theta`, and maximize the log-likelihood of observed preferences:

```
L_DPO(theta) = -E_{(x, y+, y-) ~ D} [
    log sigma( beta * ( log(pi_theta(y+|x) / pi_ref(y+|x))
                      - log(pi_theta(y-|x) / pi_ref(y-|x)) ) )
]
```

Define the **implicit reward** for cleaner notation:

```
r_theta(x, y) = beta * log( pi_theta(y|x) / pi_ref(y|x) )
```

Then the loss becomes:

```
L_DPO(theta) = -E [ log sigma( r_theta(x, y+) - r_theta(x, y-) ) ]
```

**In plain English:** "Maximize the probability that the implicit reward of the chosen response exceeds the implicit reward of the rejected response."

### Numerical Example

Suppose `beta = 0.1` and for a given prompt:

| Quantity | Chosen (`y+`) | Rejected (`y-`) |
|----------|---------------|-----------------|
| `log pi_theta(y\|x)` | -5.0 | -4.5 |
| `log pi_ref(y\|x)` | -5.2 | -4.3 |
| log-ratio `log(pi_theta / pi_ref)` | 0.2 | -0.2 |
| implicit reward `beta * log-ratio` | 0.02 | -0.02 |

```
logit = r_theta(y+) - r_theta(y-) = 0.02 - (-0.02) = 0.04
loss  = -log sigma(0.04) = -log(0.510) = 0.673
```

The loss is slightly below `log(2) = 0.693` (the loss at chance), meaning the model slightly prefers the chosen response. Gradient descent will push the logit higher, increasing `pi_theta(y+|x)` and decreasing `pi_theta(y-|x)` relative to the reference.

### Compact Implementation

```python
import torch.nn.functional as F

def dpo_loss(pi_chosen_logps, pi_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    All inputs are log-probabilities (summed over tokens), shape (batch,).
    """
    chosen_logratios = pi_chosen_logps - ref_chosen_logps      # log(pi_theta/pi_ref) for chosen
    rejected_logratios = pi_rejected_logps - ref_rejected_logps  # log(pi_theta/pi_ref) for rejected
    logits = beta * (chosen_logratios - rejected_logratios)
    return -F.logsigmoid(logits).mean()
```

---

## The Beta Parameter

`beta` is the single most important hyperparameter in DPO. It controls how far the policy can drift from the reference model.

| beta value | Behavior | Risk |
|------------|----------|------|
| 0.01 | Aggressive optimization, large deviations from SFT | Overfitting to preferences, reward hacking |
| 0.1 | Balanced (most common default) | Good starting point |
| 0.5 | Conservative, small adjustments | May underfit preferences |
| 1.0 | Very conservative, barely moves from SFT | Almost no alignment effect |

**Analogy:** `beta` is like the learning rate of preference learning. Too small and you overshoot (overfit to noisy preferences). Too large and you barely learn anything.

**Practical guidance:** Start with `beta = 0.1`. If the model degenerates (repetitive, sycophantic), increase `beta`. If alignment is too weak (still produces harmful content), decrease `beta`. Monitor the implicit reward margin `r_theta(y+) - r_theta(y-)` -- if it saturates quickly, `beta` is too small.

---

## DPO vs PPO Comparison

| Aspect | PPO (RLHF) | DPO |
|--------|-------------|-----|
| Reward model | Required (separate model) | Not needed (implicit) |
| Value model | Required | Not needed |
| Models in memory | 4 (policy, ref, reward, value) | 2 (policy + reference) |
| Training type | Online RL (generate, score, update) | Offline supervised (fixed dataset) |
| Stability | Tricky (many hyperparams to tune) | Stable (essentially supervised learning) |
| Data | Generates new data each iteration | Fixed preference dataset |
| Scalability | Expensive (4 models + generation) | Cheap (just forward passes) |
| Performance ceiling | Higher (can improve beyond data) | Limited by dataset quality |
| Best for | Max performance, online improvement | Quick alignment, limited compute |

**Decision guide:**

```
Do you have paired preference data?
  No  --> Use KTO (unpaired) or collect pairwise data
  Yes --> Continue

Is compute very limited (can only fit 1-2 models)?
  Yes --> Use ORPO or SimPO (no reference model needed)
  No  --> Continue

Do you need maximum performance and have engineering resources?
  Yes --> Use PPO (online RLHF) or Online DPO
  No  --> Use DPO (simple, competitive, well-understood)
```

---

## DPO Variants

### IPO (Identity Preference Optimization)

**Insight:** DPO can overfit when preference labels are noisy. The sigmoid-based loss drives the logit to infinity on easy examples. IPO uses a squared loss that saturates more gracefully.

**Loss:**

```
L_IPO = E[ ( log(pi_theta(y+|x)/pi_ref(y+|x)) - log(pi_theta(y-|x)/pi_ref(y-|x)) - 1/(2*beta) )^2 ]
```

In words: "Push the log-ratio gap toward a target of `1/(2*beta)` instead of toward infinity."

**When to use:** Noisy preference data, crowdsourced annotations, or when DPO training curves show the reward margin diverging.

### KTO (Kahneman-Tversky Optimization)

**Insight:** You do not need paired data (chosen vs rejected for the same prompt). Just label individual responses as "good" or "bad." KTO applies prospect theory: humans weight losses more heavily than gains.

**Loss (simplified):**

```
L_KTO = E_good[ 1 - sigma(beta * (log(pi/pi_ref) - E_ref)) ]
      + lambda * E_bad[ 1 - sigma(-beta * (log(pi/pi_ref) - E_ref)) ]
```

where `lambda = 1.33` (loss aversion coefficient from prospect theory) and `E_ref` is a running estimate of the reference KL.

**When to use:** You have thumbs-up/thumbs-down data (like chatbot feedback) rather than pairwise A-vs-B comparisons. Much easier to collect at scale.

### ORPO (Odds Ratio Preference Optimization)

**Insight:** Combine SFT and preference alignment into a single loss. No reference model needed -- just one model in memory.

**Loss:**

```
L_ORPO = L_SFT(y+) - lambda * log( odds(y+) / odds(y-) )

where odds(y) = P(y|x) / (1 - P(y|x))
```

In words: "Do supervised learning on the chosen response, plus a penalty based on the odds ratio between chosen and rejected."

**When to use:** Memory constrained (can only fit one model), want single-stage training (skip separate SFT step).

### SimPO (Simple Preference Optimization)

**Insight:** Length bias is a major problem in DPO -- longer responses accumulate higher total log-probability. SimPO normalizes by response length and adds a margin term.

**Loss:**

```
L_SimPO = -log sigma( beta * (r(y+)/|y+| - r(y-)/|y-|) - gamma )
```

where `|y|` is the number of tokens and `gamma` is a margin (typically 0.5-1.0). Also eliminates the reference model by using the policy itself as the reference.

**When to use:** Length bias is an issue (model outputs are getting longer over training), or you want to remove the reference model.

### Variant Comparison

| Method | Paired Data | Reference Model | Models in Memory | Key Feature |
|--------|-------------|-----------------|------------------|-------------|
| DPO | Yes | Yes | 2 | Simple, effective baseline |
| IPO | Yes | Yes | 2 | Robust to noisy labels |
| KTO | No | Yes | 2 | Works with unpaired data |
| ORPO | Yes | No | 1 | Memory efficient, single stage |
| SimPO | Yes | No | 1 | Length normalized, margin-based |

---

## DPO Limitations

1. **Offline only.** DPO trains on a fixed dataset. It cannot generate new responses and improve beyond what the data contains. PPO can explore and find better responses iteratively.

2. **Distribution mismatch.** The preference data was collected from `pi_ref` (or another model), but during training `pi_theta` drifts away from `pi_ref`. The preferences may not be valid for the new distribution.

3. **Length bias.** Longer responses accumulate higher total log-probability, so DPO tends to prefer verbose answers. This is why SimPO normalizes by length.

4. **Data quality bottleneck.** DPO is only as good as the preference annotations. Noisy, inconsistent, or biased labels directly corrupt the implicit reward.

5. **Reward hacking via formatting.** The model can learn superficial preferences (e.g., bullet points always win) rather than genuine quality differences.

**Mitigations:**

- **Online DPO / Iterative DPO:** Generate new responses with `pi_theta`, collect fresh preferences, retrain. Bridges the gap to PPO.
- **Rejection sampling:** Use a reward model to filter generated responses before DPO.
- **Length normalization:** SimPO or explicit length penalties.
- **Data curation:** Invest in high-quality, consistent preference labels.

---

## Key Papers

- [DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) -- Direct Preference Optimization: Your Language Model is Secretly a Reward Model
- [IPO (Azar et al., 2023)](https://arxiv.org/abs/2310.12036) -- A General Theoretical Paradigm to Understand Learning from Human Feedback
- [KTO (Ethayarajh et al., 2024)](https://arxiv.org/abs/2402.01306) -- KTO: Model Alignment as Prospect Theoretic Optimization
- [ORPO (Hong et al., 2024)](https://arxiv.org/abs/2403.07691) -- ORPO: Monolithic Preference Optimization without Reference Model
- [SimPO (Meng et al., 2024)](https://arxiv.org/abs/2405.14734) -- SimPO: Simple Preference Optimization with a Reference-Free Reward

## Related

- [PPO for LLMs](02_PPO.md) -- The online RL alternative that DPO replaces
- [Reward Models](03_Reward_Models.md) -- What DPO eliminates with its implicit reward
- [GRPO](05_GRPO_and_Modern_Methods.md) -- Another approach that eliminates the value model
- [Reading RL Math](06_Reading_RL_Math.md) -- Notation guide for the symbols used here
- [AI_Infra Algorithms](../../AI_Infra/rl/01_Algorithms.md) -- Code examples with TRL
