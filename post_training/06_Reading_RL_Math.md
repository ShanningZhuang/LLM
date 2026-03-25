# Reading RL Math

> Parent: [Post-Training & Alignment](00_Post_Training.md)

## Overview

RL papers can feel impenetrable because of dense notation. Equations sprawl across half a page, Greek letters pile up, and subscripts nest inside subscripts. But here's the secret: there are only about 20 symbols used repeatedly across PPO, DPO, GRPO, and related papers. Once you internalize them, most equations become readable — even skimmable.

This guide is your Rosetta Stone. It covers:
- The core symbols and what they mean in plain English
- How to parse subscripts, superscripts, and expectations
- The "log trick" that makes policy gradients work
- KL divergence intuition
- Worked examples parsing real equations from PPO and DPO papers

Keep this open alongside any RLHF paper, and the math will start making sense.

## The Core Symbols

| Symbol | Name | Plain English |
|--------|------|---------------|
| π | Policy | The model — maps inputs to probability distributions over outputs |
| π_θ | Parameterized policy | The model with weights θ (the one being trained) |
| π_ref | Reference policy | Frozen copy of the SFT model (used as an anchor) |
| π* | Optimal policy | The best possible policy (theoretical ideal) |
| θ | Parameters (theta) | Model weights — millions or billions of numbers |
| φ | Parameters (phi) | Usually reward model weights (to distinguish from policy weights) |
| E[·] | Expectation | Average value over some distribution |
| ∇_θ | Gradient (nabla) | Direction of steepest increase with respect to θ |
| σ(·) | Sigmoid | Squashes any real number to (0, 1): σ(z) = 1/(1+e^(-z)) |
| log | Natural logarithm | ln(x); inverse of exp. log(e) = 1, log(1) = 0 |
| ∝ | Proportional to | Equal up to a constant factor: f ∝ g means f = c·g for some constant c |
| ~ | Sampled from | "x ~ D" means x is drawn randomly from distribution D |
| r(x, y) | Reward | Scalar score for response y given prompt x |
| A(s, a) | Advantage | How much better this action is compared to the average from this state |
| V(s) | Value function | Expected total reward from state s onward |
| Q(s, a) | Action-value | Expected total reward from state s after taking action a |
| KL(P‖Q) | KL divergence | How different distribution P is from Q (asymmetric!) |
| β (beta) | Temperature/coefficient | Controls strength of regularization (e.g., KL penalty weight) |
| ε (epsilon) | Clip range | How far the probability ratio can deviate in PPO; typically 0.2 |
| γ (gamma) | Discount factor | How much to value future vs present rewards; typically 0.99-1.0 |
| λ (lambda) | GAE parameter | Bias-variance trade-off in advantage estimation; typically 0.95 |
| D | Dataset | Collection of training examples (prompts, responses, preferences) |

Tip: When you encounter a symbol you don't recognize, check this table first. Nine times out of ten, it's one of these.

## Subscript and Superscript Conventions

Subscripts and superscripts carry specific meanings in RL notation. Learning to read them is half the battle.

**Subscripts** typically indicate:
- **Parameterization**: π_θ means "policy parameterized by θ." The subscript tells you WHICH weights define this policy
- **Distribution specification**: E_{x~D} means "expectation when x is sampled from D." The subscript tells you WHAT is random and WHERE it comes from
- **Time index**: r_t means "reward at time step t." The subscript indexes a sequence
- **Differentiation variable**: ∇_θ means "gradient with respect to θ." The subscript says what we're differentiating by

**Superscripts** typically indicate:
- **Method name**: A^GAE means "advantage computed using GAE (Generalized Advantage Estimation)"
- **Estimation**: Â_t (with a hat) means "estimated advantage" — the hat ^ signals it's an approximation, not the true value
- **Optimality**: π* means "the optimal policy"
- **Iteration**: θ^(k) means "parameters at iteration k"

**Example**: In L^CLIP(θ), the superscript CLIP names the variant of the loss, and θ in parentheses shows what the loss depends on.

## Reading Expectations

The E[·] notation trips up many readers. Here's how to parse it systematically.

Consider this expression:

```
E_{x~D, y~π_θ(·|x)} [r(x, y)]
```

Read it in layers:

1. **What's inside the brackets** [r(x, y)]: We're computing the reward for prompt x and response y
2. **What's in the subscript** {x~D, y~π_θ(·|x)}: This tells us WHERE x and y come from
   - x ~ D: sample a prompt x from the dataset D
   - y ~ π_θ(·|x): generate a response y from the current policy, conditioned on x
3. **The E itself**: Average over many such samples

So the full expression means: "Sample many prompts from the dataset, generate responses with the current model, score them, and take the average."

In Python, this maps directly to:

```python
total = 0
for x in dataset:                    # x ~ D
    y = model.generate(x)            # y ~ π_θ(·|x)
    total += reward_model(x, y)      # r(x, y)
average = total / len(dataset)       # E[...]
```

**Common patterns**:
- E_{x~D}[f(x)]: average f over the dataset
- E_{t}[·]: average over time steps in a trajectory
- Ê_t[·]: the hat means "empirical estimate" — computed from a finite batch, not the true expectation

When you see nested expectations, read from outside in. Each layer introduces a new source of randomness.

## The Log Trick

Why does "log π" appear everywhere in policy gradient equations? Because of a mathematical identity called the **log-derivative trick** (also called the REINFORCE trick):

```
∇_θ π_θ(y|x) = π_θ(y|x) · ∇_θ log π_θ(y|x)
```

Or equivalently:

```
∇_θ log π_θ(y|x) = ∇_θ π_θ(y|x) / π_θ(y|x)
```

This identity lets us rewrite the policy gradient as:

```
∇_θ E[r] = E[ r · ∇_θ log π_θ(y|x) ]
```

**Why this matters for LLMs**: The probability of a full response y = (token_1, token_2, ..., token_T) is a product of per-token probabilities:

```
π_θ(y|x) = π_θ(token_1|x) · π_θ(token_2|x, token_1) · ... · π_θ(token_T|x, token_1, ..., token_{T-1})
```

This product can be astronomically small (e.g., 0.1^100 for a 100-token response). Taking the log converts the product into a numerically stable sum:

```
log π_θ(y|x) = Σ_t log π_θ(token_t | x, token_1, ..., token_{t-1})
```

This is just the sum of per-token log-probabilities — exactly what language models compute natively during training.

## KL Divergence Intuition

KL divergence appears in virtually every RLHF objective as a regularizer. Here's the definition and intuition:

```
KL(P || Q) = E_{x~P}[log(P(x) / Q(x))]
           = E_{x~P}[log P(x) - log Q(x)]
```

**Intuition**: "If you believe the data comes from Q, but it actually comes from P, how surprised are you on average?" The more P and Q differ, the larger the KL divergence.

Key properties:
- KL(P || Q) >= 0 always (Gibbs' inequality)
- KL(P || Q) = 0 if and only if P = Q
- **KL is asymmetric**: KL(P||Q) ≠ KL(Q||P) in general

**For LLMs**, KL(π_θ || π_ref) measures how far the current policy has drifted from the reference SFT model. Computed per-token and summed:

```
KL(π_θ || π_ref) = Σ_t [ log π_θ(token_t | context) - log π_ref(token_t | context) ]
```

- If the policy assigns the **same** probabilities as the reference for every token: KL = 0
- If the policy assigns very **different** probabilities: KL is large and positive

In RLHF, we penalize large KL to prevent the policy from "hacking" the reward model by producing unnatural text. The β coefficient controls how strongly we enforce this: large β keeps the model close to the reference; small β allows more freedom.

**Why KL(π_θ||π_ref) and not KL(π_ref||π_θ)?** The direction matters. KL(π_θ||π_ref) is "mode-covering" — the policy is penalized for putting probability mass where the reference does not. This prevents the model from generating outputs the SFT model would consider very unlikely.

## How Papers Build Equations

RL papers follow a remarkably consistent pattern when presenting their objectives. Once you see the pattern, new papers become much easier to read:

1. **Start with the simplest objective**: maximize expected reward
   ```
   max_θ E[r(x, y)]
   ```
2. **Add a constraint or regularizer**: prevent the policy from drifting too far
   ```
   max_θ E[r(x, y)] - β · KL(π_θ || π_ref)
   ```
3. **Introduce practical approximations**: importance sampling for off-policy data, clipping for stability
   ```
   max_θ E[ min(ratio · A, clip(ratio, 1-ε, 1+ε) · A) ] - β · KL
   ```
4. **Add variance reduction**: baselines, GAE, group normalization

When reading a new equation, ask yourself these four questions:
- **What is being optimized?** (maximized or minimized?)
- **What is the main term?** (reward signal, log-likelihood, preference comparison?)
- **What constraints or regularization are added?** (KL penalty, clipping, entropy bonus?)
- **What is being averaged over?** (check the E[·] subscripts — prompts? responses? time steps?)

This framework applies to PPO, DPO, GRPO, RLOO, KTO, and essentially every method in the post-training literature.

## Worked Example: PPO's Equation

From the PPO paper (Schulman et al., 2017), Equation 7:

```
L^CLIP(θ) = Ê_t [ min(r_t(θ) Â_t,  clip(r_t(θ), 1-ε, 1+ε) Â_t) ]
```

Parsing each piece:

- **L^CLIP(θ)**: A loss function. The superscript CLIP names this variant (vs. L^KL or L^SIMPLE). It depends on θ, the current policy weights
- **Ê_t**: Empirical average over time steps. The hat ^ means "estimated from a finite batch of samples," not the true expectation
- **r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)**: The probability ratio. How much more (or less) likely is this action under the new policy vs. the old? If r_t = 1.5, the new policy is 50% more likely to take this action
- **Â_t**: Estimated advantage at time t. Positive means "better than expected," negative means "worse." Computed via GAE in practice
- **clip(r_t(θ), 1-ε, 1+ε)**: Constrains the ratio to [0.8, 1.2] when ε = 0.2. Prevents any single update from changing probabilities too drastically
- **min(·, ·)**: Takes the more pessimistic (lower) of the clipped and unclipped objectives. This creates a "trust region" — the policy can't exploit large ratio values

The full expression says: "For each time step, compute how the policy change affects expected advantage, but cap the effect using clipping, then average over the batch."

## Worked Example: DPO's Equation

From the DPO paper (Rafailov et al., 2023):

```
L_DPO(π_θ; π_ref) = -E_{(x,y_w,y_l)~D} [
    log σ( β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)) )
]
```

Parsing each piece:

- **L_DPO(π_θ; π_ref)**: Loss to minimize. Depends on the current policy π_θ and the frozen reference π_ref (semicolon separates the trainable from the fixed)
- **E_{(x, y_w, y_l)~D}**: Average over the preference dataset. Each sample is a triple: prompt x, winning response y_w, losing response y_l
- **log σ(·)**: Log-sigmoid. Makes this look like binary cross-entropy — the model is doing classification ("which response is better?")
- **β log(π_θ(y_w|x) / π_ref(y_w|x))**: The implicit reward of the winning response. This is how much more the current policy likes y_w compared to the reference. Higher = model has learned to prefer the winner
- **β log(π_θ(y_l|x) / π_ref(y_l|x))**: The implicit reward of the losing response
- **The subtraction**: implicit_reward(winner) - implicit_reward(loser). We want this to be positive and large
- **The negative sign out front**: Converts maximization to minimization (since optimizers minimize loss)

The whole equation says: "Minimize the negative log-probability that the implicit reward of the winning response exceeds the losing response's implicit reward." In other words: **make the model prefer winners over losers**, where "preference" is measured by how much the policy's log-probabilities shift relative to the reference.

## Tips for Reading New Papers

1. **When you see E[...]**: Ask "average over WHAT?" and check the subscripts. The subscripts define the sources of randomness
2. **When you see ∇_θ**: Ask "gradient of WHAT with respect to WHAT?" The subscript is the variable; the expression after ∇ is the function
3. **When you see a new symbol**: Check the paper's "Notation" or "Preliminaries" section. If there isn't one, find the first occurrence — authors usually define symbols on first use
4. **When equations look overwhelming**: Identify the MAIN TERM first (usually the reward or log-probability), then look at additive terms (regularizers, baselines) one by one
5. **"It follows that" or "it can be shown"**: This signals a known result. Don't try to derive it — search for the referenced lemma, theorem, or prior paper
6. **Most RL equations follow one template**: E[gradient · advantage] + regularization. Once you see this skeleton, the specifics are just variations on the theme
7. **Subscripts vs. arguments**: f_θ(x) means "function f parameterized by θ, evaluated at x." The subscript is the model; the argument is the input
8. **Products become sums under log**: Whenever you see log applied to a sequence probability, it becomes a sum of per-token log-probs. This is the single most common transformation in LLM math

## Key Resources

- [Lilian Weng: Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) — Comprehensive blog post with clear notation and derivations
- [OpenAI Spinning Up](https://spinningup.openai.com/) — Excellent introduction to RL notation, key concepts, and intuition
- [Sutton & Barto, Chapter 13](http://incompleteideas.net/book/the-book.html) — The canonical reference for policy gradient methods

## Related

- [RL Foundations](01_RL_Foundations_for_LLMs.md) — Applies these symbols in full derivations
- [PPO for LLMs](02_PPO.md) — Clipped objective explained with this notation
- [DPO & Variants](04_DPO_and_Variants.md) — DPO derivation uses these conventions
