# Post-Training & Alignment

> Parent: [LLM Learning Path](../00_LLM.md)

## Overview

Pre-trained LLMs predict the next token — they don't follow instructions, refuse harmful requests, or produce helpful responses. **Post-training** transforms a raw language model into a useful assistant through two stages:

```
Pre-training              SFT                    RL Alignment
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Next-token   │ ───▶ │ Supervised   │ ───▶ │ PPO / DPO /  │
│ prediction   │      │ fine-tuning  │      │ GRPO         │
│ on web text  │      │ on (prompt,  │      │ on human     │
│              │      │  response)   │      │ preferences  │
└──────────────┘      └──────────────┘      └──────────────┘
   "predict"            "follow"              "align"
```

**SFT** teaches the model the *format* (instruction → response). **RL alignment** teaches the model *quality* (which responses are better). This section focuses on the RL alignment stage — the algorithms, the math, and how to read the papers.

## Learning Path

| # | Topic | File | What You Learn | Priority |
|---|-------|------|---------------|----------|
| 1 | RL Foundations for LLMs | [01_RL_Foundations_for_LLMs.md](01_RL_Foundations_for_LLMs.md) | MDP mapping, policy gradient, SAC→LLM bridge | ★★★★★ |
| 2 | PPO for LLMs | [02_PPO.md](02_PPO.md) | Clipped objective, GAE, 4-model setup, interview-ready | ★★★★★ |
| 3 | Reward Models | [03_Reward_Models.md](03_Reward_Models.md) | Bradley-Terry, RM architecture, PRM vs ORM | ★★★★☆ |
| 4 | DPO & Variants | [04_DPO_and_Variants.md](04_DPO_and_Variants.md) | DPO derivation, IPO, KTO, ORPO, SimPO | ★★★★★ |
| 5 | GRPO & Modern Methods | [05_GRPO_and_Modern_Methods.md](05_GRPO_and_Modern_Methods.md) | Group normalization, DeepSeek-R1, algorithm selection | ★★★★☆ |
| 6 | Reading RL Math | [06_Reading_RL_Math.md](06_Reading_RL_Math.md) | Paper notation, symbols, how to parse equations | ★★★★★ |

**Recommended order**: Start with **06** (notation guide) if paper math feels opaque, then **01** → **02** → **03** → **04** → **05**.

## Related Resources (Other Sections)

For **infrastructure** (distributed training, memory optimization, rollout engines):
→ [AI_Infra/rl/02_Infrastructure.md](../../AI_Infra/rl/02_Infrastructure.md)

For **frameworks** (TRL, OpenRLHF, veRL):
→ [AI_Infra/rl/03_Frameworks.md](../../AI_Infra/rl/03_Frameworks.md)

For **algorithm overview with code examples** (PPO, DPO, GRPO pseudocode and TRL snippets):
→ [AI_Infra/rl/01_Algorithms.md](../../AI_Infra/rl/01_Algorithms.md)

## Key Papers

| Paper | Year | Key Contribution |
|-------|------|------------------|
| [PPO](https://arxiv.org/abs/1707.06347) | 2017 | Clipped surrogate objective for stable policy updates |
| [InstructGPT](https://arxiv.org/abs/2203.02155) | 2022 | RLHF pipeline: SFT → RM → PPO for LLMs |
| [Constitutional AI](https://arxiv.org/abs/2212.08073) | 2022 | RLAIF — AI feedback replaces human labelers |
| [DPO](https://arxiv.org/abs/2305.18290) | 2023 | Skip reward model, optimize preferences directly |
| [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300) | 2024 | Group normalization replaces value function |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 2025 | Pure RL without SFT — reasoning emerges |

## Related

- [LLM Training](../training/01_Pretraining.md) — Pre-training is the stage before post-training
- [Generation & Decoding](../generation/01_Autoregressive_Decoding.md) — How models generate text (the "action" in RL)
