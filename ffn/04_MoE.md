# Mixture of Experts (MoE)

> Parent: [FFN](00_FFN.md)

## Overview

Mixture of Experts (MoE) replaces the single FFN in each Transformer block with **N parallel expert FFNs**, but only activates the **top-k** experts for each token. This decouples model capacity (total parameters) from compute cost (active parameters per token), allowing models to scale to trillions of parameters while keeping inference FLOPs manageable. MoE has become a dominant scaling strategy -- Mixtral 8x7B has 46.7B total parameters but only 12.9B active per token, matching dense 70B-class performance.

## Core Idea: Sparse Computation

```
Dense FFN (standard):
  Every token → 1 FFN → uses 100% of FFN parameters

MoE FFN:
  Every token → Router → selects top-k of N experts
                       → uses only k/N of FFN parameters

Example (Mixtral 8x7B):
  8 experts, top-2 routing
  Each token uses 2/8 = 25% of the FFN parameters
  But the model has 8x the total FFN capacity
```

## MoE Architecture

```
                          x ∈ ℝ^d
                          │
                     ┌────▼────┐
                     │ Router  │  Linear(d, N) + Softmax
                     │ g(x)   │  produces N expert scores
                     └────┬────┘
                          │
              scores: [0.05, 0.42, 0.01, 0.35, 0.02, 0.08, 0.04, 0.03]
                            ↑                   ↑
                          top-1 = 0.42        top-2 = 0.35
                          Expert 2            Expert 4
                          │                     │
              ┌───────────┴─────────┐  ┌───────┴──────────┐
              │                     │  │                   │
              ▼                     │  ▼                   │
   ┌──────────────────┐    │  ┌──────────────────┐    │
   │   Expert 2       │    │  │   Expert 4       │    │
   │   (SwiGLU FFN)   │    │  │   (SwiGLU FFN)   │    │
   │   d → d_ff → d   │    │  │   d → d_ff → d   │    │
   └────────┬─────────┘    │  └────────┬─────────┘    │
            │              │           │              │
            ▼              │           ▼              │
         e₂(x)            │        e₄(x)            │
            │              │           │              │
            │   ×0.55      │           │  ×0.45      │
            │  (renorm)    │           │  (renorm)   │
            └──────────────┴───────────┘             │
                           │                          │
                      ┌────▼────┐                     │
                      │   Sum   │  weighted combination
                      └────┬────┘
                           │
                           ▼
              output = 0.55·e₂(x) + 0.45·e₄(x)
```

## MoE Formula

```
y = Σᵢ∈TopK  g(x)ᵢ · Expertᵢ(x)

Where:
  x         = input token hidden state ∈ ℝ^d
  g(x)      = Router(x) = Softmax(W_router · x) ∈ ℝ^N
  TopK      = indices of the k largest values in g(x)
  Expertᵢ   = independent FFN (typically SwiGLU)
  g(x)ᵢ     = renormalized gate score for expert i

Renormalization (ensures weights sum to 1):
  g(x)ᵢ = softmax_score_i / Σⱼ∈TopK softmax_score_j

Router is a simple linear layer:
  W_router ∈ ℝ^{N × d}     (N = number of experts)
  router_logits = W_router · x
  router_probs  = Softmax(router_logits)
```

## Key Components

### 1. The Router (Gate)

```
Router:  x ∈ ℝ^d  →  Linear(d, N)  →  Softmax  →  scores ∈ ℝ^N

The router is the simplest component -- just a linear layer:
  Parameters: d × N  (e.g., 4096 × 8 = 32K -- negligible)

Router decides which experts process each token.
This is the ONLY learned component that determines routing.
```

### 2. Top-k Selection

```
Scores:  [0.05, 0.42, 0.01, 0.35, 0.02, 0.08, 0.04, 0.03]

Top-1:   Expert 2 (score 0.42)
         → output = e₂(x)
         → simplest, used by Switch Transformer

Top-2:   Expert 2 (0.42), Expert 4 (0.35)
         → renormalize: 0.42/(0.42+0.35)=0.55, 0.35/(0.42+0.35)=0.45
         → output = 0.55·e₂(x) + 0.45·e₄(x)
         → used by Mixtral, most modern MoE

Top-6:   Top 6 experts activated
         → used by DeepSeek-V2/V3 (with 256 fine-grained experts)
```

### 3. Expert FFNs

Each expert is an independent FFN -- typically SwiGLU in modern models:

```
Expert_i(x) = W_down_i · (SiLU(W_gate_i · x) ⊙ W_up_i · x)

Each expert has its own W_gate, W_up, W_down
All experts have the same architecture, different weights
Experts learn to specialize in different token patterns
```

## Key Challenges

### Challenge 1: Load Balancing (Expert Collapse)

```
Problem: the router may learn to send most tokens to a few "popular" experts,
         while other experts receive very few tokens and never improve.

         Expert 1:  ████████████████████████  80% of tokens  ← overloaded
         Expert 2:  ███                        10% of tokens
         Expert 3:  █                           3% of tokens
         Expert 4:  █                           2% of tokens
         Expert 5:  ▏                           1% of tokens  ← starved
         Expert 6:  ▏                           1% of tokens
         Expert 7:  ▏                           1% of tokens
         Expert 8:  █                           2% of tokens

This is "expert collapse" -- most model capacity is wasted.
Ideal: each expert gets ~12.5% of tokens (1/8 for 8 experts).
```

### Challenge 2: Auxiliary Load Balancing Loss

```
Solution: add an auxiliary loss that encourages uniform routing.

L_balance = α · N · Σᵢ fᵢ · pᵢ

Where:
  N  = number of experts
  fᵢ = fraction of tokens routed to expert i (in a batch)
  pᵢ = average router probability for expert i (in a batch)
  α  = balance loss coefficient (typically 0.01)

Intuition:
  If expert i gets too many tokens (high fᵢ), the loss is high
  The router learns to spread tokens more evenly
  The α coefficient controls the strength of this regularization

This loss is added to the main language modeling loss:
  L_total = L_LM + L_balance
```

### Challenge 3: Training Instability

```
Routing instability:
  - Small changes in router weights → large changes in which experts are selected
  - Expert specialization can oscillate rather than converge
  - Sparse gradients: only k/N experts get gradients per token

Mitigation strategies:
  1. Router z-loss: penalize large router logits to prevent sharp distributions
     L_z = (1/T) Σₜ (log Σᵢ exp(zₜᵢ))²
  2. Expert capacity factor: cap maximum tokens per expert
  3. Token dropping: drop tokens when an expert is over capacity
  4. Noise in routing: add noise to router logits during training
     noisy_logits = logits + noise * softplus(W_noise · x)
```

## Expert Parallelism

```
In a standard dense model:
  Each GPU holds part of every layer (tensor parallelism)

In MoE:
  Each GPU holds a subset of experts (expert parallelism)
  Tokens are routed to the correct GPU via all-to-all communication

                GPU 0           GPU 1           GPU 2           GPU 3
              ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
              │Expert 0 │    │Expert 2 │    │Expert 4 │    │Expert 6 │
              │Expert 1 │    │Expert 3 │    │Expert 5 │    │Expert 7 │
              └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘
                   │              │              │              │
                   └──────────────┴──────────────┴──────────────┘
                                       │
                              All-to-All Communication
                    (each token sent to GPU hosting its selected expert)

Workflow:
  1. Each GPU runs the shared attention layer on its local tokens
  2. Router computes expert assignments for all tokens
  3. All-to-all: tokens dispatched to GPUs hosting selected experts
  4. Each GPU runs its local experts on received tokens
  5. All-to-all: results sent back to original GPUs
  6. Results combined with gate weights

Communication overhead is the main cost of MoE inference.
```

## Notable MoE Models

### Switch Transformer (Fedus et al., 2021)

```
Key innovation: top-1 routing (simplest possible MoE)
  - Only 1 expert per token → simpler, faster
  - Up to 1.6T parameters with 2048 experts
  - Showed MoE can scale to extreme sizes
  - Used in T5-style encoder-decoder

Limitation: top-1 routing is less expressive than top-2
```

### Mixtral 8x7B (Mistral AI, 2024)

```
Architecture:
  - 8 experts per layer, top-2 routing
  - Each expert: SwiGLU FFN with d_ff = 14336
  - 32 layers, d_model = 4096, 32 attention heads (GQA, 8 KV heads)

Parameters:
  - Total parameters:  46.7B  (all experts)
  - Active parameters: 12.9B  (2/8 experts per token)
  - Attention params:   ~2.1B  (shared, always active)
  - Expert params:     ~44.6B  (only 2/8 active)

Performance:
  - Matches or exceeds LLaMA-2 70B on most benchmarks
  - At the inference cost of a ~13B dense model
  - 6x fewer active params than 70B for similar quality
```

### DeepSeek-V2 / V3

```
Key innovation: fine-grained experts

DeepSeek-V2:
  - 160 experts per layer, top-6 routing
  - Much smaller experts → more specialization
  - 236B total, ~21B active

DeepSeek-V3:
  - 256 fine-grained experts + 1 shared expert
  - Top-8 routing among the 256 routed experts
  - Shared expert always active (handles common patterns)
  - 671B total, ~37B active
  - Auxiliary-loss-free load balancing (bias terms instead)

Shared expert architecture:
  output = SharedExpert(x) + Σᵢ∈Top8 gᵢ · Expertᵢ(x)
  → shared expert handles "general" computation
  → routed experts handle "specialized" computation
```

### Qwen MoE

```
Qwen1.5-MoE-A2.7B:
  - 60 experts, top-4 routing
  - 14.3B total, 2.7B active
  - Performance comparable to Qwen1.5-7B (dense)
  - Fine-grained experts similar to DeepSeek approach

Qwen2-57B-A14B (MoE):
  - 64 experts, top-8 routing
  - 57B total, 14B active
```

## Comparison: Dense vs MoE

| Aspect | Dense (LLaMA-2 70B) | MoE (Mixtral 8x7B) | MoE (DeepSeek-V3) |
|--------|---------------------|---------------------|---------------------|
| Total params | 70B | 46.7B | 671B |
| Active params | 70B | 12.9B | ~37B |
| FLOPs/token | ~140B | ~25B | ~74B |
| Memory (fp16) | 140 GB | 93 GB | 1.3 TB |
| Quality | Baseline | ~matches 70B | Exceeds most dense |
| Inference speed | Slow | ~5x faster | Moderate |
| Training data efficiency | Baseline | ~similar | Better |
| Serving complexity | Simple | Expert parallelism | Expert parallelism |
| Communication overhead | Minimal | All-to-all | All-to-all |

### When to Use MoE vs Dense

```
Use MoE when:
  ✓ You need high quality at low inference cost
  ✓ You have enough memory for all experts (even if sparse)
  ✓ Your serving infrastructure supports expert parallelism
  ✓ Batch sizes are large enough to amortize communication

Use Dense when:
  ✓ Memory is the bottleneck (MoE needs all params in memory)
  ✓ Serving infrastructure is simple (single GPU)
  ✓ Batch sizes are small (MoE overhead > savings)
  ✓ You need predictable, uniform compute per token
```

## PyTorch Pseudocode

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    """Single expert: a SwiGLU FFN."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    """Mixture of Experts layer with top-k routing.

    Replaces a single FFN with N expert FFNs and a learned router.
    Only the top-k experts are activated per token.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 8,
        top_k: int = 2,
        balance_coeff: float = 0.01,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.balance_coeff = balance_coeff

        # Router: simple linear layer
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # N independent expert FFNs
        self.experts = nn.ModuleList([
            SwiGLUExpert(d_model, d_ff) for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
            aux_loss: scalar load balancing loss
        """
        B, T, D = x.shape

        # Compute router scores
        router_logits = self.router(x)           # (B, T, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts per token
        top_k_probs, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )  # both (B, T, top_k)

        # Renormalize top-k probabilities to sum to 1
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute expert outputs (simplified loop -- real impl uses batched ops)
        output = torch.zeros_like(x)             # (B, T, D)
        for i in range(self.n_experts):
            # Find which tokens selected this expert
            mask = (top_k_indices == i).any(dim=-1)   # (B, T)
            if not mask.any():
                continue

            # Get the gate weight for this expert
            expert_idx_in_topk = (top_k_indices == i).float()
            gate_weight = (top_k_probs * expert_idx_in_topk).sum(dim=-1)  # (B, T)

            # Run expert on selected tokens
            expert_input = x[mask]                     # (num_selected, D)
            expert_output = self.experts[i](expert_input)

            # Weighted addition to output
            output[mask] += gate_weight[mask].unsqueeze(-1) * expert_output

        # Auxiliary load balancing loss
        # f_i = fraction of tokens routed to expert i
        # p_i = average router probability for expert i
        tokens_per_expert = torch.zeros(self.n_experts, device=x.device)
        for k in range(self.top_k):
            for i in range(self.n_experts):
                tokens_per_expert[i] += (top_k_indices[:, :, k] == i).float().sum()
        f = tokens_per_expert / (B * T * self.top_k)   # fraction per expert
        p = router_probs.mean(dim=[0, 1])               # avg prob per expert
        aux_loss = self.balance_coeff * self.n_experts * (f * p).sum()

        return output, aux_loss


# Demo
if __name__ == "__main__":
    d_model = 4096
    d_ff = 14336       # Mixtral expert size
    n_experts = 8
    top_k = 2

    moe = MoELayer(d_model, d_ff, n_experts, top_k)
    x = torch.randn(2, 32, d_model)
    out, loss = moe(x)

    total_params = sum(p.numel() for p in moe.parameters())
    expert_params = sum(p.numel() for p in moe.experts.parameters())
    active_params = expert_params * top_k / n_experts

    print(f"Output shape:   {out.shape}")            # [2, 32, 4096]
    print(f"Aux loss:       {loss.item():.6f}")
    print(f"Total params:   {total_params:,}")        # ~1.41B
    print(f"Expert params:  {expert_params:,}")       # ~1.41B
    print(f"Active params:  {int(active_params):,}")  # ~352M (2/8 of experts)
    print(f"Router params:  {d_model * n_experts:,}") # 32,768 (negligible)
```

## MoE Scaling Laws

```
Dense scaling:       Quality ∝ (Compute)^α       α ≈ 0.05 (Chinchilla)

MoE scaling:         Quality ∝ (Active_Compute)^α · (Total_Params)^β

MoE advantage:
  - Total params can scale independently of compute
  - Adding more experts increases capacity without increasing FLOPs per token
  - Diminishing returns: going from 8 → 16 experts helps less than 1 → 8

Practical guidance (from Mixtral, DeepSeek):
  - 8 experts, top-2: good balance of quality vs complexity
  - 64-256 experts, top-6-8: fine-grained, better routing, more complexity
  - Memory is the bottleneck: all expert weights must fit in memory (or be distributed)
```

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Outrageously Large Neural Networks (Shazeer et al., 2017)](https://arxiv.org/abs/1701.06538) | Original MoE layer for Transformers |
| [Switch Transformers (Fedus et al., 2021)](https://arxiv.org/abs/2101.03961) | Simplified top-1 routing, scaled to 1.6T params |
| [ST-MoE (Zoph et al., 2022)](https://arxiv.org/abs/2202.08906) | Stable training recipes for MoE |
| [Mixtral 8x7B (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088) | Practical MoE at scale, open-weight |
| [DeepSeek-V2 (DeepSeek AI, 2024)](https://arxiv.org/abs/2405.04434) | Fine-grained experts + shared expert |
| [DeepSeek-V3 (DeepSeek AI, 2024)](https://arxiv.org/abs/2412.19437) | 671B MoE, auxiliary-loss-free balancing |

## Related

- [Standard MLP](01_MLP.md) -- the single FFN that MoE replaces with multiple experts
- [Gated FFN (SwiGLU)](03_Gated_FFN.md) -- each expert is typically a SwiGLU FFN
- [Activation Functions](02_Activation_Functions.md) -- activation used inside each expert
