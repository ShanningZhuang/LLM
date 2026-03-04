# LLM Algorithms Learning Path

> Goal: Systematic understanding of Large Language Model algorithms, from architecture to generation
> Structure: Top-to-bottom — decoder-only architecture → components → training → generation

## Overview: The LLM Algorithm Stack (Top to Bottom)

```
┌─────────────────────────────────────────────┐
│           Generation & Decoding             │  ← Output
├─────────────────────────────────────────────┤
│              Training                       │
├─────────────────────────────────────────────┤
│   ┌───────────────────────────────────┐     │
│   │  Attention    │  Position  │ FFN  │     │  ← Transformer Block
│   │  MHA/GQA/MQA │  RoPE/ALiBi│SwiGLU│     │     internals
│   └───────────────────────────────────┘     │
├─────────────────────────────────────────────┤
│         Transformer Block (×N)              │  ← Residual + LayerNorm
├─────────────────────────────────────────────┤
│         Embedding & Output Head             │  ← Tokenization + LM Head
├─────────────────────────────────────────────┤
│        Decoder-Only Architecture            │  ← Bird's eye view
└─────────────────────────────────────────────┘
```

---

## Phase 1: Architecture Overview

### 1.1 Decoder-Only Architecture → [Notes](architecture/01_Decoder_Only.md)
- [ ] Full architecture walkthrough (input → output)
- [ ] Why decoder-only won (GPT vs BERT vs T5)
- [ ] Parameter counting and model dimensions

### 1.2 Model Families → [Notes](architecture/02_Model_Family.md)
- [ ] GPT series evolution
- [ ] LLaMA / LLaMA-2 / LLaMA-3
- [ ] Mistral / Mixtral
- [ ] DeepSeek-V2 / V3

### 1.3 Scaling Laws → [Notes](architecture/03_Scaling_Laws.md)
- [ ] Kaplan scaling laws
- [ ] Chinchilla optimal compute
- [ ] Emergent abilities debate

---

## Phase 2: Input & Output

### 2.1 Tokenization → [Notes](embedding/01_Tokenization.md)
- [ ] BPE (Byte Pair Encoding)
- [ ] WordPiece, SentencePiece, Unigram
- [ ] Vocabulary size trade-offs

### 2.2 Token Embedding → [Notes](embedding/02_Token_Embedding.md)
- [ ] Embedding layer mechanics
- [ ] Embedding dimension choices

### 2.3 Output Head → [Notes](embedding/03_Output_Head.md)
- [ ] LM head (linear + softmax)
- [ ] Weight tying with embedding

---

## Phase 3: Transformer Block Structure

### 3.1 Residual Connections → [Notes](transformer_block/01_Residual_Connection.md)
- [ ] Skip connections and gradient flow
- [ ] Why residuals are essential for deep networks

### 3.2 Layer Normalization → [Notes](transformer_block/02_Layer_Normalization.md)
- [ ] LayerNorm vs BatchNorm
- [ ] RMSNorm (LLaMA-style)

### 3.3 Pre-Norm vs Post-Norm → [Notes](transformer_block/03_Pre_Norm_Post_Norm.md)
- [ ] Architecture variants and training stability

---

## Phase 4: Attention Mechanism

### 4.1 Self-Attention → [Notes](attention/01_Self_Attention.md)
- [ ] Query, Key, Value computation
- [ ] Scaled dot-product attention

### 4.2 Multi-Head Attention → [Notes](attention/02_Multi_Head_Attention.md)
- [ ] Head splitting and concatenation
- [ ] Why multiple heads help

### 4.3 MQA & GQA → [Notes](attention/03_MQA_GQA.md)
- [ ] Multi-Query Attention
- [ ] Grouped-Query Attention
- [ ] KV cache size implications

### 4.4 Causal Masking → [Notes](attention/04_Causal_Mask.md)
- [ ] Autoregressive masking
- [ ] Implementation details

### 4.5 Efficient Attention → [Notes](attention/05_Efficient_Attention.md)
- [ ] FlashAttention (tiling, IO-awareness)
- [ ] Linear attention variants

---

## Phase 5: Position Encoding

### 5.1 Absolute Position Encoding → [Notes](position_encoding/01_Absolute_Encoding.md)
- [ ] Sinusoidal encoding (original Transformer)
- [ ] Learned position embeddings

### 5.2 RoPE → [Notes](position_encoding/02_RoPE.md)
- [ ] Rotary Position Embedding
- [ ] Why RoPE is dominant in modern LLMs

### 5.3 ALiBi → [Notes](position_encoding/03_ALiBi.md)
- [ ] Attention with Linear Biases
- [ ] Length generalization

### 5.4 Context Extension → [Notes](position_encoding/04_Context_Extension.md)
- [ ] YaRN, NTK-aware scaling
- [ ] Position Interpolation

---

## Phase 6: Feed-Forward Networks

### 6.1 Standard MLP → [Notes](ffn/01_MLP.md)
- [ ] Two-layer FFN architecture
- [ ] Expansion ratio and hidden dimension

### 6.2 Activation Functions → [Notes](ffn/02_Activation_Functions.md)
- [ ] ReLU → GELU → SiLU/Swish evolution
- [ ] Why smooth activations matter

### 6.3 Gated FFN → [Notes](ffn/03_Gated_FFN.md)
- [ ] SwiGLU (LLaMA, PaLM)
- [ ] GeGLU and gating mechanism

### 6.4 Mixture of Experts → [Notes](ffn/04_MoE.md)
- [ ] MoE architecture and routing
- [ ] Load balancing and expert collapse
- [ ] DeepSeek MoE innovations

---

## Phase 7: Training

### 7.1 Pretraining → [Notes](training/01_Pretraining.md)
- [ ] Next-token prediction objective
- [ ] Training data and tokenization pipeline

### 7.2 Optimizer → [Notes](training/02_Optimizer.md)
- [ ] AdamW and its properties
- [ ] Gradient accumulation

### 7.3 Learning Rate Schedule → [Notes](training/03_Learning_Rate.md)
- [ ] Warmup + cosine decay
- [ ] WSD (Warmup-Stable-Decay)

### 7.4 Regularization → [Notes](training/04_Regularization.md)
- [ ] Dropout, weight decay, gradient clipping

---

## Phase 8: Generation & Decoding

### 8.1 Autoregressive Decoding → [Notes](generation/01_Autoregressive_Decoding.md)
- [ ] Prefill and decode phases
- [ ] KV cache mechanism

### 8.2 Sampling Strategies → [Notes](generation/02_Sampling.md)
- [ ] Temperature, top-k, top-p (nucleus)
- [ ] Repetition penalty

### 8.3 Advanced Decoding → [Notes](generation/03_Advanced_Decoding.md)
- [ ] Beam search
- [ ] Speculative decoding
- [ ] Constrained generation

---

## Phase 9: Linear Attention & Hybrid Architectures

### 9.1 Linear Attention Basics → [Notes](linear_attention/01_Linear_Attention_Basics.md)
- [ ] Kernel trick: drop softmax, use feature maps
- [ ] Dual forms: parallel (training) vs recurrent (inference)
- [ ] RWKV, RetNet

### 9.2 State Space Models → [Notes](linear_attention/02_State_Space_Models.md)
- [ ] Continuous-time SSM, discretization (ZOH)
- [ ] HiPPO initialization
- [ ] S4 and diagonal simplification (S4D)

### 9.3 Mamba → [Notes](linear_attention/03_Mamba.md)
- [ ] Mamba-1: selective SSM (input-dependent B, C, Δ)
- [ ] Mamba-2: State Space Duality (SSM = linear attention)
- [ ] Chunk-wise parallel training

### 9.4 Gated DeltaNet → [Notes](linear_attention/04_Gated_DeltaNet.md)
- [ ] GLA: data-dependent gating, WY representation
- [ ] DeltaNet: delta rule for overwriting associations
- [ ] Gated DeltaNet: unified gating + delta rule (ICLR 2025)

### 9.5 Hybrid Architecture → [Notes](linear_attention/05_Hybrid_Architecture.md)
- [ ] Interleaving softmax + linear attention layers
- [ ] Qwen 3.5: hybrid Gated DeltaNet + sliding window
- [ ] Inference advantages: O(1) memory for linear layers

---

## Priority Matrix

| Topic | Priority | Depth |
|-------|----------|-------|
| Decoder-Only Architecture | ★★★★★ | Deep |
| Attention (MHA/GQA) | ★★★★★ | Deep |
| Position Encoding (RoPE) | ★★★★★ | Deep |
| FFN & SwiGLU | ★★★★☆ | Medium-Deep |
| Transformer Block Structure | ★★★★☆ | Medium-Deep |
| Embedding & Tokenization | ★★★★☆ | Medium |
| Training Algorithms | ★★★☆☆ | Medium |
| Generation & Decoding | ★★★☆☆ | Medium |
| Scaling Laws | ★★★☆☆ | Conceptual |
| MoE | ★★★☆☆ | Conceptual |
| Linear Attention & Hybrid | ★★★★☆ | Deep |

---

## Key Papers

1. **Foundation**: "Attention Is All You Need" (Vaswani et al., 2017)
2. **GPT**: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
3. **Scaling**: "Scaling Laws for Neural Language Models" (Kaplan et al., 2020)
4. **Chinchilla**: "Training Compute-Optimal LLMs" (Hoffmann et al., 2022)
5. **LLaMA**: "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)
6. **RoPE**: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
7. **FlashAttention**: "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)
8. **GQA**: "GQA: Training Generalized Multi-Query Transformers" (Ainslie et al., 2023)
9. **SwiGLU**: "GLU Variants Improve Transformer" (Shazeer, 2020)
10. **MoE**: "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer" (Shazeer et al., 2017)
