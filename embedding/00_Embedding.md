# Embedding & Output Layers

> Parent: [LLM Algorithms](../00_LLM.md)

## Overview

The embedding layer is the first component of an LLM — it converts discrete token IDs into continuous vectors that the Transformer can process. At the output end, the LM head projects hidden states back to vocabulary logits. These two layers bookend the entire model and together account for a significant fraction of parameters (especially with large vocabularies).

## Input → Output Flow

```
Raw text: "Hello world"
         │
    ┌────▼─────┐
    │ Tokenizer │   Converts text → token IDs
    └────┬─────┘   "Hello" → 15496, "world" → 995
         │
    ┌────▼──────────────┐
    │ Embedding Layer    │   token_id → d_model vector
    │ E ∈ ℝ^{V × d}     │   lookup table (no computation)
    └────┬──────────────┘
         │
    ┌────▼──────────────┐
    │ Transformer Blocks │   N layers of attention + FFN
    │ (× N layers)       │
    └────┬──────────────┘
         │
    ┌────▼──────────────┐
    │ LM Head            │   d_model → V logits
    │ W ∈ ℝ^{V × d}     │   often tied with E
    └────┬──────────────┘
         │
    ┌────▼──────────────┐
    │ Softmax + Sample   │   logits → probabilities → token
    └───────────────────┘
```

## Topics

| Topic | File | Description |
|-------|------|-------------|
| Tokenization | [01_Tokenization.md](01_Tokenization.md) | BPE, WordPiece, SentencePiece algorithms |
| Token Embedding | [02_Token_Embedding.md](02_Token_Embedding.md) | Embedding layer, vocab size, dimension |
| Output Head | [03_Output_Head.md](03_Output_Head.md) | LM head, weight tying |

## Parameter Count

For a model with vocabulary size V and embedding dimension d:

| Component | Parameters | Example (V=32K, d=4096) |
|-----------|-----------|------------------------|
| Embedding | V × d | 131M |
| LM Head | V × d (or 0 if tied) | 131M (or 0) |
| **Total** | **V × d (or 2V × d)** | **131M–262M** |

With weight tying, embedding + LM head share the same matrix, saving ~131M parameters for a 7B model.

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [BPE (Sennrich et al., 2016)](https://arxiv.org/abs/1508.07909) | Byte Pair Encoding for NMT |
| [SentencePiece (Kudo, 2018)](https://arxiv.org/abs/1808.06226) | Language-independent tokenizer |
| [Weight Tying (Press & Wolf, 2017)](https://arxiv.org/abs/1608.05859) | Shared embedding/output weights |
