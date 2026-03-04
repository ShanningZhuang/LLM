# LLM Algorithms Knowledge Base

A structured collection of notes on Large Language Model algorithms, covering decoder-only architecture, attention mechanisms, position encoding, feed-forward networks, training, and generation.

## Folder Structure

```
LLM/
├── 00_LLM.md                      # Root index file
├── architecture/                   # Bird's eye view
│   ├── 00_Architecture.md
│   ├── 01_Decoder_Only.md
│   ├── 02_Model_Family.md
│   └── 03_Scaling_Laws.md
├── embedding/                      # Input/Output layers
│   ├── 00_Embedding.md
│   ├── 01_Tokenization.md
│   ├── 02_Token_Embedding.md
│   └── 03_Output_Head.md
├── transformer_block/              # Block structure
│   ├── 00_Transformer_Block.md
│   ├── 01_Residual_Connection.md
│   ├── 02_Layer_Normalization.md
│   └── 03_Pre_Norm_Post_Norm.md
├── attention/                      # Attention deep dive
│   ├── 00_Attention.md
│   ├── 01_Self_Attention.md
│   ├── 02_Multi_Head_Attention.md
│   ├── 03_MQA_GQA.md
│   ├── 04_Causal_Mask.md
│   └── 05_Efficient_Attention.md
├── position_encoding/              # Position information
│   ├── 00_Position_Encoding.md
│   ├── 01_Absolute_Encoding.md
│   ├── 02_RoPE.md
│   ├── 03_ALiBi.md
│   └── 04_Context_Extension.md
├── ffn/                            # Feed-forward networks
│   ├── 00_FFN.md
│   ├── 01_MLP.md
│   ├── 02_Activation_Functions.md
│   ├── 03_Gated_FFN.md
│   └── 04_MoE.md
├── training/                       # Training algorithms
│   ├── 00_Training.md
│   ├── 01_Pretraining.md
│   ├── 02_Optimizer.md
│   ├── 03_Learning_Rate.md
│   └── 04_Regularization.md
├── generation/                     # Decoding strategies
│   ├── 00_Generation.md
│   ├── 01_Autoregressive_Decoding.md
│   ├── 02_Sampling.md
│   └── 03_Advanced_Decoding.md
├── linear_attention/              # Linear attention & hybrid
│   ├── 00_Linear_Attention.md
│   ├── 01_Linear_Attention_Basics.md
│   ├── 02_State_Space_Models.md
│   ├── 03_Mamba.md
│   ├── 04_Gated_DeltaNet.md
│   └── 05_Hybrid_Architecture.md
└── images/                         # Diagrams
```

## Naming Conventions

### Folders
- Use lowercase with underscores: `folder_name/`
- Each folder represents a topic area

### Files
- Format: `XX_Topic_Name.md`
- `XX` is a two-digit number for ordering (00, 01, 02, ...)
- `00_*.md` is the index/overview file for each folder
- Use underscores to separate words: `01_Self_Attention.md`
- Keep acronyms uppercase: `MQA_GQA`, `RoPE`, `MoE`, `FFN`

### Index Files
- Each folder has a `00_FolderName.md` as the entry point
- Contains overview and links to subtopics
- Links to parent folder for navigation

## How to Use

This knowledge base is designed to work with [build-your-knowledge](https://github.com/ShanningZhuang/build-your-knowledge), a VitePress-based template that:

1. Auto-generates sidebar from folder structure
2. Supports math equations (KaTeX)
3. Deploys easily to Vercel

### Quick Start

1. Clone the template: `git clone https://github.com/ShanningZhuang/build-your-knowledge.git`
2. Copy this LLM folder into the template
3. Run `npm install && npm run docs:dev`
4. Deploy to Vercel

## Topics Covered

| Topic | Description |
|-------|-------------|
| **Architecture** | Decoder-only design, model families, scaling laws |
| **Embedding** | Tokenization, embedding layers, output head |
| **Transformer Block** | Residual connections, layer normalization, pre/post-norm |
| **Attention** | Self-attention, MHA, MQA/GQA, causal masking, FlashAttention |
| **Position Encoding** | Sinusoidal, RoPE, ALiBi, context extension |
| **FFN** | MLP, activation functions, SwiGLU, Mixture of Experts |
| **Training** | Pretraining, AdamW, learning rate schedules, regularization |
| **Generation** | Autoregressive decoding, sampling, beam search, speculative decoding |
| **Linear Attention** | Linear attention, SSMs, Mamba, Gated DeltaNet, hybrid architectures |

## Generating Content with AI

See [PROMPT.md](PROMPT.md) for a ready-to-use prompt when asking LLMs (Claude, GPT, etc.) to help generate or expand your knowledge base.
