# Knowledge Base Generation Prompt

Use this prompt when asking LLMs to help generate or expand your knowledge base.

---

## System Prompt

```
You are helping me build a structured knowledge base about LLM algorithms. Follow these conventions strictly:

### Folder Structure
- Each topic area is a folder with lowercase and underscores: `topic_name/`
- Folders are NOT numbered (e.g., `attention/`, not `01_attention/`)
- Each folder contains markdown files numbered for reading order

### File Naming Convention
- Format: `XX_Topic_Name.md` where XX is a two-digit number (00, 01, 02, ...)
- `00_*.md` is always the index/overview file for the folder
- Use underscores between words: `01_Self_Attention.md`
- Keep acronyms uppercase: `MQA_GQA`, `RoPE`, `MoE`, `FFN`
- Examples:
  - `00_Attention.md` (index file)
  - `01_Self_Attention.md`
  - `02_Multi_Head_Attention.md`
  - `03_MQA_GQA.md`

### Index File Format (00_*.md)
Each folder must have an index file with this structure:

```markdown
# Topic Title

> Parent: [Parent Topic](../00_Parent.md)

## Overview

Brief description of what this topic covers.

## Topics

1. **Subtopic 1** - Brief description
2. **Subtopic 2** - Brief description
3. **Subtopic 3** - Brief description
```

### Content File Format (01_*.md, 02_*.md, etc.)
```markdown
# Topic Title

> Parent: [Parent Index](00_Index.md)

## Overview

Introduction to the topic.

## Section 1

Content...

## Section 2

Content...

## Related

- [Related Topic 1](01_Related.md) - Description
- [Related Topic 2](02_Another.md) - Description
```

### Linking Rules
- Always use relative paths: `./`, `../`
- Link to the exact filename with number prefix
- Include the .md extension
- Examples:
  - Same folder: `[Topic](01_Topic.md)`
  - Parent folder: `[Parent](../00_Parent.md)`
  - Sibling folder: `[Other](../other_folder/00_Other.md)`

### Content Guidelines
- Use clear, concise language
- Include code examples where relevant (Python/PyTorch)
- Use ASCII diagrams for architecture/flow visualization
- Add practical examples and use cases
- Structure content from fundamentals to advanced
- **Math & Equations**: Always use LaTeX math notation, never code blocks for equations
  - Inline math: `$x = y + z$` renders as math, not code
  - Display math: use `$$` blocks for standalone equations
  - Example — write `$L = -\sum_{t} \log P(x_t | x_{<t})$` instead of putting equations in ``` code fences
  - Use `\text{}` for words inside math: `$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{KL}}$`
  - Common patterns:
    - Softmax: `$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$`
    - Attention: `$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$`
    - Loss functions: `$$\mathcal{L} = -\mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)] - \mathbb{E}_{z \sim p_z} [\log(1 - D(G(z)))]$$`
```

---

## Example User Prompts

### Creating a New Topic Folder

```
Create a knowledge base folder for "position_encoding" with the following structure:
- 00_Position_Encoding.md (index)
- 01_Absolute_Encoding.md
- 02_RoPE.md
- 03_ALiBi.md
- 04_Context_Extension.md

Follow the knowledge base conventions. The parent folder is LLM (link to ../00_LLM.md).
```

### Expanding an Existing Topic

```
Expand the file `attention/03_MQA_GQA.md` with:
- More detailed explanation of Grouped-Query Attention
- KV cache memory comparison table
- Code examples in PyTorch
- ASCII diagram showing head grouping

Follow the knowledge base conventions and maintain links to related files.
```

### Adding a New Subtopic

```
Add a new file `06_Sliding_Window.md` to the `attention/` folder.
Cover: sliding window attention (Mistral), local attention patterns, dilated attention.
Link it from 00_Attention.md and add cross-references to 05_Efficient_Attention.md.
Follow the knowledge base conventions.
```

---

## Quick Reference

| Element | Convention | Example |
|---------|------------|---------|
| Folder | lowercase_underscores | `position_encoding/` |
| Index file | 00_FolderName.md | `00_Position_Encoding.md` |
| Content file | XX_Topic_Name.md | `02_RoPE.md` |
| Acronyms | UPPERCASE | `MQA_GQA`, `FFN` |
| Links | relative with .md | `[Link](../00_LLM.md)` |

---

## Full Example Structure

```
LLM/
├── 00_LLM.md                   # Root index
├── attention/
│   ├── 00_Attention.md          # Index: links to ../00_LLM.md
│   ├── 01_Self_Attention.md     # Links to 00_Attention.md
│   ├── 02_Multi_Head_Attention.md
│   ├── 03_MQA_GQA.md
│   ├── 04_Causal_Mask.md
│   └── 05_Efficient_Attention.md
├── ffn/
│   ├── 00_FFN.md
│   ├── 01_MLP.md
│   ├── 02_Activation_Functions.md
│   ├── 03_Gated_FFN.md
│   └── 04_MoE.md
└── README.md                    # GitHub readme (not part of docs)
```
