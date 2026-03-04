# Tokenization

> Parent: [Embedding](00_Embedding.md)

## Overview

Tokenization is the process of converting raw text into a sequence of integer IDs that a model can process. It sits at the very beginning of the LLM pipeline -- before any neural network computation. The choice of tokenizer affects vocabulary size, sequence length, multilingual capability, and even model quality. All modern LLMs use **subword tokenization**: a middle ground between character-level (too granular) and word-level (too sparse) representations.

```
Raw text        Tokenizer         Token IDs       Embedding
"unhappiness" ──────────────► ["un", "happi", "ness"] ──► [348, 17456, 1108] ──► vectors
                  (subword)
```

## Why Tokenization Matters

1. **The model never sees text** -- it operates entirely on integer sequences
2. **Vocabulary size** directly determines the embedding matrix size (V x d_model parameters)
3. **Token boundaries** affect what the model can learn (e.g., "New York" as 1 vs 2 tokens)
4. **Sequence length** depends on tokenization granularity -- more tokens = more compute (attention is O(n^2))
5. **Unknown words** must be handled gracefully (subword methods solve this)

## BPE (Byte Pair Encoding)

BPE is the most widely used tokenization algorithm in modern LLMs. Originally a data compression technique, it was adapted for NMT by Sennrich et al. (2016).

### Algorithm

```
Input:  Training corpus
Output: Vocabulary of size V (target)

1. Initialize vocabulary with all individual characters (+ special tokens)
2. Count frequency of every adjacent pair of tokens in corpus
3. Merge the most frequent pair into a new token
4. Add new token to vocabulary
5. Repeat steps 2-4 until vocabulary reaches target size V
```

### Step-by-Step Example

Starting corpus (with word frequencies):

```
"low"  : 5     → l o w
"lower": 2     → l o w e r
"new"  : 6     → n e w
"newer": 3     → n e w e r
"wide" : 3     → w i d e
```

Initial vocabulary: {l, o, w, e, r, n, i, d}

```
Iteration 1:  Most frequent pair = (e, w) → count: 9 (from "new"×6 + "newer"×3)
              Merge: e + w → ew
              Vocabulary: {l, o, w, e, r, n, i, d, ew}
              Corpus:  "low":5→l o w,  "lower":2→l o w e r,
                       "new":6→n ew,   "newer":3→n ew e r,  "wide":3→w i d e

Iteration 2:  Most frequent pair = (n, ew) → count: 9
              Merge: n + ew → new
              Vocabulary: {l, o, w, e, r, n, i, d, ew, new}
              Corpus:  "low":5→l o w,  "lower":2→l o w e r,
                       "new":6→new,    "newer":3→new e r,   "wide":3→w i d e

Iteration 3:  Most frequent pair = (l, o) → count: 7 (from "low"×5 + "lower"×2)
              Merge: l + o → lo
              Vocabulary: {l, o, w, e, r, n, i, d, ew, new, lo}

Iteration 4:  Most frequent pair = (lo, w) → count: 7
              Merge: lo + w → low
              Vocabulary: {..., low}

...continue until target vocab size is reached
```

### BPE Merge Process

```
Original:   "n  e  w  e  r"
                ──┬──
Merge (e,w):  "n  ew  e  r"
              ──┬───
Merge (n,ew): "new  e  r"
                   ──┬──
Merge (e,r):  "new  er"
              ───┬────
Merge (new,er): "newer"          ← single token (if frequent enough)
```

### Merge Rules at Inference

At inference time, the learned merge table is applied greedily:

1. Split input into characters
2. Apply merges in the order they were learned (priority order)
3. Stop when no more merges apply

This deterministic process ensures the same text always produces the same tokens.

## WordPiece (BERT)

WordPiece is similar to BPE but uses a **likelihood-based** criterion instead of raw frequency.

### Key Difference from BPE

| Aspect | BPE | WordPiece |
|--------|-----|-----------|
| Merge criterion | Most frequent pair | Pair that maximizes likelihood of training data |
| Merge score | count(pair) | count(pair) / (count(left) x count(right)) |
| Subword prefix | None | `##` for continuation tokens |
| Used by | GPT, LLaMA, most LLMs | BERT, DistilBERT, Electra |

### Example Tokenization

```
Input:  "unbelievable"
Output: ["un", "##believ", "##able"]
         ↑      ↑          ↑
         start  continuation  continuation
```

The `##` prefix signals that a token is a continuation of the previous word (not a word start).

### Merge Score

For candidate pair (A, B) → AB:

```
score(A, B) = count(AB) / (count(A) * count(B))
```

This favors merging pairs where the combined form is disproportionately common relative to each part individually -- essentially measuring how much the pair co-occurs beyond what chance predicts.

## Unigram (SentencePiece)

Unigram takes the opposite approach: start with a **large** vocabulary and iteratively **prune** it down.

### Algorithm

```
1. Start with a large initial vocabulary (all substrings up to max length)
2. Define a unigram language model: P(token) = count(token) / total_count
3. For each token, compute how much the overall corpus likelihood drops if removed
4. Remove tokens whose removal causes the least likelihood decrease
5. Repeat until vocabulary reaches target size
```

### Subword Regularization

A unique advantage of Unigram: because it models probabilities over segmentations, it can **sample** different tokenizations for the same word during training:

```
"international"  →  ["international"]           P = 0.002
                 →  ["inter", "national"]       P = 0.15
                 →  ["inter", "nation", "al"]   P = 0.12
                 →  ["in", "ter", "nation", "al"] P = 0.03
```

This acts as a data augmentation technique, making the model more robust to different subword splits.

### Probability of a Segmentation

For a segmentation S = (x_1, x_2, ..., x_n):

```
P(S) = product_{i=1}^{n} P(x_i)
```

The most likely segmentation is found via the Viterbi algorithm (dynamic programming).

## SentencePiece

SentencePiece is not a tokenization algorithm itself but a **framework** that implements both BPE and Unigram in a language-independent way.

### Key Properties

| Property | Traditional Tokenizers | SentencePiece |
|----------|----------------------|---------------|
| Pre-tokenization | Whitespace splitting required | Treats input as raw byte stream |
| Whitespace | Discarded | Preserved as `_` (U+2581) |
| Language dependency | Often language-specific rules | Fully language-independent |
| Input format | Pre-segmented words | Raw text |

### Example

```
Input:     "Hello world"
Traditional: ["Hello", "world"]      ← whitespace is implicit
SentencePiece: ["_Hello", "_world"]   ← whitespace is explicit with _
```

This matters for languages like Japanese and Chinese where words are not separated by spaces, and for detokenization (you can perfectly reconstruct the original text).

## Byte-Level BPE (GPT-2)

GPT-2 introduced a variant of BPE that operates at the **byte level**, using the 256 base byte values as the initial vocabulary.

### Key Advantage: No Unknown Tokens

```
Traditional BPE:  "Привет" → [UNK]     ← unknown if not in training data
Byte-level BPE:   "Привет" → [0xD0, 0x9F, 0xD1, 0x80, ...]  → merged subwords
```

Since any text can be encoded as bytes, byte-level BPE can represent **any** Unicode string. Frequent byte sequences get merged into subword tokens, while rare characters fall back to raw bytes.

### Trade-off

```
Pros:
  + Zero unknown tokens for any input
  + No pre-processing or normalization required
  + Uniform treatment of all languages and scripts

Cons:
  - Rare scripts produce longer token sequences (1 char → multiple byte tokens)
  - Base vocabulary starts at 256 (bytes) instead of ~26 (English alphabet)
```

## Vocabulary Size Trade-offs

| Vocab Size | Avg Tokens per Word | Embedding Params (d=4096) | Pros | Cons |
|-----------|--------------------|-----------------------------|------|------|
| 4K | ~3.5 | 16M | Small model, fast softmax | Very long sequences |
| 32K | ~1.3 | 131M | Good balance (LLaMA) | -- |
| 50K | ~1.2 | 205M | Slightly shorter sequences (GPT-2) | More parameters |
| 100K | ~1.1 | 410M | Short sequences, multilingual (GPT-4) | Large embedding table |
| 256K | ~1.0 | 1.05B | Near word-level for common words | Huge embedding, sparse updates |

Key trade-off:

```
Small Vocabulary (e.g., 4K)          Large Vocabulary (e.g., 256K)
├── Longer sequences                 ├── Shorter sequences
├── More compute (O(n^2) attention)  ├── Less compute per sample
├── Fewer embedding parameters       ├── More embedding parameters
├── Better subword generalization    ├── More whole-word tokens
└── Slower training (more steps/doc) └── Sparse gradient updates
```

## Tokenizer Comparison Across Models

| Model | Algorithm | Vocab Size | Framework | Notes |
|-------|-----------|-----------|-----------|-------|
| GPT-2 | Byte-level BPE | 50,257 | Custom | 256 bytes + 50K merges + 1 special |
| GPT-3/3.5 | Byte-level BPE | 50,257 | tiktoken | Same as GPT-2 |
| GPT-4 | Byte-level BPE | ~100,000 | tiktoken | Larger vocab, better multilingual |
| BERT | WordPiece | 30,522 | HuggingFace | Case-sensitive, ## continuation |
| LLaMA / LLaMA-2 | BPE | 32,000 | SentencePiece | Byte-fallback for UNK |
| LLaMA-3 | Byte-level BPE | 128,256 | tiktoken | 4x larger vocab than LLaMA-2 |
| Mistral-7B | BPE | 32,000 | SentencePiece | Same tokenizer as LLaMA |
| Gemma | BPE | 256,000 | SentencePiece | Very large vocab for multilingual |
| DeepSeek-V2 | Byte-level BPE | 100,015 | Custom | Similar approach to GPT-4 |

## Code Example: Tokenization with tiktoken

```python
import tiktoken

# Load the GPT-4 tokenizer
enc = tiktoken.encoding_for_model("gpt-4")

text = "Hello, world! Tokenization is fascinating."
tokens = enc.encode(text)
print(f"Text:   {text}")
print(f"Tokens: {tokens}")
print(f"Count:  {len(tokens)} tokens")
# Text:   Hello, world! Tokenization is fascinating.
# Tokens: [9906, 11, 1917, 0, 9857, 2065, 374, 27387, 13]
# Count:  9 tokens

# Decode individual tokens to see subwords
for t in tokens:
    print(f"  {t:>6d} → '{enc.decode([t])}'")
#    9906 → 'Hello'
#      11 → ','
#    1917 → ' world'
#       0 → '!'
#    9857 → ' Token'
#    2065 → 'ization'
#     374 → ' is'
#   27387 → ' fascinating'
#      13 → '.'

# Compare tokenizers
for model in ["gpt-3.5-turbo", "gpt-4"]:
    enc = tiktoken.encoding_for_model(model)
    n = len(enc.encode(text))
    print(f"{model}: {n} tokens (vocab size: {enc.n_vocab})")
```

## Summary

```
Tokenization Methods:

BPE ──────────── Bottom-up: merge frequent pairs
  │                 Used by: GPT-2, GPT-3, LLaMA
  │
  ├─ Byte-level ── BPE on raw bytes, no UNK tokens
  │                 Used by: GPT-2, GPT-4, LLaMA-3
  │
WordPiece ──────── Like BPE but likelihood-based merges
  │                 Used by: BERT
  │
Unigram ────────── Top-down: prune large vocabulary
                    Used by: T5, mBART (via SentencePiece)

SentencePiece ──── Framework implementing BPE or Unigram
                    Language-independent, raw byte input
                    Used by: LLaMA, Mistral, T5, Gemma
```

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [BPE for NMT (Sennrich et al., 2016)](https://arxiv.org/abs/1508.07909) | Applied BPE to neural machine translation |
| [WordPiece (Schuster & Nakajima, 2012)](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf) | Likelihood-based subword merging |
| [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) | Language-independent tokenization framework |
| [Subword Regularization (Kudo, 2018)](https://arxiv.org/abs/1804.10959) | Unigram LM with multiple segmentations |
| [GPT-2 (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | Byte-level BPE, no UNK tokens |

## Related

- [Token Embedding](02_Token_Embedding.md) - How token IDs become vectors
- [Output Head](03_Output_Head.md) - Converting hidden states back to vocabulary logits
