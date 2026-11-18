
# SextaFeira-V1 Nano

**SextaFeira-V1** is a GPT-2-inspired decoder-only Transformer language model, incorporating modern architectural improvements such as **Grouped-Query Attention (GQA), RoPE, RMSNorm, QK-Norm, and weight tying**.

---

## 1. Architecture

```text
Token IDs
   |
   v
Embedding -------------------------------+
   |                                     |
   v                                     |
+----------------------------------+     |
|       Transformer Block × 6     |     |
|                                  |     |
|  RMSNorm (Attn Input)            |     |
|       |                          |     |
|       v                          |     |
|  GQA + QK-Norm + RoPE             |     |
|       |                          |     |
|  RMSNorm (Attn Output)           |     |
|       |                          |     |
|       +---------------- Residual |     |
|                                  |     |
|  RMSNorm (FFN Input)             |     |
|       |                          |     |
|       v                          |     |
|  FFN (4 × d_model, GELU)          |     |
|       |                          |     |
|  RMSNorm (FFN Output)            |     |
|       |                          |     |
|       +---------------- Residual |     |
+----------------------------------+     |
   |                                     |
   v                                     |
Final RMSNorm                            |
   |                                     |
   v                                     |
LM Head <--------------------------------+
   |
   v
Logits (49,152)
```

### Components

| Component | Description |
|---|---|
| **Embedding** | Maps token IDs into 384-dimensional vectors |
| **Transformer Blocks** | Stack of 6 identical blocks, each containing attention and a feed-forward network |
| **Attention** | Grouped-Query Attention (GQA), with 6 query heads and 2 key/value heads |
| **QK-Norm** | Applies RMSNorm independently to query and key representations before RoPE |
| **FFN** | Position-wise MLP with a 4× expansion, GELU activation, and dropout |
| **Final RMSNorm** | Normalizes the final hidden representations |
| **LM Head** | Projects hidden states into the vocabulary, sharing its weights with the embedding layer |

---

## 3. Key Features

| Feature | Where it is applied | Purpose |
|---|---|---|
| **GQA** | Attention | Shares K/V heads across multiple query heads, reducing KV-cache memory usage |
| **RoPE** | Queries and Keys | Injects positional information into attention |
| **QK-Norm** | Queries and Keys | Normalizes Q/K representations to improve attention stability |
| **RMSNorm** | Input of each sub-block | Normalizes the input representations before attention and FFN |
| **RMSNorm** | Output of each sub-block | Normalizes branch outputs before adding them to the residual stream |
| **Weight Tying** | Embedding and LM Head | Shares parameters, reducing the total parameter count |
| **KV Cache** | Autoregressive inference | Reuses previous keys and values during incremental decoding |

### GQA Configuration

The model uses:

- Query heads: 6
- Key/Value heads: 2
- Head dimension: 64
- Query heads per KV head: 3

Each KV head is shared by 3 query heads, reducing the number of stored K/V heads compared with standard Multi-Head Attention.

### KV Cache

The model supports autoregressive inference using a KV cache.

- **Prefill:** Processes the initial prompt and stores the keys and values.
- **Decoding:** Processes one new token at a time, reusing the cached keys and values.
- **Key cache:** Stores RoPE-rotated keys.
- **Value cache:** Stores the original projected values.

The current implementation uses dynamic cache concatenation with `torch.cat`.

---

## 4. Hyperparameters

| Parameter | Value |
|---|---|
| `vocab_size` | 49,152 |
| `max_seq_len` | 512 |
| `d_model` | 384 |
| `n_blocks` | 6 |
| `n_heads` | 6 |
| `n_kv_heads` | 2 |
| `head_dim` | 64 *(derived: `d_model / n_heads`)* |
| `ffn_hidden` | 1,536 *(derived: `4 × d_model`)* |
| `dropout` | 0.15 |
| `rope_base` | 10,000 |
| `rms_norm_eps` | 1e-6 |

**Estimated model size: ~28.34M parameters**, including the shared embedding/LM Head weights and the linear layer biases.

---

## 5. Weight Initialization

| Layer | Distribution | Notes |
|---|---|---|
| **Linear layers** | `N(0, 0.02)` | Mean = 0, standard deviation = 0.02 |
| **Embeddings** | `N(0, 0.02)` | Mean = 0, standard deviation = 0.02 |
| **Norm layers** | Constant | RMSNorm weights initialized to 1 |
| **Attention output projection** | `N(0, 0.02 / √(2 · n_blocks))` | Scaled residual projection |
| **FFN down-projection** | `N(0, 0.02 / √(2 · n_blocks))` | Scaled residual projection |


