# SextaFeiraV1

**SextaFeiraV1** is a decoder-only Transformer language model whose configuration and architecture are based on **GPT-2**, with a few modern upgrades (GQA, RoPE, RMSNorm, weight tying).

---

## Architecture

```
Token IDs
   │
   ▼
Embedding ──────────────────────────────┐ (shared weights)
   │                                    │
   ▼                                    │
┌──────────────────────────────┐        │
│ Transformer Block  × 24      │        │
│  ├─ RMSNorm → GQA (+ RoPE)   │        │
│  │            → RMSNorm → +  │        │
│  └─ RMSNorm → FFN            │        │
│               → RMSNorm → +  │        │
└──────────────────────────────┘        │
   │                                    │
   ▼                                    │
Final Norm                              │
   │                                    │
   ▼                                    │
LM Head ◄───────────────────────────────┘
   │
   ▼
Logits
```

### Components

| Component | Description |
|---|---|
| **Embedding** | Maps token IDs to `d_model`-dimensional vectors |
| **Transformer Blocks** | Stack of identical blocks, each with attention and a feed-forward network |
| **Attention** | **GQA** (Grouped-Query Attention): query heads share key/value heads, reducing KV-cache size and memory bandwidth |
| **FFN** | Position-wise MLP with a **4 × `d_model`** up-projection, followed by a down-projection back to `d_model` |
| **LM Head** | Output projection to the vocabulary, **tied to the Embedding weights** (saves parameters and often improves quality in small models) |

### Key Features

| Feature | Where it is applied | Why |
|---|---|---|
| **RoPE** | Queries and keys | Encodes relative positions directly in attention; generalizes better to longer contexts than learned absolute positions |
| **RMSNorm** | QK matrices (**QK-Norm**) | Stabilizes attention logits and prevents them from blowing up |
| **RMSNorm** | Input of each sub-block | Stabilizes the signal entering attention and FFN |
| **RMSNorm** | Output of each sub-block | Controls the magnitude added back to the residual stream |

---

## Hyperparameters

| Parameter | Value |
|---|---|
| `vocab_size` | 49,152 |
| `d_model` | 1,024 |
| `n_blocks` | 24 |
| `n_heads` | 16 |
| `head_dim` | 64 *(derived: `d_model / n_heads`)* |
| `ffn_hidden` | 4,096 *(derived: `4 × d_model`)* |
| `dropout` | 0.05 |

> **Size estimate:** roughly **~350M parameters**. This is an upper bound that assumes full-size K/V projections; GQA makes the real number slightly smaller. The tied embedding (≈50M) is counted only once.

---

## Weight Initialization

| Layer | Distribution | Notes |
|---|---|---|
| **Linear layers** | `N(0, std)` | mean = 0, std = **0.2** ¹ |
| **Embeddings** | `N(0, std)` | mean = 0, std = **0.2** ¹ |
| **Norm layers** | constant | weights = **1**, bias = **0** (if a bias exists) |
| **Attention output projection** | `N(0, 0.02 / √(2 · n_blocks))` | ≈ 0.0029 for 24 blocks |
| **FFN down-projection** | `N(0, 0.02 / √(2 · n_blocks))` | ≈ 0.0029 for 24 blocks |

### Why scale the residual projections?

Each block adds **two** contributions to the residual stream (attention and FFN). Across `n_blocks` blocks that is `2 · n_blocks` additions, so the variance grows with depth. Dividing the std by `√(2 · n_blocks)` keeps the residual stream's scale roughly constant at initialization. This is the same trick used by GPT-2.
