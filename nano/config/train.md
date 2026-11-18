
#Training Configuration

SextaFeira-V1 was trained from scratch on a **NVIDIA L4 GPU using Google Colab**, with an effective batch size of 98,304 tokens per optimization step.

### Hardware & Training Setup

| Parameter | Value |
|---|---|
| GPU | NVIDIA L4 (Google Colab) |
| Model parameters | ~28.3M |
| Training steps | 50,000 |
| Batch size | 48 |
| Gradient accumulation | 4 |
| Sequence length | 512 |
| Effective batch size | 192 sequences |
| Tokens per micro-batch | 24,576 |
| Tokens per optimization step | 98,304 |
| Estimated total tokens processed | ~4.92B |
| Training duration | ~21 hours |

### Data Loading

| Parameter | Value |
|---|---|
| CPU cores | 12 |
| Dataset shards | 38 |
| Training workers | 10 |
| Evaluation workers | 2 |

### Throughput

| Metric | Performance |
|---|---|
| GPU throughput | ~0.07M tokens/s |
| GPU processing speed | ~2.9 micro-batches/s |
| Data loader throughput | ~0.01M tokens/s |
| Data loader speed | ~0.5 batches/s |
| Loader / GPU ratio | 0.19x |

The data loading pipeline was a potential bottleneck, with the loader delivering approximately 19% of the measured GPU processing throughput.

---

## 7. Training Results

### Final Metrics

| Metric | Value |
|---|---:|
| Training Loss | 2.6949 |
| Validation Loss | 2.9537 |
| Validation Perplexity | 19.2 |
| Training duration | ~21 hours |
| Optimization steps | 50,000 |

The model achieved a final validation loss of **2.9537**, corresponding to a perplexity of approximately **19.2**.

---

## 8. Modern Techniques

SextaFeira-V1 incorporates several modern Transformer techniques to improve architectural efficiency, numerical stability, and autoregressive inference.

| Technique | Application | Purpose |
|---|---|---|
| **Weight Initialization** | Linear layers and residual projections | Controls initial parameter scales and residual magnitudes |
| **Grouped-Query Attention (GQA)** | Attention mechanism | Reduces the number of KV heads and KV-cache memory requirements |
| **Rotary Positional Embeddings (RoPE)** | Queries and Keys | Encodes positional information directly into attention |
| **RMSNorm** | Transformer sub-blocks | Normalizes hidden representations with reduced computational overhead |
| **QK-Norm** | Query and Key projections | Stabilizes attention by normalizing Q/K representations |
| **Weight Tying** | Embedding and LM Head | Shares embedding and output projection parameters |
| **KV Cache** | Autoregressive inference | Reuses previously computed keys and values, avoiding redundant computation |

These techniques combine architectural improvements with inference optimizations, allowing the model to use modern Transformer design principles despite its relatively small parameter count.
