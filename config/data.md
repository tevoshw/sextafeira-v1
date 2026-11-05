# 1. Pre-Training Data

**SextaFeira-V1** was pre-trained on **GigaVerbo-v2**, a large-scale Portuguese corpus released by [Polygl0t](https://huggingface.co/Polygl0t). The dataset itself was used as-is, with a basic data engineering pipeline applied before training.

### Overview

| Property | Value |
|---|---|
| Dataset | GigaVerbo-v2 (Polygl0t) |
| Total tokens | ~7 billion |
| Total size | ~14 GB |
| Shards | 7 |
| Chunks per shard | 10 |
| Chunk size | ~200 MB |
| Tokenizer | Tucano2-0.6B-Base (vocab size 49,152) |

### Data Layout

The ~7B tokens are split into **7 shards**, each containing **10 chunks** of ~200 MB (70 chunks in total). This keeps loading memory-friendly: chunks are streamed one at a time instead of loading the whole dataset into RAM, and training can resume from a specific shard/chunk after an interruption.

### Data Engineering

Before training, the raw data went through a **basic data engineering pipeline** that cleans, processes, tokenizes, and organizes it into the shard/chunk layout described above. The pipeline is intentionally simple and is fully documented, with runnable code, in this notebook:

**[Data engineering notebook (Google Colab)](https://colab.research.google.com/drive/1XhX8uBxzWMcvkVIZd6GAuzN6Qi2GSIHE)**

# 2. Continued Pre-Trainig (CPT) Data
.

# 3. Supervised Fine-Tunninf (SFT) Data
.

# 4. Alingment Data
.