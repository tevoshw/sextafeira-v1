O SextaFeiraV1 tem suas config e arquiteturas baseado no GPT2


# Struct

- Embedding
- Blocos de Transformers com:
    - GQA
    - FFN (4 * d_model em up-projection)
- LM Head possuindo os mesmos pesos do Embedding (economia )

Com características
- RoPE
- RMSNorm em:
    - QK matrix
    - Input of sub-blocks
    - Output of sub-blocks

# Params 
vocab_size = 49152
d_model = 1024
n_blocks = 24
n_heads = 16
dropout = 0.05


# Inicialização de pesos
- Camadas lineares: 
    - média == 0
    - std == 0.2
- Camadas Embeddings
    - média == 0
    - std == 0.2
- Camadas Norm
    - bias = inicializados com 0
    - pesos = inicializados com 1
- Projection in ATTN
    - média == 0
    - std == 0.02 / sqrt(2 * n_blocks)
- Feed Forward (Down-Projection)
    - média == 0
    - std == 0.02 / sqrt(2 * n_blocks)