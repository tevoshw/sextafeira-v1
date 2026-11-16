import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def precompute_rope(head_dim, max_seq_len, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # [hd/2]
    pos = torch.arange(max_seq_len).float()
    freqs = torch.outer(pos, inv_freq)            # [T, hd/2]
    emb = torch.cat((freqs, freqs), dim=-1)       # [T, hd]
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    # x: [B, H, T, hd] | cos, sin: [T, hd] (broadcast)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    """Atenção causal com GQA (n_kv_heads < n_heads), QK-Norm e RoPE.

    n_kv_heads=None  -> MHA (n_kv_heads = n_heads)
    n_kv_heads=1     -> MQA
    """

    def __init__(self, d_model, n_heads, n_kv_heads=None):
        super().__init__()
        n_kv_heads = n_kv_heads or n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

        self.wQ = nn.Linear(d_model, n_heads * self.head_dim)
        self.wK = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.wV = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.wO = nn.Linear(n_heads * self.head_dim, d_model)

        self.rms_norm_attn_q = nn.RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=True)
        self.rms_norm_attn_k = nn.RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=True)

    def forward(self, x, cos, sin, cache=None):
        b, t, d = x.shape

        # Q: [B, H, T, hd] | K, V: [B, H_kv, T, hd]
        Q = self.wQ(x).reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.wK(x).reshape(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = self.wV(x).reshape(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK-Norm (antes do RoPE)
        Q = self.rms_norm_attn_q(Q)
        K = self.rms_norm_attn_k(K)

        # RoPE em Q e K (o cache guarda K já rotacionado, com H_kv cabeças)
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        if cache is not None:
            assert t == 1, "com cache, só decode de 1 token por vez"
            K = torch.cat((cache[0], K), dim=2)
            V = torch.cat((cache[1], V), dim=2)
            out = F.scaled_dot_product_attention(Q, K, V, enable_gqa=True)
        else:
            out = F.scaled_dot_product_attention(Q, K, V, is_causal=True, enable_gqa=True)
        new_cache = (K, V)

        out = out.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim)
        return self.wO(out), new_cache


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None, dropout=0.05):
        super().__init__()
        self.attention = Attention(d_model, n_heads, n_kv_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),   # 0 (up-projection)
            nn.Dropout(dropout),               # 1
            nn.GELU(),                         # 2
            nn.Linear(d_model * 4, d_model),   # 3 (down-projection)
            nn.Dropout(dropout),               # 4
        )
        self.rms_norm_attn_in = nn.RMSNorm(d_model, eps=1e-6, elementwise_affine=True)
        self.rms_norm_attn_out = nn.RMSNorm(d_model, eps=1e-6, elementwise_affine=True)
        self.rms_norm_ffn_in = nn.RMSNorm(d_model, eps=1e-6, elementwise_affine=True)
        self.rms_norm_ffn_out = nn.RMSNorm(d_model, eps=1e-6, elementwise_affine=True)

    def forward(self, x, cos, sin, cache=None):
        # Pre-norm + post-norm em attn/FFN (estilo Gemma 2)
        attn_out, cache = self.attention(self.rms_norm_attn_in(x), cos, sin, cache)
        x = x + self.rms_norm_attn_out(attn_out)
        x = x + self.rms_norm_ffn_out(self.feedforward(self.rms_norm_ffn_in(x)))
        return x, cache


class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, n_blocks, n_heads,
                 dropout, n_kv_heads=None):
        super().__init__()
        self.max_seq_len = max_seq_len

        # 1. Embedding (sem pos_emb: a posição vem do RoPE)
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 2. RoPE: tabelas pré-calculadas (não entram no state_dict)
        cos, sin = precompute_rope(d_model // n_heads, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # 3. Blocos
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, dropout)
            for _ in range(n_blocks)
        ])

        # 4. Head
        self.norm_final = nn.RMSNorm(d_model, eps=1e-6, elementwise_affine=True)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight  # weight tying

        # 5. Inicialização de pesos
        self.apply(self.init_weights)
        std = 0.02 / math.sqrt(2 * n_blocks)
        for block in self.blocks:
            nn.init.normal_(block.attention.wO.weight, mean=0.0, std=std)
            nn.init.normal_(block.feedforward[3].weight, mean=0.0, std=std)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, caches=None):
        t = idx.size(1)
        offset = 0 if caches is None else caches[0][0].size(2)
        assert offset + t <= self.max_seq_len, "sequência maior que max_seq_len"

        cos = self.rope_cos[offset:offset + t]
        sin = self.rope_sin[offset:offset + t]

        x = self.embedding(idx)

        new_caches = []
        for i, block in enumerate(self.blocks):
            x, c = block(x, cos, sin, None if caches is None else caches[i])
            new_caches.append(c)

        return self.head(self.norm_final(x)), new_caches


if __name__ == "__main__":
    VOCAB = 49152

    model = GPT(
        vocab_size=VOCAB,
        max_seq_len=512,
        d_model=384,
        n_blocks=6,
        n_heads=6,
        n_kv_heads=2,      # 3 cabeças de Q por par K/V; precisa dividir n_heads
        dropout=0.15,
    ).to(device)

    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parâmetros")

    # Teste: decode com KV cache == forward completo
    model.eval()
    idx = torch.randint(0, VOCAB, (1, 16), device=device)
    with torch.no_grad():
        full, _ = model(idx)
        _, caches = model(idx[:, :-1])
        step, _ = model(idx[:, -1:], caches)
    print(torch.allclose(full[:, -1], step[:, 0], atol=1e-4))   # True
    print(caches[0][0].shape)   # [1, 2, 15, 64]