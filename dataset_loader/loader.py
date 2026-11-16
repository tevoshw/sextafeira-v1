"""Loader em streaming para o gigaverbo-v2-synth.

Fluxo:
    1) find_max_batch_size(model, ...)   -> maior micro-batch que cabe na GPU
    2) build_loaders(batch_size)         -> train (80%), val (10%), test (10%), tokenizer
    3) benchmark_loader(train_loader)    -> confere se o loader alimenta a GPU
    4) to_device(batch, device)          -> move (x, y) para a GPU como int64
"""
import gc
import os
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # paralelismo é feito pelos workers

DATASET = "Polygl0t/gigaverbo-v2-synth"
TOKENIZER = "Polygl0t/Tucano2-0.6B-Base"
RANGES = {"train": (0, 80), "val": (80, 90), "test": (90, 100)}  # buckets de 0 a 99


def bucket(doc_id, salt="split-v1"):
    """Bucket determinístico (0-99) a partir do id do documento."""
    return zlib.crc32(f"{salt}:{doc_id}".encode()) % 100


# --------------------------------------------------------------------------- #
# Dataset: streaming + split por hash + tokenização + packing
# --------------------------------------------------------------------------- #
class PackedStream(IterableDataset):
    """Cada item é um batch pronto (x, y) com shape [batch_size, seq_len] (int32),
    onde y é x deslocado em 1 token. Documentos são separados por EOS."""

    def __init__(self, split, seq_len, batch_size, tokenizer,
                 shuffle_buffer=0, seed=42, tok_batch=256):
        self.split, self.L, self.B = split, seq_len, batch_size
        self.tok = tokenizer
        self.eos = np.array([tokenizer.eos_token_id], dtype=np.int32)
        self.shuffle_buffer, self.seed, self.tok_batch = shuffle_buffer, seed, tok_batch

    def __iter__(self):
        lo, hi = RANGES[self.split]
        ds = load_dataset(DATASET, split="train", streaming=True)
        ds = ds.select_columns(["id", "text"])
        if self.shuffle_buffer:
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
        # a divisão de shards entre os workers do DataLoader é feita pelo próprio HF

        L, B = self.L, self.B
        need = B * L + 1
        chunks, total, texts = [], 0, []

        for ex in ds:
            if not (lo <= bucket(ex["id"]) < hi):   # filtra ANTES de tokenizar
                continue
            texts.append(ex["text"])
            if len(texts) < self.tok_batch:
                continue

            ids = self.tok(texts, add_special_tokens=False,
                           return_attention_mask=False)["input_ids"]
            texts = []
            for t in ids:
                a = np.asarray(t, dtype=np.int32)
                chunks += [a, self.eos]
                total += len(a) + 1

            if total >= need:
                arr = np.concatenate(chunks)
                while len(arr) >= need:
                    x = torch.from_numpy(arr[:B * L].reshape(B, L).copy())
                    y = torch.from_numpy(arr[1:B * L + 1].reshape(B, L).copy())
                    yield x, y
                    arr = arr[B * L:]
                chunks, total = [arr], len(arr)
        # a sobra final (menos de um batch) é descartada


class CachedEval:
    """Lê n_batches do loader (streaming) na 1ª iteração e reaproveita da RAM nas
    seguintes. Evita re-varrer o dataset a cada avaliação e deixa a val comparável."""

    def __init__(self, loader, n_batches):
        self.loader, self.n, self._cache = loader, n_batches, None

    def __iter__(self):
        if self._cache is None:
            cache = []
            for batch in self.loader:
                cache.append(batch)
                if len(cache) >= self.n:
                    break
            self._cache = cache
        return iter(self._cache)

    def __len__(self):
        return self.n if self._cache is None else len(self._cache)


def build_loaders(batch_size, seq_len=512, train_workers=None, eval_workers=2,
                  val_tokens=10_000_000, test_tokens=None,
                  shuffle_buffer=10_000, seed=42, tok_batch=256, prefetch_factor=8):
    """Retorna (train_loader, val_loader, test_loader, tokenizer).

    train: 80% dos documentos, streaming infinito-ish com shuffle.
    val:   10%, fixo em `val_tokens` tokens e cacheado em RAM (comparável entre avaliações).
    test:  10%; por padrão streaming completo. Passe test_tokens=N para fixar e cachear.
    """
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    tok.model_max_length = 10**9  # evita warning em documentos longos
    assert tok.eos_token_id is not None, "tokenizer sem EOS: defina outro separador"

    n_shards = load_dataset(DATASET, split="train", streaming=True).n_shards
    cpu = os.cpu_count() or 4
    if train_workers is None:
        train_workers = max(1, cpu - 2)          # deixa núcleos para o processo principal
    train_workers = min(train_workers, n_shards)
    eval_workers = min(eval_workers, n_shards)
    print(f"cpus={cpu} | shards={n_shards} | workers train={train_workers} eval={eval_workers} "
          f"| {batch_size * seq_len:,} tokens/batch")

    def mk(split, shuffle, nw, persistent, pin):
        ds = PackedStream(split, seq_len, batch_size, tok,
                          shuffle_buffer=shuffle, seed=seed, tok_batch=tok_batch)
        kw = dict(batch_size=None, num_workers=nw, pin_memory=pin and torch.cuda.is_available())
        if nw > 0:
            kw.update(prefetch_factor=prefetch_factor, persistent_workers=persistent)
        return DataLoader(ds, **kw)

    tokens_per_batch = batch_size * seq_len
    train = mk("train", shuffle_buffer, train_workers, True, True)

    val = CachedEval(mk("val", 0, eval_workers, False, False),
                     max(1, val_tokens // tokens_per_batch))

    test = mk("test", 0, eval_workers, False, False)
    if test_tokens:
        test = CachedEval(test, max(1, test_tokens // tokens_per_batch))

    return train, val, test, tok


def to_device(batch, device):
    x, y = batch
    return x.to(device, non_blocking=True).long(), y.to(device, non_blocking=True).long()


# --------------------------------------------------------------------------- #
# Utilitários para não deixar a GPU ociosa
# --------------------------------------------------------------------------- #
def find_max_batch_size(model, seq_len, vocab_size, start=16, max_bs=1024, safety=0.9):
    """Maior micro-batch que cabe na GPU (passo forward+backward+optimizer em bf16).

    Sobe em potências de 2 até dar OOM e refina por bisseção (passo de 8).
    Usa lr=0, então os pesos do modelo não são alterados.
    """
    assert torch.cuda.is_available(), "precisa de GPU"
    device = next(model.parameters()).device
    was_training = model.training
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=0.0, fused=True)

    def fits(bs):
        ok = True
        x = y = logits = loss = None
        try:
            x = torch.randint(0, vocab_size, (bs, seq_len), device=device)
            y = torch.randint(0, vocab_size, (bs, seq_len), device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            ok = False
        del x, y, logits, loss
        opt.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        return ok

    lo, hi, bs = 0, None, start
    while bs <= max_bs:
        if fits(bs):
            lo, bs = bs, bs * 2
        else:
            hi = bs
            break
    if lo == 0:
        raise RuntimeError(f"start={start} já não cabe na GPU; reduza `start`.")
    if hi is not None:
        while hi - lo > 8:
            mid = (lo + hi) // 2
            if fits(mid):
                lo = mid
            else:
                hi = mid

    del opt
    gc.collect()
    torch.cuda.empty_cache()
    model.train(was_training)
    return max(8, int(lo * safety) // 8 * 8)


def benchmark_loader(loader, n_batches=50, warmup=5):
    """Mede tokens/s que o loader entrega. Compare com os tokens/s do treino:
    se o loader entregar MENOS, a GPU fica esperando dados."""
    it = iter(loader)
    for _ in range(warmup):
        next(it)
    tokens, t0 = 0, time.perf_counter()
    for _ in range(n_batches):
        x, _ = next(it)
        tokens += x.numel()
    dt = time.perf_counter() - t0
    print(f"loader: {tokens / dt / 1e6:.2f}M tokens/s ({n_batches / dt:.1f} batches/s)")
    return tokens / dt


if __name__ == "__main__":
    train_loader, val_loader, test_loader, tok = build_loaders(
        batch_size=8, seq_len=512, train_workers=0, eval_workers=0, val_tokens=100_000)
    x, y = next(iter(train_loader))
    print(x.shape, y.shape, (x[:, 1:] == y[:, :-1]).all().item())  # [8,512] [8,512] True
    print(tok.decode(x[0][:60].long()))
    print("val batches:", sum(1 for _ in val_loader))