"""Train a small GPT with configurable vocab size to measure vocab vs performance."""

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import decoders, pre_tokenizers, processors
from datasets import load_dataset
from tqdm import tqdm


SAMPLE_PROMPTS = [
    "The history of",
    "In the early 20th century",
    "Scientists discovered that",
]


def set_seed(seed):
    """Seed every RNG that affects a run so vocab sizes are compared fairly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    """Tokenized text dataset for language modeling."""

    def __init__(self, tokens, seq_len, stride=None):
        # stride defaults to seq_len (non-overlapping windows). Stride 1 would
        # emit one sample per token, showing the model the same text seq_len
        # times per epoch for no extra signal.
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = stride or seq_len

    def __len__(self):
        usable = len(self.tokens) - self.seq_len - 1
        if usable < 0:
            return 0
        return usable // self.stride + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class TinyGPT(nn.Module):
    """Small decoder-only transformer."""

    def __init__(self, vocab_size, dim, n_heads, n_layers, seq_len, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # weight tying
        self.head.weight = self.tok_emb.weight

        self.seq_len = seq_len
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)

        tok = self.tok_emb(x)
        pos = self.pos_emb(pos)
        x = self.drop(tok + pos)

        # causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def count_embedding_params(self):
        return self.tok_emb.weight.numel()


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=60, temperature=0.8, top_k=40, device=None):
    """Sample a continuation of `prompt` from a trained model."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    ids = tokenizer.encode(prompt).ids
    if not ids:
        raise ValueError("Prompt tokenized to zero tokens")

    for _ in range(max_new_tokens):
        # the model only has positional embeddings up to seq_len
        context = ids[-model.seq_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1] / max(temperature, 1e-6)

        if top_k:
            k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, k).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        ids.append(int(torch.multinomial(probs, num_samples=1)))

    return tokenizer.decode(ids)


def estimate_params(vocab_size, dim, n_layers, seq_len):
    """Exact parameter count for TinyGPT, without building the model.

    Weight tying makes head.weight the same tensor as tok_emb.weight, so the
    embedding table is counted once. Per layer: attention in_proj (3*dim^2+3*dim)
    and out_proj (dim^2+dim), two FFN projections (8*dim^2+5*dim), two norms
    (4*dim) -- 12*dim^2 + 13*dim in total.
    """
    emb = vocab_size * dim + seq_len * dim
    layers = n_layers * (12 * dim * dim + 13 * dim)
    return emb + layers + 2 * dim  # + final LayerNorm


def get_model_config(vocab_size, target_params=15_000_000, seq_len=256,
                     n_heads=6, n_layers=6):
    """Pick dim so total params land on target_params for any vocab size.

    A bigger vocab spends more of the budget on the embedding table, so dim
    shrinks to compensate. Solved numerically rather than hand-tuned, because
    hand-tuned values drifted up to 33% apart, which confounds the comparison.
    """
    # dim must divide evenly into the attention heads
    best = None
    for dim in range(n_heads, 2048 + 1, n_heads):
        diff = abs(estimate_params(vocab_size, dim, n_layers, seq_len) - target_params)
        if best is None or diff < best[1]:
            best = (dim, diff)

    return {"dim": best[0], "n_heads": n_heads, "n_layers": n_layers}


def train_tokenizer(texts, vocab_size, save_path):
    """Train a lossless byte-level BPE tokenizer on the given texts.

    Byte-level pre-tokenization keeps every space and newline inside the tokens,
    so decode(encode(text)) == text and bits per character is denominated in
    the full raw text. The old Whitespace pre-tokenizer dropped spaces: the
    model never had to predict them, yet the denominator counted them.
    The initial alphabet is the 256 bytes, so any vocab_size >= 260 is honored.
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train_from_iterator(texts, trainer)

    actual = tokenizer.get_vocab_size()
    if actual > vocab_size:
        raise ValueError(
            f"Tokenizer built {actual} tokens but {vocab_size} were requested; "
            f"the 256 byte alphabet plus specials needs vocab_size >= 260."
        )
    if actual < vocab_size:
        # legitimate on a small corpus: there are not enough merges to find
        print(f"  note: corpus only supports {actual} of {vocab_size} requested tokens")

    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path} (vocab size: {actual})")
    return tokenizer


def load_wikitext(split="train", max_samples=None):
    """Load WikiText-103 dataset."""
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    texts = [t for t in ds["text"] if len(t.strip()) > 50]
    if max_samples:
        texts = texts[:max_samples]
    return texts


def tokenize_texts(tokenizer, texts):
    """Tokenize texts into a flat id list and the characters each token covers.

    The char lengths come from the encoder's offsets. They must add up to the
    raw text length, otherwise bits per character would not be over the same
    text for every tokenizer.
    """
    ids, char_lens = [], []
    for text in texts:
        enc = tokenizer.encode(text)
        lens = [end - start for start, end in enc.offsets]
        assert sum(lens) == len(text), "token offsets must cover the raw text exactly"
        ids.extend(enc.ids)
        char_lens.extend(lens)
    return ids, char_lens


def window_target_chars(char_lens, seq_len, stride=None):
    """Characters covered by the targets of every window of a TextDataset."""
    stride = stride or seq_len
    usable = len(char_lens) - seq_len - 1
    n = 0 if usable < 0 else usable // stride + 1
    csum = np.concatenate([[0], np.cumsum(char_lens)])
    starts = np.arange(n) * stride
    return int((csum[starts + seq_len + 1] - csum[starts + 1]).sum())


@torch.no_grad()
def evaluate(model, tokens, char_lens, seq_len, batch_size, device):
    """Exact validation loss and bits per character over every window.

    No drop_last: the summed negative log likelihood and the characters it
    covers come from the same windows, so the ratio is a true bits per char.
    """
    model.eval()
    ds = TextDataset(tokens, seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    total_nll, n_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_nll += nll.item()
        n_tokens += y.numel()
    chars = window_target_chars(char_lens, seq_len)
    val_loss = total_nll / max(n_tokens, 1)
    bpc = total_nll / (max(chars, 1) * math.log(2))
    return val_loss, bpc, n_tokens, chars


def train(
    vocab_size=8000,
    seq_len=256,
    batch_size=32,
    epochs=3,
    lr=3e-4,
    max_train_samples=50000,
    max_val_samples=5000,
    device=None,
    save_dir="checkpoints",
    seed=42,
):
    """Train a small GPT and return metrics."""
    set_seed(seed)

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"Training with vocab_size={vocab_size}")
    print(f"Device: {device} | seed: {seed}")
    print(f"{'='*60}")

    # load data
    print("Loading WikiText-103...")
    train_texts = load_wikitext("train", max_samples=max_train_samples)
    val_texts = load_wikitext("validation", max_samples=max_val_samples)
    print(f"Train texts: {len(train_texts)}, Val texts: {len(val_texts)}")

    # train tokenizer
    os.makedirs(save_dir, exist_ok=True)
    tok_path = os.path.join(save_dir, f"tokenizer_bl_v{vocab_size}.json")
    if os.path.exists(tok_path):
        print(f"Loading existing tokenizer from {tok_path}")
        tokenizer = Tokenizer.from_file(tok_path)
        if tokenizer.get_vocab_size() != vocab_size:
            # a stale cached tokenizer
            print(f"  cached tokenizer has {tokenizer.get_vocab_size()} tokens, "
                  f"expected {vocab_size} -- retraining")
            tokenizer = train_tokenizer(train_texts, vocab_size, tok_path)
    else:
        print(f"Training tokenizer with vocab_size={vocab_size}...")
        tokenizer = train_tokenizer(train_texts, vocab_size, tok_path)

    # tokenize
    print("Tokenizing...")
    train_tokens, _ = tokenize_texts(tokenizer, train_texts)
    val_tokens, val_char_lens = tokenize_texts(tokenizer, val_texts)
    print(f"Train tokens: {len(train_tokens):,}, Val tokens: {len(val_tokens):,}")

    # compute compression ratio
    total_chars = sum(len(t) for t in train_texts)
    compression = total_chars / len(train_tokens)
    print(f"Compression ratio: {compression:.2f} chars/token")

    # Per-token loss is in a different unit for every tokenizer, so evaluate()
    # converts it to bits per character over the same raw validation text.
    val_chars = sum(len(t) for t in val_texts)
    val_compression = val_chars / len(val_tokens)

    # datasets
    train_ds = TextDataset(train_tokens, seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # model
    config = get_model_config(vocab_size)
    actual_vocab = tokenizer.get_vocab_size()
    model = TinyGPT(
        vocab_size=actual_vocab,
        dim=config["dim"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        seq_len=seq_len,
    ).to(device)

    total_params = model.count_params()
    emb_params = model.count_embedding_params()
    emb_pct = 100 * emb_params / total_params

    print(f"\nModel config: dim={config['dim']}, heads={config['n_heads']}, layers={config['n_layers']}")
    print(f"Total params: {total_params:,}")
    print(f"Embedding params: {emb_params:,} ({emb_pct:.1f}% of total)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))

    # training loop
    metrics = {
        "vocab_size": vocab_size,
        "actual_vocab": actual_vocab,
        "seed": seed,
        "dim": config["dim"],
        "n_heads": config["n_heads"],
        "n_layers": config["n_layers"],
        "total_params": total_params,
        "embedding_params": emb_params,
        "embedding_pct": emb_pct,
        "compression_ratio": compression,
        "val_compression_ratio": val_compression,
        "val_eval_tokens": None,
        "val_eval_chars": None,
        "metric_version": "bytelevel-exact-v2",
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "train_losses": [],
        "val_losses": [],
        "val_perplexities": [],
        "val_bits_per_char": [],
        "tokens_per_sec": [],
        "epoch_seconds": [],
    }

    for epoch in range(epochs):
        # train
        model.train()
        total_loss = 0
        n_batches = 0
        start = time.time()

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, actual_vocab), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / n_batches
        elapsed = time.time() - start
        tokens_per_sec = len(train_ds) * seq_len / elapsed

        # validate on every window, with the exact characters those windows cover
        val_loss, val_bpc, val_n_tokens, val_n_chars = evaluate(
            model, val_tokens, val_char_lens, seq_len, batch_size, device
        )
        val_ppl = math.exp(min(val_loss, 20))

        metrics["train_losses"].append(train_loss)
        metrics["val_losses"].append(val_loss)
        metrics["val_perplexities"].append(val_ppl)
        metrics["val_bits_per_char"].append(val_bpc)
        metrics["val_eval_tokens"] = val_n_tokens
        metrics["val_eval_chars"] = val_n_chars
        metrics["tokens_per_sec"].append(tokens_per_sec)
        metrics["epoch_seconds"].append(elapsed)

        print(f"  Epoch {epoch+1}: train_loss={train_loss:.3f} val_loss={val_loss:.3f} "
              f"val_ppl={val_ppl:.1f} bpc={val_bpc:.3f} ({elapsed:.0f}s, {tokens_per_sec:.0f} tok/s)")

    # save checkpoint
    ckpt_path = os.path.join(save_dir, f"model_bl_v{vocab_size}_s{seed}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")

    # sample generations, so generation quality can be compared across vocab sizes
    metrics["samples"] = []
    print("\nSamples:")
    for prompt in SAMPLE_PROMPTS:
        text = generate(model, tokenizer, prompt, device=device)
        metrics["samples"].append({"prompt": prompt, "text": text})
        print(f"  [{prompt}] {text}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a small GPT with configurable vocab size")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Vocabulary size")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max-train-samples", type=int, default=50000, help="Max training texts")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda/mps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_train_samples=args.max_train_samples,
        device=args.device,
        seed=args.seed,
    )
