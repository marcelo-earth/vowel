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
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset
from tqdm import tqdm


def set_seed(seed):
    """Seed every RNG that affects a run so vocab sizes are compared fairly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    """Tokenized text dataset for language modeling."""

    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
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


def get_model_config(vocab_size, target_params=15_000_000):
    """Scale model dims to keep total params roughly constant across vocab sizes.

    With weight tying, embedding params = vocab_size * dim.
    We want total params ~ target_params regardless of vocab_size.
    """
    # start with a reasonable dim and adjust
    # rough formula: total ~ vocab*dim + n_layers*(12*dim^2) + vocab*dim (tied)
    # simplified: total ~ 2*vocab*dim + n_layers*12*dim^2

    # hand-tuned so total params land around 15M for each vocab size
    # bigger vocab -> smaller dim to compensate
    configs = {
        1000: {"dim": 384, "n_heads": 6, "n_layers": 6},   # ~11M params
        4000: {"dim": 320, "n_heads": 8, "n_layers": 6},    # ~13M params
        8000: {"dim": 288, "n_heads": 6, "n_layers": 6},    # ~14M params
        32000: {"dim": 192, "n_heads": 6, "n_layers": 6},   # ~14M params
    }

    if vocab_size in configs:
        return configs[vocab_size]

    # fallback: pick dim=256 and 6 layers
    return {"dim": 256, "n_heads": 8, "n_layers": 6}


def train_tokenizer(texts, vocab_size, save_path):
    """Train a BPE tokenizer on the given texts."""
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        min_frequency=2,
    )

    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path} (vocab size: {tokenizer.get_vocab_size()})")
    return tokenizer


def load_wikitext(split="train", max_samples=None):
    """Load WikiText-103 dataset."""
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    texts = [t for t in ds["text"] if len(t.strip()) > 50]
    if max_samples:
        texts = texts[:max_samples]
    return texts


def tokenize_texts(tokenizer, texts):
    """Tokenize a list of texts and return flat token list."""
    all_tokens = []
    for text in texts:
        encoded = tokenizer.encode(text)
        all_tokens.extend(encoded.ids)
    return all_tokens


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
    tok_path = os.path.join(save_dir, f"tokenizer_v{vocab_size}.json")
    if os.path.exists(tok_path):
        print(f"Loading existing tokenizer from {tok_path}")
        tokenizer = Tokenizer.from_file(tok_path)
    else:
        print(f"Training tokenizer with vocab_size={vocab_size}...")
        tokenizer = train_tokenizer(train_texts, vocab_size, tok_path)

    # tokenize
    print("Tokenizing...")
    train_tokens = tokenize_texts(tokenizer, train_texts)
    val_tokens = tokenize_texts(tokenizer, val_texts)
    print(f"Train tokens: {len(train_tokens):,}, Val tokens: {len(val_tokens):,}")

    # compute compression ratio
    total_chars = sum(len(t) for t in train_texts)
    compression = total_chars / len(train_tokens)
    print(f"Compression ratio: {compression:.2f} chars/token")

    # datasets
    train_ds = TextDataset(train_tokens, seq_len)
    val_ds = TextDataset(val_tokens, seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

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
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "train_losses": [],
        "val_losses": [],
        "val_perplexities": [],
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

        # validate
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, actual_vocab), y.view(-1))
                val_loss += loss.item()
                val_batches += 1

        val_loss = val_loss / max(val_batches, 1)
        val_ppl = math.exp(min(val_loss, 20))

        metrics["train_losses"].append(train_loss)
        metrics["val_losses"].append(val_loss)
        metrics["val_perplexities"].append(val_ppl)

        print(f"  Epoch {epoch+1}: train_loss={train_loss:.3f} val_loss={val_loss:.3f} "
              f"val_ppl={val_ppl:.1f} ({elapsed:.0f}s, {tokens_per_sec:.0f} tok/s)")

    # save checkpoint
    ckpt_path = os.path.join(save_dir, f"model_v{vocab_size}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")

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
