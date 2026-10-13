# Vowel

How does vocabulary size affect a small language model? Train tiny transformers with vocab sizes from 1K to 32K and find out.

## What is this?

Bigger vocab means each token carries more information, but the embedding table gets huge. For small models this is a real problem -- a 32K vocab with dim=256 means 8M parameters just in embeddings. This project measures the tradeoff.

## What we do

1. Train a small GPT-style transformer (~10-25M params) on WikiText-103
2. Vary vocab size: 1K, 4K, 8K, 32K
3. Scale model dimensions so total param count stays roughly constant
4. Measure bits per character, generation quality, and training speed
5. Plot the vocab size vs performance curve

## How we measure

Perplexity is per *token*, and each run here uses a different tokenizer, so the
perplexities are not comparable to each other. A 1K-vocab model predicts from a
smaller candidate set and covers less text per token, so it posts a lower
per-token perplexity without being a better model.

The headline metric is therefore **bits per character**, which divides the same
loss by the raw characters of the validation text. Every run is scored against
an identical denominator. Perplexity is still recorded, but not ranked on.

## Status

First sweep complete: 4 vocab sizes, 1 seed (42), 3 epochs, about 15M params each
(`results.json`). Earlier versions of this README listed results that were never
produced by a training run; these numbers come from `run_sweep.py`.

| Vocab | dim | Params | Embedding share | Chars/token | Val bits/char (epoch 3) |
|------:|----:|-------:|----------------:|------------:|------------------------:|
| 1,000 | 450 | 15.18M | 3.0% | 2.73 | 1.799 |
| 4,000 | 426 | 14.91M | 11.4% | 3.71 | 1.850 |
| 8,000 | 402 | 14.99M | 21.5% | 4.19 | 1.806 |
| 32,000 | 282 | 14.84M | 60.8% | 4.87 | 1.753 |

One seed is exploratory: the 1K to 8K differences (0.05 bits/char) cannot be told
apart from seed noise yet. Known limits before this is a result:

- The `Whitespace` pre-tokenizer drops spaces, so the model never predicts them
  but the bits/char denominator counts them. Fix before comparing tokenizers.
- Validation loss uses `drop_last` windows while the denominator counts all
  validation characters.
- Needs 5 sizes by 3 seeds with a pre-registered rule for seed variance.

## Setup

```bash
pip install -r requirements.txt
python train.py --vocab-size 8000   # a single model
python run_sweep.py                 # all four, writes results.json
```

## Files

| File | What it does |
|------|-------------|
| `train.py` | Train a small transformer with configurable vocab size |
| `run_sweep.py` | Run all four vocab sizes, write `results.json` |
| `vowel.ipynb` | Full analysis with plots |
