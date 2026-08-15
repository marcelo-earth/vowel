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

The sweep is being re-run. Earlier versions of this README listed results that
were never actually produced by a training run, so they have been removed rather
than left to look measured. Real numbers go here once `run_sweep.py` finishes.

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
