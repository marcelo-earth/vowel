# Vowel

How does vocabulary size affect a small language model? Train tiny transformers with vocab sizes from 1K to 32K and find out.

## What is this?

Bigger vocab means each token carries more information, but the embedding table gets huge. For small models this is a real problem -- a 32K vocab with dim=256 means 8M parameters just in embeddings. This project measures the tradeoff.

## What we do

1. Train a small GPT-style transformer (~10-25M params) on WikiText-103
2. Vary vocab size: 1K, 4K, 8K, 32K
3. Scale model dimensions so total param count stays roughly constant
4. Measure perplexity, generation quality, and training speed
5. Plot the vocab size vs performance curve

## Key findings

- 8K vocab is the sweet spot for models under 25M params
- At 1K vocab, sequences get too long and training is slow
- At 32K vocab, embeddings eat most of the parameter budget
- Perplexity drops fast from 1K to 8K, then flattens

## Setup

```bash
pip install -r requirements.txt
python train.py --vocab-size 8000
```

## Quick results

| Vocab | Dim | Total params | Embedding % | Notes |
|-------|-----|-------------|-------------|-------|
| 1K | 384 | ~11M | ~9% | Long sequences, slow training |
| 4K | 320 | ~13M | ~19% | Good balance |
| 8K | 288 | ~14M | ~29% | Sweet spot for this model size |
| 32K | 192 | ~14M | ~53% | Embeddings dominate |

## Files

| File | What it does |
|------|-------------|
| `train.py` | Train a small transformer with configurable vocab size |
| `vowel.ipynb` | Full analysis with plots |
