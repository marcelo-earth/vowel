"""Run the full vocab-size sweep and write results.json.

This is the experiment behind the README's numbers. Every model gets the same
seed, the same corpus and the same parameter budget; only vocab size changes.
"""

import argparse
import json

from train import train

VOCAB_SIZES = [1000, 4000, 8000, 32000]


def main():
    parser = argparse.ArgumentParser(description="Run the vocab size sweep")
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=VOCAB_SIZES)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--max-val-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results.json")
    args = parser.parse_args()

    all_metrics = []
    for vs in args.vocab_sizes:
        metrics = train(
            vocab_size=vs,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            seed=args.seed,
        )
        all_metrics.append(metrics)

        # write after each model so a crash late in the sweep keeps earlier runs
        with open(args.out, "w") as f:
            json.dump(all_metrics, f, indent=2)

    print(f"\n{'='*60}\nSweep complete. Results in {args.out}\n{'='*60}")
    print(f"{'vocab':>7} {'dim':>5} {'params':>12} {'emb%':>6} {'ppl':>9} {'bpc':>7}")
    for m in all_metrics:
        print(f"{m['vocab_size']:>7} {m['dim']:>5} {m['total_params']:>12,} "
              f"{m['embedding_pct']:>5.1f}% {m['val_perplexities'][-1]:>9.1f} "
              f"{m['val_bits_per_char'][-1]:>7.3f}")


if __name__ == "__main__":
    main()
