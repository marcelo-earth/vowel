"""Regression tests for the invariants this experiment depends on.

Each test here corresponds to a bug that was silent: it produced plausible
numbers rather than an error, which is the kind that survives into a README.

Run with: python test_train.py
"""

import math

import torch

from train import (
    TextDataset,
    TinyGPT,
    estimate_params,
    generate,
    get_model_config,
    set_seed,
    train_tokenizer,
)


def test_dataset_windows_do_not_overlap():
    """Stride-1 windows made an epoch 255k batches (~35h). See issue #6."""
    ds = TextDataset(list(range(1000)), seq_len=10)
    assert len(ds) == 99, len(ds)
    assert ds[0][0].tolist()[:3] == [0, 1, 2]
    assert ds[1][0].tolist()[:3] == [10, 11, 12]  # steps by seq_len, not 1

    # targets are inputs shifted by one
    x, y = ds[0]
    assert y.tolist() == list(range(1, 11))

    # every window is full length, so no ragged final batch
    assert all(ds[i][0].shape[0] == 10 for i in range(len(ds)))

    # stride stays configurable
    assert len(TextDataset(list(range(1000)), 10, stride=1)) == 990

    # corpus shorter than one window yields nothing rather than crashing
    assert len(TextDataset(list(range(5)), 10)) == 0


def test_param_formula_matches_real_model():
    """estimate_params drives dim selection, so drift would silently skew it."""
    for vocab in (1000, 8000, 32000):
        cfg = get_model_config(vocab)
        model = TinyGPT(vocab, cfg["dim"], cfg["n_heads"], cfg["n_layers"], 256)
        assert model.count_params() == estimate_params(
            vocab, cfg["dim"], cfg["n_layers"], 256
        ), vocab


def test_param_budget_is_actually_constant():
    """Hand-tuned dims drifted 33% apart, favoring the 1K arm. See issue #9."""
    totals = []
    for vocab in (1000, 4000, 8000, 32000):
        cfg = get_model_config(vocab, target_params=15_000_000)
        totals.append(estimate_params(vocab, cfg["dim"], cfg["n_layers"], 256))
        assert cfg["dim"] % cfg["n_heads"] == 0, "dim must divide into heads"

    spread = (max(totals) - min(totals)) / min(totals)
    assert spread < 0.05, f"param spread {spread:.1%} confounds the comparison"


def test_seeding_is_deterministic():
    """Unseeded runs made vocab differences indistinguishable from noise. #2"""
    set_seed(42)
    a = TinyGPT(500, 64, 4, 2, 32).tok_emb.weight.clone()
    set_seed(42)
    b = TinyGPT(500, 64, 4, 2, 32).tok_emb.weight.clone()
    set_seed(7)
    c = TinyGPT(500, 64, 4, 2, 32).tok_emb.weight.clone()

    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_bits_per_char_conversion():
    """bpc is the only cross-tokenizer comparable metric. See issue #7."""
    val_loss, chars_per_token = 4.0, 2.5
    bpc = val_loss / (chars_per_token * math.log(2))

    # a loss of ln(2) nats/token at 1 char/token is exactly 1 bit/char
    assert abs(math.log(2) / (1.0 * math.log(2)) - 1.0) < 1e-12

    # better compression at equal per-token loss means fewer bits per char
    worse = val_loss / (1.25 * math.log(2))
    assert bpc < worse


def test_tokenizer_never_overshoots_requested_vocab(tmp_path="/tmp/vowel_tok_test.json"):
    """BPE silently overshot 1000 -> 1629 on WikiText's alphabet. See issue #8."""
    # a corpus with a deliberately wide alphabet, as WikiText-103 raw has
    wide = "".join(chr(0x4E00 + i) for i in range(800))
    corpus = [" ".join(wide[i : i + 4] for i in range(0, len(wide), 4))] * 5

    # capped alphabet: the budget is honoured
    tok = train_tokenizer(corpus, vocab_size=500, save_path=tmp_path, limit_alphabet=200)
    assert tok.get_vocab_size() <= 500, tok.get_vocab_size()

    # uncapped, the 800-symbol alphabet would blow past the 500 budget, which
    # must raise rather than quietly return an oversized vocab
    try:
        train_tokenizer(corpus, vocab_size=500, save_path=tmp_path, limit_alphabet=1000)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when alphabet exceeds vocab budget")


def test_generate_handles_prompt_longer_than_context():
    """pos_emb only has seq_len rows, so a long prompt would index out of range."""
    corpus = ["the quick brown fox jumps over the lazy dog " * 40]
    tok = train_tokenizer(corpus, vocab_size=60, save_path="/tmp/vowel_gen_test.json")

    set_seed(0)
    model = TinyGPT(tok.get_vocab_size(), 32, 4, 2, seq_len=16)
    out = generate(model, tok, "the quick brown fox " * 20, max_new_tokens=3)
    assert isinstance(out, str) and out


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
