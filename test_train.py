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
    evaluate,
    set_seed,
    tokenize_texts,
    train_tokenizer,
    window_target_bytes,
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


def test_tokenizer_is_lossless_and_honors_vocab(tmp_path="/tmp/vowel_tok_test.json"):
    """Whitespace pre-tokenization dropped spaces while bpc still counted them."""
    corpus = ["Hello  world, it's a test.\nNew line \u00e9\u4e2d " * 3] * 50
    tok = train_tokenizer(corpus, vocab_size=300, save_path=tmp_path)
    assert tok.get_vocab_size() <= 300

    text = "Hello  world, it's a test.\nNew line \u00e9\u4e2d "
    assert tok.decode(tok.encode(text).ids) == text  # spaces and newlines survive

    ids, lens = tokenize_texts(tok, [text, "abc def"])
    assert len(ids) == len(lens)
    # token bytes cover the raw UTF-8 exactly, multi-byte characters included
    assert sum(lens) == len(text.encode("utf-8")) + len("abc def")

    # a vocab below the 256 byte alphabet must raise, never overshoot silently
    try:
        train_tokenizer(corpus, vocab_size=100, save_path=tmp_path)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for vocab below the byte alphabet")


def test_window_target_bytes_matches_dataset_targets():
    """The bits per byte denominator must cover exactly the targets the loss was taken on."""
    byte_lens = [(i % 5) + 1 for i in range(1000)]
    ds = TextDataset(list(range(1000)), seq_len=10)
    expected = sum(byte_lens[int(t)] for i in range(len(ds)) for t in ds[i][1])
    assert window_target_bytes(byte_lens, 10) == expected


def test_evaluate_bits_per_byte_is_exact():
    """bpb = summed nll / (bytes covered * ln 2), over the same windows."""
    set_seed(0)
    vocab, seq_len = 50, 8
    model = TinyGPT(vocab, 16, 2, 1, seq_len)
    tokens = [i % vocab for i in range(100)]
    byte_lens = [2] * 100
    loss, bpb, n_tokens, n_bytes = evaluate(model, tokens, byte_lens, seq_len, 4, "cpu")
    assert n_bytes == 2 * n_tokens
    assert abs(bpb - loss / (2 * math.log(2))) < 1e-9


def test_generate_handles_prompt_longer_than_context():
    """pos_emb only has seq_len rows, so a long prompt would index out of range."""
    corpus = ["the quick brown fox jumps over the lazy dog " * 40]
    tok = train_tokenizer(corpus, vocab_size=300, save_path="/tmp/vowel_gen_test.json")

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
