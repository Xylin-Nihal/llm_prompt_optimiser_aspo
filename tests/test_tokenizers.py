import warnings

import pytest

from aspo import AnalyzerConfig, analyze
from aspo import tokenizers as tok
from aspo.tokenizers import (
    calibrate, count_tokens, encode, price_per_token, prompt_cost, register_tokenizer,
    savings, tokenizer_info,
)

tiktoken = pytest.importorskip("tiktoken")

HINDI = "नमस्ते दुनिया"   # "hello world" in Hindi

# (text, cl100k_base, o200k_base, Qwen2.5) - reference counts from the official tokenizers
GOLDEN = [
    ("hello world", 2, 2, 2),
    ("The quick brown fox jumps over the lazy dog.", 10, 10, 10),
    ("def add(a, b):\n    return a + b", 11, 11, 11),
    ("Do NOT exceed 50 words. Output JSON only.", 11, 11, 12),
    ("   multiple   spaces\n\n\ttabs", 7, 7, 7),
    (HINDI, 13, 5, 13),
]


@pytest.fixture(autouse=True)
def _fresh_state():
    saved = dict(tok.CALIBRATION), list(tok._custom)
    tok.clear_caches()
    yield
    tok.CALIBRATION.clear(), tok.CALIBRATION.update(saved[0])
    tok._custom[:] = saved[1]
    tok.clear_caches()


def _qwen_available() -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tok._hf_load_error("Qwen/Qwen2.5-7B-Instruct") is None


# ---------------------------------------------------------------- resolution
@pytest.mark.parametrize("model,encoding", [
    ("gpt-4o", "o200k_base"), ("gpt-4o-mini", "o200k_base"), ("openai/gpt-4.1", "o200k_base"),
    ("gpt-4", "cl100k_base"), ("gpt-3.5-turbo", "cl100k_base"), ("azure/gpt-4", "cl100k_base"),
])
def test_openai_models_use_tiktoken(model, encoding):
    info = tokenizer_info(model)
    assert (info.backend, info.name, info.exact) == ("tiktoken", encoding, True)


def test_unknown_openai_like_model_is_flagged():
    info = tokenizer_info("gpt-99-turbo-preview")
    assert info.backend == "tiktoken" and info.name == "o200k_base"
    assert not info.exact and "assumed" in info.note


@pytest.mark.parametrize("model,family", [
    ("claude-sonnet-4-5", "claude"), ("anthropic/claude-opus-5-5", "claude"),
    ("us.anthropic.claude-opus-5-5", "claude"), ("gemini/gemini-2.5-pro", "gemini"),
    ("some-unknown-model", "default"),
])
def test_closed_models_are_approximated_and_flagged(model, family):
    info = tokenizer_info(model)
    assert (info.backend, info.name, info.exact) == ("approx", family, False)
    assert info.note


@pytest.mark.parametrize("model", ["ollama/qwen2.5", "Qwen/Qwen2.5-7B-Instruct", "qwen2.5:7b"])
def test_qwen_resolves_to_hf_repo(model):
    if not _qwen_available():
        pytest.skip("Qwen2.5 tokenizer not downloadable here")
    info = tokenizer_info(model)
    assert (info.backend, info.name, info.exact) == ("hf", "Qwen/Qwen2.5-7B-Instruct", True)


def test_llama_is_exact_or_flagged():
    """Llama-3.1 is gated: exact with HF_TOKEN, otherwise approximated and flagged."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = tokenizer_info("ollama/llama3.1")
    if info.exact:
        assert (info.backend, info.name) == ("hf", "meta-llama/Llama-3.1-8B-Instruct")
    else:
        assert info.backend == "approx" and "Llama-3.1" in info.note


# ---------------------------------------------------------------- counts match official tokenizers
@pytest.mark.parametrize("text,cl100k,o200k,_qwen", GOLDEN)
def test_tiktoken_counts_match_golden(text, cl100k, o200k, _qwen):
    assert count_tokens(text, "gpt-4") == cl100k
    assert count_tokens(text, "gpt-4o") == o200k


@pytest.mark.parametrize("text,_cl,_o2,qwen", GOLDEN)
def test_qwen_counts_match_golden(text, _cl, _o2, qwen):
    if not _qwen_available():
        pytest.skip("Qwen2.5 tokenizer not downloadable here")
    assert count_tokens(text, "ollama/qwen2.5") == qwen


def test_counts_match_tiktoken_directly():
    enc = tiktoken.get_encoding("o200k_base")
    for text, *_ in GOLDEN:
        assert count_tokens(text, "gpt-4o") == len(enc.encode(text))


def test_encode_ids():
    assert encode("hello world", "gpt-4o") == (24912, 2375)
    assert encode("hello world", "gpt-4") == (15339, 1917)
    assert encode("", "gpt-4o") == ()
    with pytest.raises(ValueError):
        encode("hello", "claude-sonnet-4-5")


def test_special_token_text_is_counted_not_rejected():
    # User text containing "<|endoftext|>" must be counted as plain text, never raise.
    assert count_tokens("a <|endoftext|> b", "gpt-4o") > 3


def test_empty_text_is_zero():
    assert count_tokens("", "gpt-4o") == 0
    assert count_tokens("", "claude-sonnet-4-5") == 0


def test_tokenizers_differ_across_models():
    """The 'token tax': the same text costs different T_M on different models."""
    assert count_tokens(HINDI, "gpt-4") > 2 * count_tokens(HINDI, "gpt-4o")


# ---------------------------------------------------------------- cache
def test_counts_are_cached():
    tok._count_cached.cache_clear()
    text = "cache me " * 50
    count_tokens(text, "gpt-4o")
    count_tokens(text, "gpt-4o")
    info = tok._count_cached.cache_info()
    assert info.hits == 1 and info.misses == 1


# ---------------------------------------------------------------- approximation + calibration
def test_approx_uses_o200k_times_factor():
    text = "The quick brown fox jumps over the lazy dog."
    assert count_tokens(text, "claude-sonnet-4-5") == count_tokens(text, "gpt-4o")
    tok.CALIBRATION["claude"] = 1.5
    tok.clear_caches()
    assert count_tokens(text, "claude-sonnet-4-5") == 15


def test_calibrate_fits_least_squares_factor():
    texts = ["hello world", "The quick brown fox jumps over the lazy dog."]   # o200k: 2, 10
    k = calibrate("claude", texts, [3, 12])
    assert k == pytest.approx((2 * 3 + 10 * 12) / (2 * 2 + 10 * 10))
    assert count_tokens(texts[1], "claude-sonnet-4-5") == round(10 * k)
    with pytest.raises(ValueError):
        calibrate("claude", ["a"], [])


# ---------------------------------------------------------------- custom backend
def test_register_custom_tokenizer():
    class CharTokenizer:
        def encode(self, text):
            return [ord(c) for c in text]

    register_tokenizer(r"^my-model", "chars", CharTokenizer)
    info = tokenizer_info("my-model-v1")
    assert info.backend == "custom:chars" and info.exact
    assert count_tokens("abcde", "my-model-v1") == 5
    assert encode("ab", "my-model-v1") == (97, 98)


# ---------------------------------------------------------------- prices
litellm_missing = pytest.mark.skipif(tok._cost_table() is None, reason="litellm not installed")


@litellm_missing
def test_price_lookup_with_and_without_provider_prefix():
    p = price_per_token("gpt-4o")
    assert p is not None and p.input_per_token > 0 and p.provider == "openai"
    assert p.source.startswith("litellm") and "bundled" in p.source
    assert price_per_token("openai/gpt-4o").input_per_token == p.input_per_token
    assert p.output_per_token >= p.input_per_token


@litellm_missing
def test_local_model_is_free_and_unknown_is_none():
    assert price_per_token("ollama/llama3.1").input_per_token == 0.0
    assert price_per_token("ollama/qwen2.5").input_per_token == 0.0   # not in table, still local
    assert price_per_token("definitely-not-a-model-xyz") is None


@litellm_missing
def test_prompt_cost_and_savings():
    p = price_per_token("gpt-4o")
    text = "The quick brown fox jumps over the lazy dog."
    assert prompt_cost(text, "gpt-4o") == pytest.approx(10 * p.input_per_token)

    s = savings(text, "quick brown fox jumps lazy dog.", "gpt-4o")
    assert s["tokens_before"] == 10 and s["tokens_after"] == s["tokens_before"] - s["tokens_saved"]
    assert 0 < s["token_reduction"] < 1
    assert s["compression_ratio"] == pytest.approx(s["tokens_before"] / s["tokens_after"])
    assert s["usd_saved"] == pytest.approx(s["tokens_saved"] * p.input_per_token)
    assert s["exact"] and s["tokenizer"] == "tiktoken:o200k_base"


def test_savings_edge_cases():
    s = savings("hello world", "", "gpt-4o")
    assert s["tokens_after"] == 0 and s["compression_ratio"] == float("inf")
    assert savings("", "", "gpt-4o")["compression_ratio"] == 1.0


# ---------------------------------------------------------------- analyzer knock-on
def test_analyzer_length_in_target_model_tokens():
    prompt = "Summarize the following article in 3 bullet points.\n\n" + "The council met. " * 200
    words = analyze(prompt)
    tokens = analyze(prompt, AnalyzerConfig(target_model="gpt-4o"))
    assert "total_tokens" not in words.features
    assert tokens.features["total_tokens"] == count_tokens(prompt, "gpt-4o")
    assert tokens.task == words.task
    assert 0 < tokens.complexity_factors["length"] <= 1


def test_cli_runs(capsys):
    import json
    import sys
    from io import StringIO

    old = sys.stdin
    sys.stdin = StringIO("hello world")
    try:
        tok.main(["-m", "gpt-4o", "-m", "claude-sonnet-4-5"])
    finally:
        sys.stdin = old
    out = json.loads(capsys.readouterr().out)
    assert [r["tokens"] for r in out["models"]] == [2, 2]
    assert [r["exact"] for r in out["models"]] == [True, False]
