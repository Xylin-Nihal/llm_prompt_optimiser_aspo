import pytest

from aspo import EmptyPromptError, check_intake, count_tokens
from aspo import tokenizers as tok
from aspo.intake import prompt_texts

pytest.importorskip("tiktoken")
litellm_missing = pytest.mark.skipif(tok._cost_table() is None, reason="litellm not installed")


# ---------------------------------------------------------------- empty prompts
@pytest.mark.parametrize("prompt", [
    "", "   ", "\n\t \n", [],
    [{"role": "user", "content": ""}],
    [{"role": "system", "content": "  "}, {"role": "user", "content": "\n"}],
    [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
    [{"role": "user", "content": None}],
])
def test_empty_prompts_are_rejected(prompt):
    with pytest.raises(EmptyPromptError):
        check_intake(prompt, "gpt-4o")


def test_empty_prompt_error_is_a_value_error():
    with pytest.raises(ValueError):
        check_intake("", "gpt-4o")


@pytest.mark.parametrize("prompt", [None, 42, {"role": "user"}, ["just a string"]])
def test_malformed_prompts_raise_type_error(prompt):
    with pytest.raises(TypeError):
        check_intake(prompt, "gpt-4o")


def test_text_parts_are_collected():
    msgs = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": [{"type": "text", "text": "Describe"},
                                     {"type": "image_url", "image_url": {"url": "x"}},
                                     {"type": "text", "text": "this image."}]},
        {"role": "assistant", "content": None, "tool_calls": []},
    ]
    assert prompt_texts(msgs) == ["Be brief.", "Describe", "this image."]


# ---------------------------------------------------------------- length limit
@litellm_missing
def test_prompt_within_context_window_is_optimised():
    r = check_intake("What is the capital of France?", "gpt-4o")
    assert r.optimise and r.reason is None
    assert r.tokens == count_tokens("What is the capital of France?", "gpt-4o")
    assert r.limit == tok.context_window("gpt-4o") and r.limit_source == "context_window"
    assert r.exact


@litellm_missing
def test_prompt_over_context_window_passes_through():
    window = tok.context_window("ollama/llama3.1")
    assert window == 8192
    long_prompt = "word " * (window + 100)
    r = check_intake(long_prompt, "ollama/llama3.1")
    assert not r.optimise and r.reason == "too_long"
    assert r.tokens > r.limit == window


def test_max_tokens_overrides_context_window():
    text = "The quick brown fox jumps over the lazy dog."   # 10 o200k tokens
    assert check_intake(text, "gpt-4o", max_tokens=10).optimise          # equal fits
    r = check_intake(text, "gpt-4o", max_tokens=9)
    assert not r.optimise and r.reason == "too_long"
    assert (r.tokens, r.limit, r.limit_source) == (10, 9, "max_tokens")


def test_chat_length_sums_all_messages():
    msgs = [{"role": "system", "content": "hello world"},
            {"role": "user", "content": "hello world"}]
    assert check_intake(msgs, "gpt-4o").tokens == 4
    assert not check_intake(msgs, "gpt-4o", max_tokens=3).optimise


def test_unknown_context_window_is_not_checked():
    r = check_intake("hello world", "some-unknown-model-xyz")
    assert r.optimise and r.limit is None
    assert any("context window" in n for n in r.notes)
    assert any("approximate" in n for n in r.notes)


def test_no_model_skips_length_check():
    r = check_intake("hello world")
    assert r.optimise and r.tokens is None and r.notes
    with pytest.raises(ValueError):
        check_intake("hello world", max_tokens=100)


def test_bad_max_tokens():
    with pytest.raises(ValueError):
        check_intake("hello", "gpt-4o", max_tokens=0)


def test_to_dict():
    d = check_intake("hello world", "gpt-4o", max_tokens=1).to_dict()
    assert d["optimise"] is False and d["reason"] == "too_long" and isinstance(d["notes"], list)
