"""Module 0 - Tokenizer layer.

Every saving ASPO reports is counted in the *target model's* tokens, T_M(P):
the same prompt costs a different number of tokens on GPT-4o, Llama-3.1 and
Qwen2.5, and the removal loop (M5), the validator (M6) and the RL reward (M4)
all need that number. This module provides it:

* ``count_tokens(text, model)``   - T_M(text), cached (the loop calls it at every step)
* ``encode(text, model)``         - token ids (exact tokenizers only)
* ``tokenizer_info(model)``       - which tokenizer is used and whether the count is exact
* ``price_per_token(model)``      - input/output/cache-read prices from LiteLLM's cost table
* ``savings(original, compressed, model)`` - tokens and dollars saved, for reports

Tokenizer resolution, in order:

1. **tiktoken** for OpenAI models (exact). Names tiktoken does not know yet but
   that look like OpenAI models fall back to ``o200k_base`` and are flagged.
2. **Hugging Face tokenizers** for open models: Llama-3.x and Qwen2/2.5 (exact).
   Gated repos (Llama) need ``HF_TOKEN``; if loading fails the model is counted
   with the approximation below and flagged, with the error in ``note``.
3. **Approximation** for closed models without a public tokenizer (Claude,
   Gemini) and anything unknown: ``o200k_base`` count x a per-family factor.
   Factors start at 1.0 (uncalibrated) and are fitted with ``calibrate()``
   against the provider's counting endpoint (Anthropic ``messages.count_tokens``,
   Gemini ``countTokens``). Approximate counts carry ``exact=False`` so reports
   can flag them, as required by the evaluation protocol.

Custom counters (for example one that calls a provider endpoint) plug in with
``register_tokenizer``. Counts are of raw text with no special tokens (BOS) and
no chat-template overhead; that overhead is constant per message, so it cancels
in before/after comparisons.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Callable, Protocol

__all__ = [
    "TokenizerInfo", "ModelPrice", "count_tokens", "encode", "tokenizer_info",
    "price_per_token", "context_window", "prompt_cost", "savings", "calibrate",
    "register_tokenizer",
    "clear_caches",
]

APPROX_BASE = "o200k_base"
COUNT_CACHE_SIZE = 16384


class _Backend(Protocol):
    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class TokenizerInfo:
    model: str
    backend: str          # "tiktoken" | "hf" | "approx" | name of a registered backend
    name: str             # tiktoken encoding, HF repo id, or approximation family
    exact: bool           # False -> count is an approximation; flag it in reports
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelPrice:
    model: str                          # key matched in the cost table
    input_per_token: float              # USD
    output_per_token: float | None
    cache_read_per_token: float | None  # provider prompt-cache price, if any
    provider: str | None
    source: str                         # e.g. "litellm 1.111.0 (bundled table)"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- registry
#: Hugging Face repos for open-model families. Every Llama-3.x size shares one
#: tokenizer (128K vocab), as do Qwen2 and Qwen2.5 (151K vocab).
HF_REPOS: list[tuple[str, str]] = [
    (r"llama[-_ ]?3", "meta-llama/Llama-3.1-8B-Instruct"),
    (r"qwen[-_ ]?2", "Qwen/Qwen2.5-7B-Instruct"),
]

#: Families counted by approximation, with the official way to get exact counts.
APPROX_FAMILIES: list[tuple[str, str, str]] = [
    (r"claude", "claude", "no public tokenizer; exact counts via Anthropic messages.count_tokens"),
    (r"gemini|gemma", "gemini", "exact counts via Gemini countTokens"),
]

#: Multiplier on the o200k_base count, per approximation family. 1.0 = uncalibrated.
CALIBRATION: dict[str, float] = {"claude": 1.0, "gemini": 1.0, "default": 1.0}

_OPENAI_LIKE = re.compile(r"^(gpt-|o\d|chatgpt|text-embedding|davinci|babbage|ft:gpt)")

_custom: list[tuple[re.Pattern[str], str, Callable[[], _Backend], bool]] = []


def register_tokenizer(pattern: str, name: str, factory: Callable[[], _Backend],
                       exact: bool = True) -> None:
    """Use ``factory()`` (an object with ``encode(text) -> list[int]``) for models
    whose name matches ``pattern``. Registered rules take priority over built-ins."""
    _custom.insert(0, (re.compile(pattern, re.I), name, factory, exact))
    clear_caches()


def _short_name(model: str) -> str:
    """'together_ai/meta-llama/Llama-3.1-8B' -> 'llama-3.1-8b'; 'ollama/qwen2.5:7b' -> 'qwen2.5:7b'."""
    return model.strip().lower().rsplit("/", 1)[-1]


@lru_cache(maxsize=256)
def tokenizer_info(model: str) -> TokenizerInfo:
    """Decide which tokenizer counts T_M for ``model`` (cached per model name)."""
    short = _short_name(model)

    for rx, name, _factory, exact in _custom:
        if rx.search(model) or rx.search(short):
            return TokenizerInfo(model, f"custom:{name}", name, exact, "registered backend")

    tk = _tiktoken()
    if tk is not None:
        try:
            return TokenizerInfo(model, "tiktoken", tk.encoding_for_model(short).name, True)
        except KeyError:
            if _OPENAI_LIKE.match(short):
                return TokenizerInfo(model, "tiktoken", APPROX_BASE, False,
                                     "model unknown to this tiktoken version; assumed o200k_base")

    for rx, repo in HF_REPOS:
        if re.search(rx, short):
            err = _hf_load_error(repo)
            if err is None:
                return TokenizerInfo(model, "hf", repo, True)
            return TokenizerInfo(model, "approx", "default", False,
                                 f"could not load {repo} ({err}); approximated")

    for rx, family, how in APPROX_FAMILIES:
        if re.search(rx, short):
            return TokenizerInfo(model, "approx", family, False, how)
    return TokenizerInfo(model, "approx", "default", False, "unknown model; approximated")


# --------------------------------------------------------------------------- backends
def _tiktoken():
    try:
        import tiktoken
        return tiktoken
    except ImportError:
        return None


@lru_cache(maxsize=32)
def _tiktoken_encoding(name: str):
    tk = _tiktoken()
    if tk is None:
        raise ImportError("tiktoken is required: pip install 'aspo[tok]'")
    return tk.get_encoding(name)


class _HFBackend:
    def __init__(self, repo: str):
        try:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_pretrained(repo, token=os.environ.get("HF_TOKEN"))
            self._mode = "tokenizers"
        except ImportError:
            from transformers import AutoTokenizer  # heavier fallback
            self._tok = AutoTokenizer.from_pretrained(repo)
            self._mode = "transformers"

    def encode(self, text: str) -> list[int]:
        if self._mode == "tokenizers":
            return self._tok.encode(text, add_special_tokens=False).ids
        return self._tok.encode(text, add_special_tokens=False)


@lru_cache(maxsize=16)
def _hf_backend(repo: str) -> _HFBackend:
    return _HFBackend(repo)


@lru_cache(maxsize=16)
def _hf_load_error(repo: str) -> str | None:
    """None if the repo's tokenizer loads; otherwise a short reason (warned once)."""
    try:
        _hf_backend(repo)
        return None
    except Exception as e:  # gated repo, no network, missing library ...
        reason = type(e).__name__
        warnings.warn(f"ASPO: tokenizer {repo} unavailable ({reason}); token counts for this "
                      f"model are approximated. Set HF_TOKEN for gated repos.", stacklevel=3)
        return reason


@lru_cache(maxsize=64)
def _custom_backend(name: str) -> _Backend:
    for _rx, n, factory, _exact in _custom:
        if n == name:
            return factory()
    raise KeyError(name)


def _exact_encode(info: TokenizerInfo, text: str) -> list[int]:
    if info.backend == "tiktoken":
        # encode_ordinary: special-token strings like "<|endoftext|>" in user text
        # are counted as plain text, never raise.
        return _tiktoken_encoding(info.name).encode_ordinary(text)
    if info.backend == "hf":
        return _hf_backend(info.name).encode(text)
    if info.backend.startswith("custom:"):
        return list(_custom_backend(info.name).encode(text))
    raise ValueError(f"{info.model} has no exact tokenizer ({info.note})")


# --------------------------------------------------------------------------- counting
@lru_cache(maxsize=COUNT_CACHE_SIZE)
def _count_cached(model: str, text: str) -> int:
    info = tokenizer_info(model)
    if info.backend == "approx":
        return round(_approx_base(text) * CALIBRATION.get(info.name, 1.0))
    return len(_exact_encode(info, text))


def _approx_base(text: str) -> float:
    """o200k_base count; without tiktoken, the usual 4-characters-per-token rule."""
    if _tiktoken() is None:
        return len(text) / 4
    return len(_tiktoken_encoding(APPROX_BASE).encode_ordinary(text))


def count_tokens(text: str, model: str) -> int:
    """T_M(text): number of tokens ``text`` costs on ``model``.

    Results are LRU-cached on (model, text), since the removal loop re-counts the
    same candidate texts many times. Check ``tokenizer_info(model).exact``
    to know whether the number is exact or approximate."""
    if not text:
        return 0
    return _count_cached(model, text)


@lru_cache(maxsize=1024)
def _encode_cached(model: str, text: str) -> tuple[int, ...]:
    return tuple(_exact_encode(tokenizer_info(model), text))


def encode(text: str, model: str) -> tuple[int, ...]:
    """Token ids of ``text`` under ``model``'s tokenizer. Raises ``ValueError`` for
    approximated models, whose real ids are unknown."""
    return _encode_cached(model, text) if text else ()


def calibrate(family: str, texts: list[str], true_counts: list[int]) -> float:
    """Fit the approximation factor for ``family`` ("claude", "gemini", "default")
    from texts whose exact counts came from the provider's counting endpoint.
    Least squares through the origin: k = sum(b*t) / sum(b*b), b = o200k count."""
    if len(texts) != len(true_counts) or not texts:
        raise ValueError("need equally many texts and true counts (at least one)")
    base = [_approx_base(t) for t in texts]
    denom = sum(b * b for b in base)
    if denom == 0:
        raise ValueError("texts are empty")
    k = sum(b * t for b, t in zip(base, true_counts)) / denom
    CALIBRATION[family] = k
    _count_cached.cache_clear()
    return k


def clear_caches() -> None:
    """Drop every cached resolution and count (after registering or recalibrating)."""
    for fn in (tokenizer_info, _count_cached, _encode_cached, _custom_backend):
        fn.cache_clear()


# --------------------------------------------------------------------------- prices
@lru_cache(maxsize=1)
def _cost_table() -> tuple[dict[str, Any], str] | None:
    # Use the price table shipped with the installed LiteLLM instead of fetching the
    # live one from GitHub: reproducible numbers, no network call at import.
    # Set LITELLM_LOCAL_MODEL_COST_MAP=False to opt in to the live table.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    try:
        import litellm
    except ImportError:
        return None
    local = os.environ["LITELLM_LOCAL_MODEL_COST_MAP"].lower() == "true"
    version = getattr(litellm, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as _v
            version = _v("litellm")
        except Exception:
            version = "?"
    return litellm.model_cost, f"litellm {version} ({'bundled' if local else 'live'} table)"


def _table_entries(model: str):
    """(key, entry, source) for every cost-table key that may name ``model``:
    as given, lower-cased, without the provider prefix, and the bare last part."""
    table = _cost_table()
    if table is None:
        return
    costs, source = table
    stripped = model.split("/", 1)[1] if "/" in model else model
    for key in dict.fromkeys((model, model.lower(), stripped, stripped.lower(), _short_name(model))):
        entry = costs.get(key)
        if entry:
            yield key, entry, source


@lru_cache(maxsize=256)
def context_window(model: str) -> int | None:
    """Maximum input tokens ``model`` accepts (LiteLLM's ``max_input_tokens``), or
    None if unknown. Used by intake to pass over-long prompts through untouched."""
    for _key, entry, _source in _table_entries(model):
        n = entry.get("max_input_tokens")
        if isinstance(n, (int, float)) and n > 0:
            return int(n)
    return None


@lru_cache(maxsize=256)
def price_per_token(model: str) -> ModelPrice | None:
    """Prices in USD per token from LiteLLM's cost table, or None if LiteLLM is not
    installed or the model is not in the table (cost is then reported as unknown)."""
    for key, entry, source in _table_entries(model):
        if entry.get("input_cost_per_token") is not None:
            return ModelPrice(
                model=key,
                input_per_token=float(entry["input_cost_per_token"]),
                output_per_token=_opt_float(entry.get("output_cost_per_token")),
                cache_read_per_token=_opt_float(entry.get("cache_read_input_token_cost")),
                provider=entry.get("litellm_provider"),
                source=source,
            )
    if model.lower().startswith(("ollama/", "ollama_chat/")):   # local inference: no per-token price
        return ModelPrice(model, 0.0, 0.0, None, "ollama", "local model (not in table)")
    return None


def _opt_float(x: Any) -> float | None:
    return None if x is None else float(x)


def prompt_cost(text: str, model: str) -> float | None:
    """Input cost in USD of sending ``text`` to ``model`` (None if the price is unknown)."""
    price = price_per_token(model)
    return None if price is None else count_tokens(text, model) * price.input_per_token


def savings(original: str, compressed: str, model: str) -> dict[str, Any]:
    """Token and cost savings of ``compressed`` over ``original`` on ``model``:
    TR = 1 - T'/T (token reduction), CR = T/T' (compression ratio)."""
    before, after = count_tokens(original, model), count_tokens(compressed, model)
    price = price_per_token(model)
    info = tokenizer_info(model)
    return {
        "model": model,
        "tokens_before": before,
        "tokens_after": after,
        "tokens_saved": before - after,
        "token_reduction": (before - after) / before if before else 0.0,
        "compression_ratio": before / after if after else float("inf") if before else 1.0,
        "usd_saved": None if price is None else (before - after) * price.input_per_token,
        "exact": info.exact,
        "tokenizer": f"{info.backend}:{info.name}",
        "price_source": None if price is None else price.source,
    }


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(description="ASPO Module 0: count a prompt's tokens and cost per model.")
    ap.add_argument("file", nargs="?", help="prompt text file; stdin if omitted")
    ap.add_argument("-m", "--model", action="append",
                    help="target model (repeatable). Default: gpt-4o, ollama/llama3.1, ollama/qwen2.5")
    args = ap.parse_args(argv)
    text = open(args.file, encoding="utf-8").read() if args.file else sys.stdin.read()
    models = args.model or ["gpt-4o", "ollama/llama3.1", "ollama/qwen2.5"]

    rows = []
    for m in models:
        info, price = tokenizer_info(m), price_per_token(m)
        n = count_tokens(text, m)
        rows.append({
            "model": m, "tokens": n, "exact": info.exact,
            "tokenizer": f"{info.backend}:{info.name}",
            "usd_input": None if price is None else n * price.input_per_token,
            "note": info.note,
        })
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps({"chars": len(text), "words": len(text.split()), "models": rows}, indent=2))


if __name__ == "__main__":
    main()
