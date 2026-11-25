"""Module 1 (intake) - checks a prompt before any optimisation runs.

* **Empty prompts are rejected** with ``EmptyPromptError``: an empty string, only
  whitespace, an empty message list, or messages with no text at all.
* **Over-long prompts pass through unoptimised.** The limit is the target model's
  context window (``max_input_tokens`` from LiteLLM's table, via M0), or
  ``max_tokens`` if the caller sets one. A prompt with more T_M tokens than that
  is returned with ``optimise=False`` and ``reason="too_long"``; the pipeline then
  sends the original prompt unchanged (fallback guarantee, decided 2026-11-26).
  Prompts that fit are optimised normally.

The length is the T_M count of all text content (every message, every text
part), without chat-template overhead or non-text parts such as images; ``notes``
says when the length could not be checked.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .tokenizers import context_window, count_tokens, tokenizer_info

__all__ = ["EmptyPromptError", "IntakeResult", "check_intake", "prompt_texts"]


class EmptyPromptError(ValueError):
    """The prompt has no text to optimise or send."""


@dataclass(frozen=True)
class IntakeResult:
    optimise: bool                  # False -> send the prompt unchanged
    reason: str | None = None       # why not ("too_long"), None if optimise
    tokens: int | None = None       # T_M of all text content (None without a model)
    limit: int | None = None        # token limit that was applied
    limit_source: str | None = None  # "context_window" | "max_tokens" | None
    model: str | None = None
    exact: bool | None = None       # whether ``tokens`` is an exact count
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["notes"] = list(self.notes)
        return d


def prompt_texts(prompt: str | list[dict[str, Any]]) -> list[str]:
    """Every text piece of a prompt: the string itself, or each message's text
    content (string content or ``{"type": "text"}`` parts)."""
    if isinstance(prompt, str):
        return [prompt]
    if not isinstance(prompt, list):
        raise TypeError(f"prompt must be a string or a list of messages, not {type(prompt).__name__}")
    texts: list[str] = []
    for i, msg in enumerate(prompt):
        if not isinstance(msg, dict):
            raise TypeError(f"message {i} must be a dict, not {type(msg).__name__}")
        content = msg.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts += [p["text"] for p in content
                      if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)]
    return texts


def check_intake(prompt: str | list[dict[str, Any]], model: str | None = None,
                 max_tokens: int | None = None) -> IntakeResult:
    """Validate ``prompt`` for ``model``.

    Raises ``EmptyPromptError`` for an empty prompt and ``TypeError`` for a
    malformed one. Otherwise returns an ``IntakeResult``; ``optimise=False`` means
    "send it unchanged". ``max_tokens`` overrides the model's context window."""
    texts = prompt_texts(prompt)
    if not any(t.strip() for t in texts):
        raise EmptyPromptError("prompt is empty" if isinstance(prompt, str) or not prompt
                               else "no message has any text content")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    if model is None:
        if max_tokens is not None:
            raise ValueError("max_tokens needs a target model to count tokens with")
        return IntakeResult(True, notes=("no target model: length not checked",))

    tokens = sum(count_tokens(t, model) for t in texts)
    exact = tokenizer_info(model).exact
    notes: list[str] = [] if exact else [f"token count for {model} is approximate"]

    if max_tokens is not None:
        limit, source = max_tokens, "max_tokens"
    else:
        limit, source = context_window(model), "context_window"
        if limit is None:
            source = None
            notes.append(f"context window of {model} unknown: length not checked")

    if limit is not None and tokens > limit:
        return IntakeResult(False, "too_long", tokens, limit, source, model, exact, tuple(notes))
    return IntakeResult(True, None, tokens, limit, source, model, exact, tuple(notes))
