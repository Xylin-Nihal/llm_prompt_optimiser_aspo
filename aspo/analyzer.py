"""Module 2 - Intent & Task Analyzer.

Given a prompt (string, chat messages or an already ``SegmentedPrompt``) it
returns:

* ``task``        - one of ``TaskType`` (qa, rag_qa, summarization, code, ...)
* ``scores``      - a probability for every task type
* ``complexity``  - a 0..1 score of how demanding the request is, with factors
* ``features`` / ``signals`` - the evidence, for explainability and for training

Why it matters for ASPO: the strategy selector (module 4) uses the task to
decide *what* may be compressed (RAG: compress documents hard; transformation:
the input text must survive almost verbatim) and the complexity to decide *how
hard* (complex reasoning prompts get gentler budgets).

v1 is a transparent linear model: hand-weighted cue features -> logits ->
softmax. It has the same shape as logistic regression, so a learned model
(embeddings + LR, then DeBERTa) can replace the weights later and reuse
``features`` unchanged. v1 also serves as the "rule-based analyzer" ablation.

Cues are weighted by the *role* of the segment they occur in: "summarize" in an
instruction is strong evidence, the same word inside a retrieved document is
almost none. That is where the segmenter pays off.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .lexicon import MULTISTEP_CUES, TASK_CUES
from .segmenter import segment
from .tokenizers import count_tokens
from .types import IntentAnalysis, SegmentedPrompt, SegmentRole, SpanKind, TaskType, merge_intervals

R = SegmentRole
T = TaskType

#: How much a cue found in a segment of this role counts as evidence of the task.
ROLE_WEIGHTS: dict[SegmentRole, float] = {
    R.INSTRUCTION: 1.0,
    R.QUESTION: 1.0,
    R.OTHER: 0.8,
    R.FORMAT: 0.6,
    R.SYSTEM: 0.5,
    R.EXAMPLE: 0.3,
    R.HISTORY: 0.2,
    R.CONTEXT: 0.15,
    R.DATA: 0.1,
    R.TOOL_OUTPUT: 0.1,
    R.CODE: 0.0,          # code is handled as a structural feature instead
}

#: Prior logit per task (QA is the fallback when no cue fires).
BIASES: dict[TaskType, float] = {t: 0.0 for t in TaskType} | {T.QA: 0.5, T.CHITCHAT: -0.5}

#: Inherent difficulty of each task type, blended by task probability.
TASK_DIFFICULTY: dict[TaskType, float] = {
    T.REASONING: 0.75, T.CODE: 0.65, T.RAG_QA: 0.5, T.EXTRACTION: 0.45,
    T.SUMMARIZATION: 0.4, T.TRANSFORMATION: 0.4, T.CREATIVE: 0.35,
    T.CLASSIFICATION: 0.3, T.QA: 0.3, T.CHITCHAT: 0.05,
}

COMPLEXITY_WEIGHTS: dict[str, float] = {
    "task": 0.25, "reasoning": 0.20, "constraints": 0.15,
    "length": 0.15, "technical": 0.15, "structure": 0.10,
}

_RESIDUAL = (T.QA, T.RAG_QA, T.CHITCHAT)
SPECIFIC_EVIDENCE = 1.5   # a specific task with at least this much evidence ...
QA_CAP = 0.75             # ... caps how much QA may gain from question form alone

_CONTEXT_ROLES = (R.CONTEXT, R.DATA, R.TOOL_OUTPUT)
_CONSTRAINT_KINDS = (SpanKind.NEGATION, SpanKind.CONSTRAINT, SpanKind.FORMAT)
_TECH_KINDS = frozenset(SpanKind) - set(_CONSTRAINT_KINDS) - {SpanKind.QUOTED}
_MATH_EXPR = re.compile(r"\d\s*[+*/×÷^=<>]\s*\(?\d|\d\s+-\s+\d|\b\d+\s*%\s*of\b")
_LABELLED_OUTPUT = re.compile(r"(?im)^\s*(?:sentiment|label|category|class)\s*:")
_LIST_LINE = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+")


@dataclass
class AnalyzerConfig:
    context_min_words: int = 40          # supplied context this big turns QA into RAG-QA
    short_prompt_words: int = 12         # below this, a prompt with no task is chit-chat
    long_prompt_words: float = 4000.0    # length factor saturates here (no target model)
    target_model: str | None = None      # if set, length is measured in T_M tokens (M0)
    long_prompt_tokens: float = 5000.0   # length factor saturates here (with a target model)
    temperature: float = 1.0
    role_weights: dict[SegmentRole, float] = field(default_factory=lambda: dict(ROLE_WEIGHTS))


class IntentAnalyzer:
    def __init__(self, config: AnalyzerConfig | None = None):
        self.config = config or AnalyzerConfig()
        self._cues = {
            TaskType(task): [(re.compile(p, re.I | re.M), w) for p, w in cues]
            for task, cues in TASK_CUES.items()
        }
        self._multistep = [re.compile(p, re.I) for p in MULTISTEP_CUES]

    # ---------------------------------------------------------------- public API
    def analyze(self, prompt: str | list[dict[str, Any]] | SegmentedPrompt) -> IntentAnalysis:
        sp = prompt if isinstance(prompt, SegmentedPrompt) else segment(prompt)
        cfg = self.config
        signals: list[str] = []

        # The "task text" is everything except documents, data, history and code.
        task_segs = [s for s in sp.segments if s.role not in (*_CONTEXT_ROLES, R.HISTORY, R.CODE)]
        if not task_segs:   # e.g. the whole prompt is one paragraph classified as context
            task_segs = [s for s in sp.segments if s.role is not R.CODE]
        task_text = "\n".join(s.body for s in task_segs)

        total_words = sum(s.word_count for s in sp.segments)
        context_words = sum(len(s.body.split()) for s in sp.segments if s.role in _CONTEXT_ROLES)
        has_context = context_words >= cfg.context_min_words
        has_code = any(s.role is R.CODE for s in sp.segments) or any(
            p.kind is SpanKind.CODE for s in task_segs for p in s.protected
        )
        has_question_seg = any(s.role is R.QUESTION for s in sp.segments)
        has_task_seg = any(s.role in (R.INSTRUCTION, R.QUESTION) for s in sp.segments)
        n_questions = task_text.count("?")

        # ---- 1. lexical cues, weighted by segment role
        logits = dict(BIASES)
        for task, cues in self._cues.items():
            for rx, weight in cues:
                best, where = 0.0, None
                for s in sp.segments:
                    rw = cfg.role_weights.get(s.role, 0.5)
                    if rw > best and rx.search(s.body):
                        best, where = rw, s
                if best:
                    logits[task] += weight * best
                    if weight * best >= 0.3:
                        m = rx.search(where.body)
                        signals.append(f"{task.value}: '{m.group(0).strip()}' in {where.role.value} "
                                       f"(+{weight * best:.2f})")
        cue_logits = {t: logits[t] - BIASES[t] for t in TaskType}

        # ---- 2. structural features
        def bump(task: TaskType, amount: float, why: str) -> None:
            logits[task] += amount
            signals.append(f"{task.value}: {why} ({amount:+.2f})")

        question_signal = cue_logits[T.QA] + (1.0 if has_question_seg else 0.0)
        if has_question_seg:
            bump(T.QA, 1.0, "question segment present")
        if has_context and question_signal > 0:
            bump(T.RAG_QA, 1.0 + question_signal, f"question over {context_words} words of supplied context")
            bump(T.QA, -1.5, "supplied context makes this RAG rather than closed-book")
        elif not has_context:
            logits[T.RAG_QA] -= 3.0
        else:   # context but no question: summarise/extract/rewrite it, not RAG-QA
            logits[T.RAG_QA] -= 2.0
        if has_code:
            bump(T.CODE, 3.0, "code block / inline code present")
            if question_signal > 0:
                bump(T.QA, -1.5, "question is about the supplied code")
        if any(_LABELLED_OUTPUT.search(s.text) for s in sp.by_role(R.EXAMPLE)):
            bump(T.CLASSIFICATION, 1.5, "few-shot examples with label outputs")
        if _MATH_EXPR.search(task_text):
            bump(T.REASONING, 2.0, "arithmetic expression in the task")
        if total_words <= cfg.short_prompt_words and not has_code and not has_context:
            bump(T.CHITCHAT, 1.0 if has_task_seg else 1.5, f"very short prompt ({total_words} words)")
        elif total_words > 3 * cfg.short_prompt_words:
            logits[T.CHITCHAT] -= 2.0

        # QA is the residual class: "What is 17*23?" or "Is this spam?" are *phrased*
        # as questions, but a specific task explains them better.
        specific = max(logits[t] - BIASES[t] for t in TaskType if t not in _RESIDUAL)
        qa_gain = logits[T.QA] - BIASES[T.QA]
        if specific >= SPECIFIC_EVIDENCE and qa_gain > QA_CAP:
            bump(T.QA, QA_CAP - qa_gain, "question form explained by a more specific task")

        scores = _softmax(logits, cfg.temperature)
        task = max(scores, key=scores.get)

        # ---- 3. complexity
        multistep_hits = sum(len(rx.findall(task_text)) for rx in self._multistep)
        list_items = len(_LIST_LINE.findall(task_text))
        subtasks = max(1, n_questions + list_items +
                       sum(1 for s in sp.segments if s.role is R.INSTRUCTION))
        n_constraints = sum(1 for s in task_segs for p in s.protected if p.kind in _CONSTRAINT_KINDS)
        total_chars = sum(len(s.text) for s in sp.segments) or 1
        tech_chars = sum(
            e - st
            for s in sp.segments
            for st, e in merge_intervals([(p.start, p.end) for p in s.protected if p.kind in _TECH_KINDS])
        )
        if any(s.role is R.CODE for s in sp.segments):
            tech_chars += sum(len(s.text) for s in sp.by_role(R.CODE))
        roles_used = {s.role for s in sp.segments}
        if cfg.target_model:   # T_M(P): length as the target model sees it
            total_tokens = count_tokens(sp.to_text(), cfg.target_model)
            length = math.log1p(total_tokens) / math.log1p(cfg.long_prompt_tokens)
        else:                  # no model known: fall back to words
            total_tokens = None
            length = math.log1p(total_words) / math.log1p(cfg.long_prompt_words)

        factors = {
            "task": sum(scores[t] * TASK_DIFFICULTY[t] for t in TaskType),
            "reasoning": min(1.0, 0.25 * multistep_hits + 0.15 * (subtasks - 1)),
            "constraints": min(1.0, n_constraints / 6),
            "length": min(1.0, length),
            "technical": min(1.0, 2.0 * tech_chars / total_chars),
            "structure": min(1.0, (len(roles_used) - 1) / 5),
        }
        complexity = sum(COMPLEXITY_WEIGHTS[k] * v for k, v in factors.items())

        features: dict[str, float] = {
            "total_words": total_words,
            "context_words": context_words,
            "n_segments": len(sp.segments),
            "n_questions": n_questions,
            "n_subtasks": subtasks,
            "n_constraints": n_constraints,
            "multistep_hits": multistep_hits,
            "has_context": float(has_context),
            "has_code": float(has_code),
            "has_question_segment": float(has_question_seg),
            "is_chat": float(sp.is_chat),
            **({"total_tokens": total_tokens} if total_tokens is not None else {}),
            **{f"role_{r.value}": float(r in roles_used) for r in SegmentRole},
            **{f"cue_{t.value}": cue_logits[t] for t in TaskType},
        }
        return IntentAnalysis(
            task=task,
            confidence=scores[task],
            scores=scores,
            complexity=min(1.0, max(0.0, complexity)),
            complexity_factors=factors,
            features=features,
            signals=signals,
        )


def _softmax(logits: dict[TaskType, float], temperature: float) -> dict[TaskType, float]:
    m = max(logits.values())
    exps = {t: math.exp((v - m) / temperature) for t, v in logits.items()}
    z = sum(exps.values())
    return {t: e / z for t, e in exps.items()}



_DEFAULT = IntentAnalyzer()


def analyze(prompt: str | list[dict[str, Any]] | SegmentedPrompt,
            config: AnalyzerConfig | None = None) -> IntentAnalysis:
    """Classify the task type and complexity of a prompt."""
    return (IntentAnalyzer(config) if config else _DEFAULT).analyze(prompt)


def _describe(a: IntentAnalysis) -> str:
    lines = [f"task:        {a.task.value}  (confidence {a.confidence:.0%})",
             "scores:      " + ", ".join(f"{t.value} {p:.0%}" for t, p in a.top(4)),
             f"complexity:  {a.complexity:.2f} ({a.complexity_level})",
             "factors:     " + ", ".join(f"{k} {v:.2f}" for k, v in a.complexity_factors.items()),
             "evidence:"]
    lines += [f"  - {s}" for s in a.signals] or ["  - (no cues fired; default prior)"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="ASPO Module 2: analyze a prompt's task and complexity.")
    ap.add_argument("file", nargs="?", help="prompt file (.txt, or .json chat messages); stdin if omitted")
    ap.add_argument("--json", action="store_true", help="print full JSON output")
    ap.add_argument("-m", "--model", help="target model; measures length in its tokens (M0)")
    args = ap.parse_args(argv)

    raw = open(args.file, encoding="utf-8").read() if args.file else sys.stdin.read()
    prompt: Any = json.loads(raw) if args.file and args.file.endswith(".json") else raw
    result = analyze(prompt, AnalyzerConfig(target_model=args.model) if args.model else None)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) if args.json else _describe(result))


if __name__ == "__main__":
    main()
