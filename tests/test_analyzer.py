import json
from pathlib import Path

import pytest

from aspo import AnalyzerConfig, TaskType as T, analyze, segment

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
ARTICLE = "The city council met on Tuesday to discuss the new riverside park and its budget. " * 6

CASES = [
    ("What is the capital of Australia?", T.QA),
    ("Who wrote Pride and Prejudice?", T.QA),
    ((EXAMPLES / "rag_prompt.txt").read_text(encoding="utf-8"), T.RAG_QA),
    ("Context:\n" + ARTICLE + "\n\nQuestion: When did the council meet?", T.RAG_QA),
    ("Summarize the following article in 3 bullet points.\n\n" + ARTICLE, T.SUMMARIZATION),
    ("tl;dr this please:\n\n" + ARTICLE, T.SUMMARIZATION),
    ("Write a Python function that checks whether a string is a palindrome.", T.CODE),
    ("Why does this fail?\n```python\nx = [1, 2]\nprint(x[5])\n```", T.CODE),
    ("Refactor this JavaScript to use async/await.", T.CODE),
    ("A train travels 120 km in 2 hours. How many hours will 300 km take at the same speed? "
     "Solve step by step.", T.REASONING),
    ("What is 17 * 23 + 4?", T.REASONING),
    ("Extract all person names and dates from the text below as JSON.\n\n" + ARTICLE, T.EXTRACTION),
    ("Classify the sentiment of each review.\n\nReview: Great phone!\nSentiment: positive\n\n"
     "Review: Battery died fast.\nSentiment:", T.CLASSIFICATION),
    ("Is this email spam or not? 'You won a free cruise, click here!'", T.CLASSIFICATION),
    ("Translate the following into French: I would like a cup of coffee, please.", T.TRANSFORMATION),
    ("Rewrite this email in a professional tone: hey boss, cant come tmrw, sick.", T.TRANSFORMATION),
    ("Fix the grammar in this sentence: me and him goes to school yesterday.", T.TRANSFORMATION),
    ("Write a short poem about the monsoon in Kerala.", T.CREATIVE),
    ("Write me a story about a robot who learns to paint.", T.CREATIVE),
    ("hi how are you", T.CHITCHAT),
    ("hello!", T.CHITCHAT),
    ("thanks!", T.CHITCHAT),
]


@pytest.mark.parametrize("prompt,expected", CASES, ids=[f"{c[1].value}-{i}" for i, c in enumerate(CASES)])
def test_task_type(prompt, expected):
    a = analyze(prompt)
    assert a.task is expected, (a.top(3), a.signals)


def test_scores_are_a_distribution():
    a = analyze("Summarize this report.")
    assert set(a.scores) == set(T)
    assert sum(a.scores.values()) == pytest.approx(1.0)
    assert a.confidence == a.scores[a.task] == max(a.scores.values())


def test_cue_inside_document_is_discounted():
    # "summarize"/"translate" inside a retrieved document must not decide the task.
    doc = ("Our team was asked to summarize and translate the report, then classify it. " * 5).strip()
    a = analyze(f"### Context\n{doc}\n\n### Question\nWho asked the team to do the work?")
    assert a.task is T.RAG_QA


def test_complexity_bounds_and_factors():
    for prompt, _ in CASES:
        a = analyze(prompt)
        assert 0.0 <= a.complexity <= 1.0
        assert all(0.0 <= v <= 1.0 for v in a.complexity_factors.values())
        assert a.complexity_level in {"low", "medium", "high"}


def test_complexity_ordering():
    trivial = analyze("hi")
    simple = analyze("What is the capital of France?")
    hard = analyze(
        "You are a senior backend engineer.\n\n"
        "Task: First, refactor the function below to be thread-safe. Then write unit tests. "
        "Finally, compare the old and new versions and explain why the change is safe.\n\n"
        "```python\ndef incr(counter):\n    counter['n'] += 1\n    return counter['n']\n```\n\n"
        "Constraints:\n- Must not change the public API.\n- Do not use global locks.\n"
        "- Only use the standard library.\n\nRespond in markdown with exactly 3 sections."
    )
    assert trivial.complexity < simple.complexity < hard.complexity
    assert trivial.complexity_level == "low"
    assert hard.complexity_level in {"medium", "high"}
    assert hard.task is T.CODE


def test_constraints_raise_complexity():
    base = "Write a product description for a water bottle."
    constrained = base + " Must be under 50 words. Do not mention price. Only use British spelling."
    assert analyze(constrained).complexity_factors["constraints"] > analyze(base).complexity_factors["constraints"]
    assert analyze(constrained).complexity > analyze(base).complexity


def test_accepts_chat_and_segmented_prompt():
    msgs = json.loads((EXAMPLES / "chat.json").read_text(encoding="utf-8"))
    a = analyze(msgs)
    assert a.features["is_chat"] == 1.0
    sp = segment(msgs)
    assert analyze(sp).scores == a.scores


def test_chat_history_does_not_dominate():
    msgs = [
        {"role": "user", "content": "Translate 'good morning' into Spanish."},
        {"role": "assistant", "content": "Buenos días."},
        {"role": "user", "content": "Now write a Python function that reverses a list."},
    ]
    assert analyze(msgs).task is T.CODE


def test_deterministic_and_serialisable():
    p = (EXAMPLES / "rag_prompt.txt").read_text(encoding="utf-8")
    a, b = analyze(p), analyze(p)
    assert a.to_dict() == b.to_dict()
    d = json.loads(json.dumps(a.to_dict()))
    assert d["task"] == "rag_qa" and d["signals"]
    assert {"task", "reasoning", "constraints", "length", "technical", "structure"} == set(d["complexity_factors"])


def test_features_cover_every_task():
    a = analyze("Write a haiku.")
    for t in T:
        assert f"cue_{t.value}" in a.features


def test_config_context_threshold():
    p = "Context: The Nile is about 6,650 km long.\nQuestion: How long is the Nile?"
    assert analyze(p).task is T.QA                                   # 7 words of context: closed-book
    assert analyze(p, AnalyzerConfig(context_min_words=5)).task is T.RAG_QA


def test_empty_prompt():
    a = analyze("")
    assert a.task in T and 0.0 <= a.complexity <= 1.0
