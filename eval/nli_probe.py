"""NLI probe for the M6 validator: raw NLI vs statement rewriting, with the threshold tuned.

Every pair is (original instruction, candidate after word deletion, ok?). "ok" means the
deletion keeps the task; "bad" means part of the task is lost. Deletions only, as in the loop.

Run:  python -m eval.nli_probe            (uses the real DeBERTa model, CUDA if available)
"""
from __future__ import annotations

import re
import sys

# (original, candidate, ok)
PAIRS: list[tuple[str, str, bool]] = [
    # ---- meaning kept (filler, politeness, articles, "following", hedges)
    ("Summarize the following article in 3 bullet points.", "Summarize the article in 3 bullet points.", True),
    ("Summarize the following article in 3 bullet points.", "Summarize article in 3 bullet points.", True),
    ("Summarize the following article in 3 bullet points.", "Summarize article 3 bullet points.", True),
    ("Please summarize the article below.", "summarize the article below.", True),
    ("Could you please explain how photosynthesis works?", "explain how photosynthesis works?", True),
    ("I would like you to write a short poem about the ocean.", "write a short poem about the ocean.", True),
    ("Kindly translate the following sentence into French.", "translate the sentence into French.", True),
    ("Can you help me to rewrite this paragraph so that it is more formal?", "rewrite this paragraph so that it is more formal?", True),
    ("Basically, just list all the main characters in the story.", "list all the main characters in the story.", True),
    ("Write a Python function that basically checks whether a number is prime.", "Write a Python function that checks whether a number is prime.", True),
    ("Explain, in simple terms, what a neural network actually is.", "Explain, in simple terms, what a neural network is.", True),
    ("Give me a list of the five largest countries by area.", "list the five largest countries by area.", True),
    ("Your task is to classify the sentiment of each review as positive or negative.", "classify the sentiment of each review as positive or negative.", True),
    ("Read the text carefully and then answer the question that follows.", "Read the text and answer the question that follows.", True),
    ("Please make sure that you respond in a friendly tone.", "respond in a friendly tone.", True),
    ("What is the capital city of Australia?", "What is the capital of Australia?", True),
    ("Can you tell me what the boiling point of water is at sea level?", "what the boiling point of water is at sea level?", True),
    ("Generate a catchy title for the following blog post about remote work.", "Generate a catchy title for the blog post about remote work.", True),
    ("In your answer, please include a brief explanation of your reasoning.", "include a brief explanation of your reasoning.", True),
    ("I want you to describe the main causes of the French Revolution.", "describe the main causes of the French Revolution.", True),
    ("Could you suggest three vegetarian dinner recipes for me?", "suggest three vegetarian dinner recipes?", True),
    ("Please compare the economic policies of the two candidates in detail.", "compare the economic policies of the two candidates in detail.", True),
    ("Now, using only the passage above, answer the question.", "using only the passage above, answer the question.", True),
    ("Summarize the article for a ten-year-old reader, please.", "Summarize the article for a ten-year-old reader.", True),
    # ---- meaning lost (a qualifier, object, audience or language dropped)
    ("Write a short poem about the ocean.", "Write a poem about the ocean.", False),
    ("Write a short poem about the ocean.", "Write a short poem.", False),
    ("Translate the following sentence into French.", "Translate the following sentence.", False),
    ("Explain, in simple terms, what a neural network is.", "Explain what a neural network is.", False),
    ("Rewrite this paragraph so that it is more formal.", "Rewrite this paragraph.", False),
    ("List the five largest countries by area.", "List the largest countries by area.", False),
    ("List the five largest countries by area.", "List the five largest countries.", False),
    ("Classify the sentiment of each review as positive or negative.", "Classify the sentiment of each review.", False),
    ("Write a Python function that checks whether a number is prime.", "Write a function that checks whether a number is prime.", False),
    ("Write a Python function that checks whether a number is prime.", "Write a Python function.", False),
    ("What is the capital city of Australia?", "What is the city of Australia?", False),
    ("Generate a catchy title for the blog post about remote work.", "Generate a title for the blog post about remote work.", False),
    ("Respond in a friendly tone.", "Respond.", False),
    ("Include a brief explanation of your reasoning.", "Include an explanation of your reasoning.", False),
    ("Summarize the article for a ten-year-old reader.", "Summarize the article.", False),
    ("Compare the economic policies of the two candidates.", "Compare the policies of the two candidates.", False),
    ("Describe the main causes of the French Revolution.", "Describe the French Revolution.", False),
    ("Suggest three vegetarian dinner recipes.", "Suggest three dinner recipes.", False),
    ("Answer the question using only information from the passage.", "Answer the question.", False),
    ("Please summarize the following article in 3 bullet points.", "summarize in 3 bullet points.", False),
    ("Could you please explain how photosynthesis works in plants?", "explain how photosynthesis works?", False),
    ("Write a formal email declining the meeting invitation.", "Write an email declining the meeting invitation.", False),
    ("Proofread the essay and fix any grammar mistakes.", "Proofread the essay.", False),
    ("Explain the difference between TCP and UDP.", "Explain TCP.", False),
]

# --------------------------------------------------------------------------- statement rewrites
_SENT = re.compile(r"[^.!?]+[.!?]*")


def _lcfirst(s: str) -> str:
    return s[:1].lower() + s[1:] if s[:2] != "I " else s


def rewrite_task(text: str) -> str:
    """Per sentence: 'The task is to <s>'; questions: 'The question is: <s>'."""
    out = []
    for s in (m.group().strip() for m in _SENT.finditer(text)):
        if not s:
            continue
        out.append(f"The question is: {s}" if s.endswith("?") else f"The task is to {_lcfirst(s)}")
    return " ".join(out)


def rewrite_asks(text: str) -> str:
    """Per sentence: 'The user asks the assistant to <s>'; questions: 'The user asks: <s>'."""
    out = []
    for s in (m.group().strip() for m in _SENT.finditer(text)):
        if not s:
            continue
        out.append(f"The user asks: {s}" if s.endswith("?") else f"The user asks the assistant to {_lcfirst(s)}")
    return " ".join(out)


def rewrite_said(text: str) -> str:
    """Whole segment: 'The user said: "<text>"'."""
    return f'The user said: "{text.strip()}"'


VARIANTS = {"raw": lambda t: t, "task": rewrite_task, "asks": rewrite_asks, "said": rewrite_said}


# --------------------------------------------------------------------------- metrics
def auc(pos: list[float], neg: list[float]) -> float:
    """P(score of an ok pair > score of a bad pair), ties count half."""
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def rates(pos, neg, thr):
    keep = sum(p >= thr for p in pos) / len(pos)      # ok edits allowed
    leak = sum(n >= thr for n in neg) / len(neg)      # bad edits allowed
    return keep, leak


def best_threshold(pos, neg, max_leak: float = 0.10):
    """Highest keep rate with leak ≤ max_leak; ties → the higher (safer) threshold."""
    best = (0.0, 1.0, 1.0)
    for thr in sorted(set(pos + neg + [1.0]), reverse=True):
        keep, leak = rates(pos, neg, thr)
        if leak <= max_leak and keep > best[0]:
            best = (keep, leak, thr)
    return best


def main() -> None:
    from aspo.validator import DebertaNLI
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    model = args[0] if args else "cross-encoder/nli-deberta-v3-small"
    nli = DebertaNLI(model)
    ok = [p[2] for p in PAIRS]
    print(f"model: {model} | pairs: {sum(ok)} ok, {len(ok) - sum(ok)} bad\n")

    rows = {}
    for name, fn in VARIANTS.items():
        pairs = []
        for a, b, _ in PAIRS:
            ra, rb = fn(a), fn(b)
            pairs += [(ra, rb), (rb, ra)]
        probs = nli.entailment(pairs)
        fwd, bwd = probs[0::2], probs[1::2]
        rows[name] = (fwd, bwd)

    print(f"{'variant':8} {'score':9} {'AUC':>5} | {'@0.80 keep/leak':>16} | {'tuned thr':>9} {'keep/leak':>11}")
    for name, (fwd, bwd) in rows.items():
        for sname, score in (("min", [min(f, b) for f, b in zip(fwd, bwd)]), ("backward", bwd), ("forward", fwd)):
            pos = [s for s, k in zip(score, ok) if k]
            neg = [s for s, k in zip(score, ok) if not k]
            k8, l8 = rates(pos, neg, 0.80)
            keep, leak, thr = best_threshold(pos, neg)
            print(f"{name:8} {sname:9} {auc(pos, neg):5.3f} | {k8:7.0%} / {l8:4.0%}   | {thr:9.3f} {keep:5.0%} / {leak:3.0%}")
        print()

    grid = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8)
    print("keep / leak of the min score on a fixed grid:")
    print(f"{'variant':8} " + " ".join(f"{t:>11}" for t in grid))
    for name, (fwd, bwd) in rows.items():
        score = [min(f, b) for f, b in zip(fwd, bwd)]
        pos = [s for s, k in zip(score, ok) if k]
        neg = [s for s, k in zip(score, ok) if not k]
        cells = (rates(pos, neg, t) for t in grid)
        print(f"{name:8} " + " ".join(f"{k:4.0%} / {l:3.0%}" for k, l in cells))
    print()

    if "-v" in sys.argv:
        print("per pair (min score): raw | task | asks | said")
        for i, (a, b, k) in enumerate(PAIRS):
            vals = " | ".join(f"{min(rows[n][0][i], rows[n][1][i]):.3f}" for n in VARIANTS)
            print(f"{'ok ' if k else 'BAD'} {vals}   {b}")


if __name__ == "__main__":
    main()
