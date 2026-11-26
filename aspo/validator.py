"""Module 6 - Validator.

Decides whether a compressed candidate P' may replace the original prompt P.
Formula (9) of the plan:

    V(P') = 1  iff  H(P') ∧ Q(P') ∧ [cos(e(P), e(P')) ≥ τ(C)] ∧ N(P')
    τ(C)  = τ0 + (τ1 − τ0)·C,   τ0 = 0.80, τ1 = 0.95                      (3)

* **H** ("hard")       - every hard span of each segment is present verbatim, in order.
* **Q** ("question")   - every QUESTION and FORMAT segment still has text.
* **S** ("similarity") - cos(e(P), e(P')) ≥ τ(C), with e = all-MiniLM-L6-v2.
* **N** ("nli")        - every changed instruction-type segment (INSTRUCTION,
  QUESTION, FORMAT, SYSTEM, OTHER) is entailed both ways by its original, p ≥ 0.8
  (cross-encoder/nli-deberta-v3-small).

Checks run cheapest first (H, Q, S, N). Inside the removal loop the validator
stops at the first failure (``short_circuit``); for evaluation it can compute
every value. ``gate()`` is H ∧ Q ∧ S, the reward gate G of formula (12).
``final_check()`` re-runs H and Q on the rebuilt output after the loop.

**Long prompts.** MiniLM reads at most 256 word pieces, so one embedding of a
long prompt would ignore everything after that point. e(P) is therefore the
*chunked mean*: each segment is split into chunks that fit the model, every
chunk is embedded, and e(P) is the mean of the chunk embeddings weighted by
their word-piece counts:

    e(P) = Σ_c n_c·e(c) / Σ_c n_c      (chunks c never cross segment boundaries)

Chunk embeddings are cached by text, so unchanged segments cost nothing in later
loop steps.

The embedder and NLI model are small interfaces (``Embedder``, ``NLIModel``); the
real ones load lazily, and tests pass fakes.
"""

from __future__ import annotations

import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .types import IntentAnalysis, Segment, SegmentedPrompt, SegmentRole

R = SegmentRole

__all__ = [
    "ValidatorConfig", "Validator", "ValidationResult", "CheckResult", "Embedder", "NLIModel",
    "MiniLMEmbedder", "DebertaNLI", "tau",
]

QUESTION_ROLES = frozenset({R.QUESTION, R.FORMAT})
NLI_ROLES = frozenset({R.INSTRUCTION, R.QUESTION, R.FORMAT, R.SYSTEM, R.OTHER})
_WORD = re.compile(r"\S+")


# =========================================================================== config
@dataclass
class ValidatorConfig:
    # which checks run (ablation hooks)
    check_hard: bool = True
    check_question: bool = True
    check_similarity: bool = True
    check_nli: bool = True
    # thresholds
    tau0: float = 0.80                 # τ(0): simplest prompt
    tau1: float = 0.95                 # τ(1): most complex prompt
    nli_threshold: float = 0.05        # chosen by eval/nli_probe.py (2026-11-27): DeBERTa-small
                                       # scores lost qualifiers ≈0.00–0.04, kept-meaning edits mostly
                                       # >0.05; at 0.80 it rejected half the meaning-preserving edits
    nli_roles: frozenset[SegmentRole] = NLI_ROLES
    question_roles: frozenset[SegmentRole] = QUESTION_ROLES
    # behaviour
    short_circuit: bool = True         # stop at the first failed check (loop mode)
    # models
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    device: str | None = None          # None = CUDA if available, else CPU
    cache_size: int = 50_000           # cached chunk embeddings / NLI pairs


def tau(complexity: float, tau0: float = 0.80, tau1: float = 0.95) -> float:
    """Formula (3): τ(C) = τ0 + (τ1 − τ0)·C, with C clamped to [0, 1]."""
    c = min(1.0, max(0.0, float(complexity)))
    return tau0 + (tau1 - tau0) * c


# =========================================================================== results
@dataclass
class CheckResult:
    name: str                          # "hard" | "question" | "similarity" | "nli" | "*_final"
    passed: bool
    value: float | None = None         # e.g. cosine, or the lowest NLI probability
    threshold: float | None = None
    details: list[dict[str, Any]] = field(default_factory=list)   # what failed / per-item scores
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed,
                "value": None if self.value is None else round(self.value, 4),
                "threshold": None if self.threshold is None else round(self.threshold, 4),
                "details": self.details, "ms": round(self.seconds * 1000, 2)}


@dataclass
class ValidationResult:
    passed: bool
    failed: str | None                 # first failed check: the loop's stop reason
    checks: dict[str, CheckResult]     # checks that ran, in order
    tau: float | None = None
    soft_retention: float | None = None   # share of soft spans kept verbatim (metric)
    seconds: float = 0.0

    def __bool__(self) -> bool:
        return self.passed

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "failed": self.failed,
                "tau": None if self.tau is None else round(self.tau, 4),
                "soft_retention": None if self.soft_retention is None else round(self.soft_retention, 4),
                "ms": round(self.seconds * 1000, 2),
                "checks": {k: c.to_dict() for k, c in self.checks.items()}}


# =========================================================================== model interfaces
class Embedder(Protocol):
    max_tokens: int                                   # longest input the model reads (word pieces)

    def count(self, text: str) -> int: ...           # word pieces in text (no special tokens)
    def encode(self, texts: list[str]) -> list[Sequence[float]]: ...


class NLIModel(Protocol):
    def entailment(self, pairs: list[tuple[str, str]]) -> list[float]: ...   # P(entail | premise, hyp)


def _device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class MiniLMEmbedder:
    """all-MiniLM-L6-v2 through sentence-transformers (L2-normalised embeddings)."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model, device=_device(device))
        self.max_tokens = int(self.model.max_seq_length)
        self._tok = self.model.tokenizer

    def count(self, text: str) -> int:
        return len(self._tok.tokenize(text))

    def encode(self, texts: list[str]) -> list[Sequence[float]]:
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).tolist()


class DebertaNLI:
    """cross-encoder/nli-deberta-v3-small; returns P(entailment) per (premise, hypothesis)."""

    def __init__(self, model: str = "cross-encoder/nli-deberta-v3-small", device: str | None = None):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model, device=_device(device))
        id2label = self.model.model.config.id2label
        self._entail = next(i for i, lab in id2label.items() if lab.lower().startswith("entail"))

    def entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        logits = self.model.predict(pairs, batch_size=16, convert_to_numpy=True,
                                    show_progress_bar=False).tolist()
        out = []
        for row in logits:
            row = list(row)
            if abs(sum(row) - 1.0) < 1e-4 and min(row) >= 0:   # already probabilities
                out.append(float(row[self._entail]))
                continue
            m = max(row)
            exps = [math.exp(x - m) for x in row]
            out.append(exps[self._entail] / sum(exps))
        return out


class _LRU(OrderedDict):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def get_(self, key):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None

    def put(self, key, value) -> None:
        self[key] = value
        self.move_to_end(key)
        if len(self) > self.size:
            self.popitem(last=False)


# =========================================================================== validator
class Validator:
    def __init__(self, config: ValidatorConfig | None = None,
                 embedder: Embedder | None = None, nli: NLIModel | None = None):
        self.config = config or ValidatorConfig()
        self._embedder = embedder
        self._nli = nli
        self._emb_cache = _LRU(self.config.cache_size)    # chunk text -> embedding
        self._nli_cache = _LRU(self.config.cache_size)    # (premise, hyp) -> P(entail)
        self._wp_cache = _LRU(self.config.cache_size)     # word -> word-piece count

    # ---------------------------------------------------------------- lazy models
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = MiniLMEmbedder(self.config.embed_model, self.config.device)
        return self._embedder

    @property
    def nli(self) -> NLIModel:
        if self._nli is None:
            self._nli = DebertaNLI(self.config.nli_model, self.config.device)
        return self._nli

    # ---------------------------------------------------------------- public API
    def validate(self, sp: SegmentedPrompt, replacements: Mapping[int, str],
                 complexity: float | IntentAnalysis = 0.0) -> ValidationResult:
        """Formula (9): run H, Q, S, N on the candidate ``sp.to_text(replacements)``."""
        cfg = self.config
        return self._run(sp, replacements, complexity,
                         hard=cfg.check_hard, question=cfg.check_question,
                         similarity=cfg.check_similarity, nli=cfg.check_nli)

    def gate(self, sp: SegmentedPrompt, replacements: Mapping[int, str],
             complexity: float | IntentAnalysis = 0.0) -> ValidationResult:
        """Formula (12) gate G = H ∧ Q ∧ [cos ≥ τ(C)]: the validator without NLI."""
        cfg = self.config
        return self._run(sp, replacements, complexity,
                         hard=cfg.check_hard, question=cfg.check_question,
                         similarity=cfg.check_similarity, nli=False)

    def similarity(self, sp: SegmentedPrompt, replacements: Mapping[int, str]) -> float:
        """cos(e(P), e(P')) with the chunked-mean embedding (used by the reward too)."""
        orig = [s.text for s in sp.segments]
        new = [replacements.get(s.id, s.text) for s in sp.segments]
        return _cos(self._prompt_vector(orig), self._prompt_vector(new))

    def final_check(self, sp: SegmentedPrompt, replacements: Mapping[int, str],
                    output: str | list[dict[str, Any]]) -> ValidationResult:
        """After the loop: H and Q again, on the *rebuilt output* (text or messages).

        Catches bugs in the loop or in reconstruction. If it fails, the pipeline
        sends the original prompt and logs ``failed``."""
        t0 = time.perf_counter()
        _check_ids(sp, replacements)
        sources = _output_sources(sp, output)
        checks: dict[str, CheckResult] = {}

        t = time.perf_counter()
        missing = []
        for si, text in enumerate(sources):
            cursor = 0
            for seg in (s for s in sp.segments if s.source_index == si):
                for a, b in seg.hard_intervals():
                    span = seg.text[a:b]
                    pos = text.find(span, cursor)
                    if pos < 0:
                        missing.append({"segment": seg.id, "span": span})
                    else:
                        cursor = pos + len(span)
        checks["hard_final"] = CheckResult("hard_final", not missing, details=missing,
                                           seconds=time.perf_counter() - t)

        t = time.perf_counter()
        lost = []
        for seg in sp.segments:
            if seg.role in self.config.question_roles:
                new = replacements.get(seg.id, seg.text).strip()
                if not new or new not in sources[seg.source_index]:
                    lost.append({"segment": seg.id, "role": seg.role.value})
        checks["question_final"] = CheckResult("question_final", not lost, details=lost,
                                               seconds=time.perf_counter() - t)

        failed = next((k for k, c in checks.items() if not c.passed), None)
        return ValidationResult(failed is None, failed, checks,
                                soft_retention=_soft_retention(sp, replacements),
                                seconds=time.perf_counter() - t0)

    # ---------------------------------------------------------------- the checks
    def _run(self, sp, replacements, complexity, *, hard, question, similarity, nli) -> ValidationResult:
        t0 = time.perf_counter()
        _check_ids(sp, replacements)
        c = complexity.complexity if isinstance(complexity, IntentAnalysis) else float(complexity)
        threshold = tau(c, self.config.tau0, self.config.tau1)
        plan = [("hard", hard, self._check_hard), ("question", question, self._check_question),
                ("similarity", similarity, lambda s, r: self._check_similarity(s, r, threshold)),
                ("nli", nli, self._check_nli)]
        checks: dict[str, CheckResult] = {}
        failed = None
        for name, enabled, fn in plan:
            if not enabled:
                continue
            t = time.perf_counter()
            res = fn(sp, replacements)
            res.seconds = time.perf_counter() - t
            checks[name] = res
            if not res.passed and failed is None:
                failed = name
                if self.config.short_circuit:
                    break
        return ValidationResult(failed is None, failed, checks,
                                tau=threshold if similarity else None,
                                soft_retention=_soft_retention(sp, replacements),
                                seconds=time.perf_counter() - t0)

    def _check_hard(self, sp: SegmentedPrompt, replacements: Mapping[int, str]) -> CheckResult:
        """H: the merged hard intervals of each segment appear verbatim, in order."""
        missing = []
        for seg in sp.segments:
            if seg.id not in replacements:
                continue
            new, cursor = replacements[seg.id], 0
            for a, b in seg.hard_intervals():
                span = seg.text[a:b]
                pos = new.find(span, cursor)
                if pos < 0:
                    missing.append({"segment": seg.id, "role": seg.role.value, "span": span})
                else:
                    cursor = pos + len(span)
        return CheckResult("hard", not missing, details=missing)

    def _check_question(self, sp: SegmentedPrompt, replacements: Mapping[int, str]) -> CheckResult:
        """Q: no QUESTION or FORMAT segment is removed entirely."""
        lost = [{"segment": s.id, "role": s.role.value}
                for s in sp.segments
                if s.role in self.config.question_roles and s.text.strip()
                and not replacements.get(s.id, s.text).strip()]
        return CheckResult("question", not lost, details=lost)

    def _check_similarity(self, sp, replacements, threshold: float) -> CheckResult:
        """S: cos(e(P), e(P')) ≥ τ(C)."""
        cos = self.similarity(sp, replacements)
        return CheckResult("similarity", cos >= threshold, value=cos, threshold=threshold)

    def _check_nli(self, sp: SegmentedPrompt, replacements: Mapping[int, str]) -> CheckResult:
        """N: each changed instruction-type segment is entailed both ways, p ≥ threshold."""
        thr = self.config.nli_threshold
        changed = [s for s in sp.segments
                   if s.role in self.config.nli_roles and s.id in replacements
                   and replacements[s.id] != s.text]
        if not changed:
            return CheckResult("nli", True, value=None, threshold=thr)

        details, scores = [], []
        pairs: list[tuple[str, str]] = []
        for s in changed:
            new = replacements[s.id]
            if not new.strip():       # an emptied instruction cannot entail its original
                details.append({"segment": s.id, "role": s.role.value, "forward": 0.0,
                                "backward": 0.0, "passed": False, "note": "segment removed"})
                scores.append(0.0)
                continue
            pairs += [(s.text, new), (new, s.text)]
        probs = iter(self._entailment(pairs))
        for s in changed:
            new = replacements[s.id]
            if not new.strip():
                continue
            fwd, bwd = next(probs), next(probs)   # P ⊨ P' (nothing added), P' ⊨ P (nothing lost)
            low = min(fwd, bwd)
            scores.append(low)
            details.append({"segment": s.id, "role": s.role.value, "forward": round(fwd, 4),
                            "backward": round(bwd, 4), "passed": low >= thr})
        value = min(scores)
        return CheckResult("nli", value >= thr, value=value, threshold=thr, details=details)

    # ---------------------------------------------------------------- embeddings
    def _chunks(self, text: str) -> list[tuple[str, int]]:
        """Split one segment into (chunk text, word pieces) that fit the embedder.
        Greedy over whitespace words; chunk text keeps the original spacing."""
        emb = self.embedder
        budget = max(1, emb.max_tokens - 2)          # room for [CLS] and [SEP]
        chunks: list[tuple[str, int]] = []
        start = end = None
        n = 0
        for m in _WORD.finditer(text):
            w = self._wp_cache.get_(m.group())
            if w is None:
                w = max(1, emb.count(m.group()))
                self._wp_cache.put(m.group(), w)
            if start is not None and n + w > budget:
                chunks.append((text[start:end], n))
                start, n = None, 0
            if start is None:
                start = m.start()
            end, n = m.end(), n + w
        if start is not None:
            chunks.append((text[start:end], n))
        return chunks

    def _prompt_vector(self, segment_texts: list[str]) -> list[float] | None:
        """Chunked mean e(P) = Σ n_c·e(c) / Σ n_c over all chunks of all segments."""
        chunks = [c for t in segment_texts for c in self._chunks(t)]
        if not chunks:
            return None
        todo = list(dict.fromkeys(t for t, _ in chunks if self._emb_cache.get_(t) is None))
        if todo:
            for t, v in zip(todo, self.embedder.encode(todo)):
                self._emb_cache.put(t, [float(x) for x in v])
        total = sum(n for _, n in chunks)
        vec: list[float] | None = None
        for t, n in chunks:
            e = self._emb_cache.get_(t)
            if e is None:                                # evicted mid-call (tiny cache)
                e = [float(x) for x in self.embedder.encode([t])[0]]
            w = n / total
            vec = [w * x for x in e] if vec is None else [a + w * x for a, x in zip(vec, e)]
        return vec

    def _entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        todo = list(dict.fromkeys(p for p in pairs if self._nli_cache.get_(p) is None))
        if todo:
            for p, prob in zip(todo, self.nli.entailment(todo)):
                self._nli_cache.put(p, float(prob))
        return [self._nli_cache.get_(p) if p in self._nli_cache else self.nli.entailment([p])[0]
                for p in pairs]


# =========================================================================== helpers
def _cos(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None:
        return 1.0 if a is None and b is None else 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _check_ids(sp: SegmentedPrompt, replacements: Mapping[int, str]) -> None:
    unknown = set(replacements) - {s.id for s in sp.segments}
    if unknown:
        raise ValueError(f"replacements name unknown segment ids: {sorted(unknown)}")


def _soft_retention(sp: SegmentedPrompt, replacements: Mapping[int, str]) -> float | None:
    """Share of soft spans still present verbatim in their segment."""
    total = kept = 0
    for seg in sp.segments:
        new = replacements.get(seg.id, seg.text)
        for p in seg.soft_spans:
            total += 1
            kept += p.text in new
    return kept / total if total else None


def _output_sources(sp: SegmentedPrompt, output: str | list[dict[str, Any]]) -> list[str]:
    """The text of every source in the rebuilt output, aligned with ``sp.sources``."""
    if isinstance(output, str):
        if sp.is_chat:
            raise ValueError("chat prompt: pass the output messages, not a string")
        return [output]
    if not sp.is_chat:
        raise ValueError("text prompt: pass the output string, not messages")
    out = []
    for src in sp.sources:
        content = output[src.message_index].get("content")
        out.append(content if src.part_index is None else content[src.part_index]["text"])
    return out
