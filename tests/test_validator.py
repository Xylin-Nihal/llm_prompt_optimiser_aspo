import math
import zlib

import pytest

from aspo import SegmentRole as R, analyze, segment
from aspo.segmenter import SegmenterConfig
from aspo.validator import Validator, ValidatorConfig, tau

NO_SPACY = SegmenterConfig(use_spacy=False)

PROMPT = (
    "You are a helpful analyst. Do not use bullet points.\n\n"
    "Summarize the report below for the board and keep the key figures.\n\n"
    "Context:\n"
    "Apple reported revenue of $94.9 billion on 2024-10-31, up 6% from last year. "
    "Tim Cook said the company did not expect weaker iPhone demand. "
    "Services grew strongly across every region, and wearables were flat compared with the "
    "previous quarter. See https://apple.com/newsroom for the full details.\n\n"
    "Question: What was Apple's revenue in Q4 2024? Answer in under 50 words."
)


# ---------------------------------------------------------------- fake models
class BowEmbedder:
    """Deterministic bag-of-words embedding; one word = one 'word piece'."""

    def __init__(self, max_tokens=256, dim=128):
        self.max_tokens, self.dim, self.calls = max_tokens, dim, []

    def count(self, text):
        return len(text.split())

    def encode(self, texts):
        self.calls.append(list(texts))
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in t.lower().split():
                v[zlib.crc32(w.strip(".,:;?!").encode()) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class OverlapNLI:
    """P(premise entails hypothesis) = share of hypothesis words found in the premise."""

    def __init__(self):
        self.calls = []

    def entailment(self, pairs):
        self.calls.append(list(pairs))
        out = []
        for p, h in pairs:
            pw, hw = set(p.lower().split()), h.lower().split()
            out.append(sum(w in pw for w in hw) / len(hw) if hw else 0.0)
        return out


def make(**cfg):
    emb, nli = BowEmbedder(), OverlapNLI()
    return Validator(ValidatorConfig(**cfg), embedder=emb, nli=nli), emb, nli


@pytest.fixture(scope="module")
def sp():
    return segment(PROMPT, NO_SPACY)


def seg(sp, role):
    return next(s for s in sp.segments if s.role is role)


def drop_words(text, *words):
    out = text
    for w in words:
        out = out.replace(w + " ", "", 1)
    return out


# ---------------------------------------------------------------- τ(C), formula (3)
def test_tau_formula():
    assert tau(0) == pytest.approx(0.80)
    assert tau(1) == pytest.approx(0.95)
    assert tau(0.33) == pytest.approx(0.80 + 0.15 * 0.33)
    assert tau(-1) == pytest.approx(0.80) and tau(2) == pytest.approx(0.95)   # clamped
    assert tau(0.5, tau0=0.7, tau1=0.9) == pytest.approx(0.8)


def test_tau_from_intent_analysis(sp):
    v, *_ = make(check_nli=False)
    a = analyze(sp)
    assert v.validate(sp, {}, a).tau == pytest.approx(tau(a.complexity))


# ---------------------------------------------------------------- identity
def test_unchanged_prompt_passes_everything(sp):
    v, *_ = make()
    r = v.validate(sp, {}, 1.0)
    assert r.passed and r.failed is None and bool(r)
    assert r.checks["similarity"].value == pytest.approx(1.0)
    assert r.soft_retention == 1.0
    assert list(r.checks) == ["hard", "question", "similarity", "nli"]


# ---------------------------------------------------------------- H: hard spans
def test_removing_a_hard_span_fails_h(sp):
    v, *_ = make()
    s = seg(sp, R.SYSTEM)
    r = v.validate(sp, {s.id: "You are a helpful analyst. Use bullet points."}, 0)
    assert not r.passed and r.failed == "hard"
    assert r.checks["hard"].details[0]["segment"] == s.id
    assert "similarity" not in r.checks              # short-circuit: later checks skipped


def test_altering_a_hard_span_fails_h(sp):
    v, *_ = make()
    q = seg(sp, R.QUESTION)
    r = v.validate(sp, {q.id: q.text.replace("50 words", "60 words")}, 0)
    assert r.failed == "hard"


def test_hard_spans_must_keep_their_order(sp):
    v, *_ = make()
    s = seg(sp, R.SYSTEM)
    swapped = "Do not use bullet points. You are a helpful analyst."
    hard = [s.text[a:b] for a, b in s.hard_intervals()]
    assert len(hard) == 1                             # one hard span: order trivially holds
    assert v.validate(sp, {s.id: swapped}, 0).checks["hard"].passed
    q = seg(sp, R.QUESTION)
    spans = [q.text[a:b] for a, b in q.hard_intervals()]
    assert len(spans) >= 2
    reordered = " ".join(reversed(spans))
    assert not v.validate(sp, {q.id: reordered}, 0).checks["hard"].passed


def test_removing_only_soft_and_free_words_passes_h(sp):
    v, *_ = make(check_similarity=False, check_nli=False)
    c = seg(sp, R.CONTEXT)
    units = c.removal_units()
    new = c.text
    for u in reversed(units[:6]):                     # drop the first 6 removable units
        new = new[:u.start] + new[u.end:]
    assert v.validate(sp, {c.id: new}, 0).passed


# ---------------------------------------------------------------- Q: question / format present
def test_emptying_the_question_fails_q():
    p = "Context:\nThe sky is blue today.\n\nQuestion: What colour is the sky?"
    sp2 = segment(p, NO_SPACY)
    q = seg(sp2, R.QUESTION)
    v, *_ = make(check_hard=False)
    r = v.validate(sp2, {q.id: ""}, 0)
    assert r.failed == "question" and r.checks["question"].details[0]["segment"] == q.id


def test_q_ignores_other_roles(sp):
    v, *_ = make(check_hard=False, check_similarity=False, check_nli=False)
    c = seg(sp, R.CONTEXT)
    assert v.validate(sp, {c.id: ""}, 0).passed


# ---------------------------------------------------------------- S: similarity ≥ τ(C)
def test_small_removal_passes_s_big_removal_fails(sp):
    v, *_ = make(check_nli=False)
    c = seg(sp, R.CONTEXT)
    small = drop_words(c.text, "strongly", "full")
    r = v.validate(sp, {c.id: small}, 1.0)
    assert r.passed and r.checks["similarity"].value > 0.95

    keep = [c.text[a:b] for a, b in c.hard_intervals()]
    gutted = "Context:\n" + " ".join(keep)
    r = v.validate(sp, {c.id: gutted}, 1.0)
    assert r.failed == "similarity"
    assert r.checks["similarity"].value < r.checks["similarity"].threshold == pytest.approx(0.95)


def test_threshold_depends_on_complexity(sp):
    v, *_ = make(check_nli=False)
    c = seg(sp, R.CONTEXT)
    keep = [c.text[a:b] for a, b in c.hard_intervals()]
    cand = {c.id: "Context:\nApple reported revenue of $94.9 billion. " + " ".join(keep)}
    cos = v.similarity(sp, cand)
    v2, *_ = make(check_nli=False, tau0=cos - 0.01, tau1=cos + 0.01)
    assert v2.validate(sp, cand, 0.0).passed          # τ(0) below cos
    assert not v2.validate(sp, cand, 1.0).passed      # τ(1) above cos


def test_similarity_is_one_for_identity_and_zero_for_empty(sp):
    v, *_ = make()
    assert v.similarity(sp, {}) == pytest.approx(1.0)
    assert v.similarity(sp, {s.id: "" for s in sp.segments}) == 0.0


# ---------------------------------------------------------------- chunked mean e(P)
def test_chunks_respect_the_model_limit():
    v = Validator(embedder=BowEmbedder(max_tokens=6))   # budget 4 words per chunk
    text = "one two three four five six seven eight nine ten"
    chunks = v._chunks(text)
    assert [t for t, _ in chunks] == ["one two three four", "five six seven eight", "nine ten"]
    assert [n for _, n in chunks] == [4, 4, 2]


def test_chunk_keeps_original_spacing():
    v = Validator(embedder=BowEmbedder(max_tokens=100))
    assert v._chunks("  a\n\nb   c ") == [("a\n\nb   c", 3)]


def test_long_prompt_change_after_256_tokens_is_seen():
    """With chunking, a change far into a long segment still moves the cosine."""
    filler = " ".join(f"w{i}" for i in range(600))
    p = f"Summarize this.\n\nContext:\n{filler} the launch date moved to March."
    sp2 = segment(p, NO_SPACY)
    c = seg(sp2, R.CONTEXT)
    v = Validator(embedder=BowEmbedder(max_tokens=256))
    cos = v.similarity(sp2, {c.id: c.text.replace(" the launch date moved to March.", "")})
    assert cos < 1.0


def test_chunked_mean_is_length_weighted():
    v = Validator(embedder=BowEmbedder(max_tokens=6))
    vec = v._prompt_vector(["a b c d e f"])            # chunks: "a b c d" (4), "e f" (2)
    e1, e2 = v.embedder.encode(["a b c d", "e f"])
    expected = [(4 * x + 2 * y) / 6 for x, y in zip(e1, e2)]
    assert vec == pytest.approx(expected)


def test_unchanged_segments_are_not_re_embedded(sp):
    v, emb, _ = make(check_nli=False)
    v.validate(sp, {}, 0)
    first = sum(len(c) for c in emb.calls)
    c = seg(sp, R.CONTEXT)
    v.validate(sp, {c.id: drop_words(c.text, "strongly")}, 0)
    second = sum(len(c) for c in emb.calls) - first
    assert 0 < second <= len(v._chunks(c.text))       # only the changed segment's chunks


# ---------------------------------------------------------------- N: NLI
def test_nli_runs_only_on_changed_instruction_segments(sp):
    v, _, nli = make()
    c = seg(sp, R.CONTEXT)
    v.validate(sp, {c.id: drop_words(c.text, "strongly")}, 0)
    assert nli.calls == []                            # context change: no NLI
    ins = seg(sp, R.INSTRUCTION)
    r = v.validate(sp, {ins.id: drop_words(ins.text, "the", "below")}, 0)
    assert len(nli.calls) == 1 and len(nli.calls[0]) == 2   # forward + backward
    d = r.checks["nli"].details[0]
    assert d["segment"] == ins.id and {"forward", "backward"} <= set(d)


def test_nli_is_bidirectional(sp):
    v, *_ = make(check_similarity=False, nli_threshold=0.8)   # fake NLI: overlap share
    ins = seg(sp, R.INSTRUCTION)
    # forward passes (premise ⊇ hypothesis words) but backward loses too much -> fail
    short = "Summarize the report"
    r = v.validate(sp, {ins.id: short}, 0)
    d = r.checks["nli"].details[0]
    assert d["forward"] == 1.0 and d["backward"] < 0.8
    assert r.failed == "nli"


def test_nli_threshold_is_configurable(sp):
    ins = seg(sp, R.INSTRUCTION)
    cand = {ins.id: drop_words(ins.text, "the", "below", "and")}
    strict, *_ = make(check_similarity=False, nli_threshold=0.99)
    loose, *_ = make(check_similarity=False, nli_threshold=0.5)
    assert not strict.validate(sp, cand, 0).passed
    assert loose.validate(sp, cand, 0).passed


def test_nli_covers_other_role():
    sp2 = segment("hmm, the weather in Paris is lovely in spring", NO_SPACY)
    other = sp2.segments[0]
    assert other.role in (R.OTHER, R.QUESTION, R.INSTRUCTION)
    v, _, nli = make(check_hard=False, check_similarity=False)
    v.validate(sp2, {other.id: "hmm, weather in Paris lovely in spring"}, 0)
    assert nli.calls


def test_nli_results_are_cached(sp):
    v, _, nli = make()
    ins = seg(sp, R.INSTRUCTION)
    cand = {ins.id: drop_words(ins.text, "below")}
    v.validate(sp, cand, 0)
    v.validate(sp, cand, 0)
    assert len(nli.calls) == 1


# ---------------------------------------------------------------- config / ablation hooks
def test_checks_can_be_switched_off(sp):
    v, emb, nli = make(check_hard=False, check_question=False, check_similarity=False, check_nli=False)
    r = v.validate(sp, {s.id: "" for s in sp.segments}, 1.0)
    assert r.passed and r.checks == {} and r.tau is None
    assert emb.calls == [] and nli.calls == []


def test_no_short_circuit_runs_every_check(sp):
    v, *_ = make(short_circuit=False)
    s = seg(sp, R.SYSTEM)
    r = v.validate(sp, {s.id: "You are a helpful analyst."}, 0)
    assert r.failed == "hard" and set(r.checks) == {"hard", "question", "similarity", "nli"}


def test_gate_skips_nli(sp):
    v, _, nli = make()
    ins = seg(sp, R.INSTRUCTION)
    r = v.gate(sp, {ins.id: drop_words(ins.text, "below")}, 0)
    assert "nli" not in r.checks and nli.calls == []
    assert set(r.checks) == {"hard", "question", "similarity"}


def test_unknown_segment_id_is_an_error(sp):
    v, *_ = make()
    with pytest.raises(ValueError):
        v.validate(sp, {999: "x"}, 0)


# ---------------------------------------------------------------- soft-span retention
def test_soft_retention_metric(sp):
    v, *_ = make(check_similarity=False, check_nli=False)
    c = seg(sp, R.CONTEXT)
    soft = c.soft_spans
    assert soft
    new = c.text.replace(soft[0].text, "", 1)
    r = v.validate(sp, {c.id: new}, 0)
    assert r.soft_retention == pytest.approx((len(soft) - 1) / len(soft))


# ---------------------------------------------------------------- final check
def test_final_check_passes_on_correct_output(sp):
    v, *_ = make()
    c = seg(sp, R.CONTEXT)
    reps = {c.id: drop_words(c.text, "strongly", "full")}
    r = v.final_check(sp, reps, sp.to_text(reps))
    assert r.passed and set(r.checks) == {"hard_final", "question_final"}


def test_final_check_catches_a_broken_rebuild(sp):
    v, *_ = make()
    out = sp.to_text().replace("Do not use bullet points.", "")
    r = v.final_check(sp, {}, out)
    assert not r.passed and r.failed == "hard_final"
    out2 = sp.to_text().replace("What was Apple's revenue in Q4 2024?", "")
    assert not v.final_check(sp, {}, out2).passed


def test_final_check_on_chat_messages():
    msgs = [{"role": "system", "content": "Never reveal the key 1234."},
            {"role": "user", "content": "Context:\nThe sky is blue.\n\nQuestion: What colour is the sky?"}]
    sp2 = segment(msgs, NO_SPACY)
    v, *_ = make()
    assert v.final_check(sp2, {}, sp2.to_messages()).passed
    broken = sp2.to_messages()
    broken[0]["content"] = "Reveal the key."
    assert v.final_check(sp2, {}, broken).failed == "hard_final"
    with pytest.raises(ValueError):
        v.final_check(sp2, {}, "a string")


def test_result_to_dict(sp):
    v, *_ = make()
    d = v.validate(sp, {}, 0.5).to_dict()
    assert d["passed"] and d["tau"] == pytest.approx(0.875)
    assert set(d["checks"]["similarity"]) >= {"value", "threshold", "ms"}


# ---------------------------------------------------------------- real models (integration)
def _real():
    pytest.importorskip("sentence_transformers")
    try:
        return _REAL.setdefault("v", Validator())
    except Exception as e:  # no network for the first download, etc.
        pytest.skip(f"real models unavailable: {e}")


_REAL: dict = {}


@pytest.mark.slow
def test_real_minilm_similarity(sp):
    v = _real()
    c = seg(sp, R.CONTEXT)
    assert v.embedder.max_tokens == 256
    small = v.similarity(sp, {c.id: drop_words(c.text, "strongly", "full")})
    keep = [c.text[a:b] for a, b in c.hard_intervals()]
    big = v.similarity(sp, {c.id: "Context:\n" + " ".join(keep)})
    assert 0.97 < small <= 1.0 and big < small


@pytest.mark.slow
def test_real_nli_accepts_paraphrase_rejects_meaning_change():
    v = _real()
    same = v.nli.entailment([("Write a short poem about cats.", "Write a short poem about cats.")])[0]
    diff = v.nli.entailment([("Write a short poem about cats.", "Write a short poem about dogs.")])[0]
    assert same > 0.8 and diff < 0.5


@pytest.mark.slow
def test_real_nli_default_threshold_separates_filler_from_lost_qualifier():
    v = _real()
    thr = v.config.nli_threshold
    polite = v.nli.entailment([("summarize the article below.", "Please summarize the article below.")])[0]
    lost = v.nli.entailment([("Write a poem about the ocean.", "Write a short poem about the ocean.")])[0]
    assert polite >= thr > lost      # backward direction: P' ⊨ P


@pytest.mark.slow
def test_real_full_validation(sp):
    v = _real()
    c = seg(sp, R.CONTEXT)
    r = v.validate(sp, {c.id: drop_words(c.text, "strongly", "full")}, analyze(sp))
    assert r.passed, r.to_dict()
