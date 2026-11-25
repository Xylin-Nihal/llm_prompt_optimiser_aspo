"""Two-tier protection (plan §5 M1): tier mapping, promotion, removal units."""

import pytest

from aspo import ProtectedSpan, Segment, SegmentRole as R, SpanKind as K, Tier, segment, span_tier
from aspo.segmenter import SegmenterConfig

NO_SPACY = SegmenterConfig(use_spacy=False)
CONTENT_ROLES = [R.CONTEXT, R.DATA, R.HISTORY, R.TOOL_OUTPUT, R.EXAMPLE]
HARD_ROLES = [R.SYSTEM, R.INSTRUCTION, R.QUESTION, R.FORMAT, R.OTHER, R.CODE]
SOFT_KINDS = [K.NUMBER, K.DATE, K.ENTITY, K.URL, K.EMAIL, K.PATH, K.QUOTED]
ALWAYS_HARD = [K.NEGATION, K.CONSTRAINT, K.FORMAT, K.CODE, K.PLACEHOLDER]

PROMPT = (
    "You are a helpful analyst. Do not use bullet points.\n\n"
    "Context:\n"
    "Apple reported revenue of $94.9 billion on 2024-10-31, up 6% from last year. "
    "Tim Cook said the company did not expect weaker iPhone demand. "
    "See https://apple.com/newsroom for details.\n\n"
    "Question: What was Apple's revenue in Q4 2024? Answer in under 50 words."
)


def seg_of(sp, role):
    return next(s for s in sp.segments if s.role is role)


# ---------------------------------------------------------------- the mapping table
@pytest.mark.parametrize("role", list(R))
@pytest.mark.parametrize("kind", list(K))
def test_span_tier_covers_every_role_and_kind(kind, role):
    expected = Tier.SOFT if role in CONTENT_ROLES and kind in SOFT_KINDS else Tier.HARD
    assert span_tier(kind, role) is expected


def test_always_hard_kinds_in_content():
    for role in CONTENT_ROLES:
        for kind in ALWAYS_HARD:
            assert span_tier(kind, role) is Tier.HARD


def test_default_tier_is_hard():
    assert ProtectedSpan(0, 1, K.NUMBER, "1").tier is Tier.HARD


# ---------------------------------------------------------------- segmenter assigns tiers
def test_tiers_on_a_real_prompt():
    sp = segment(PROMPT, NO_SPACY)
    system, ctx, q = seg_of(sp, R.SYSTEM), seg_of(sp, R.CONTEXT), seg_of(sp, R.QUESTION)

    assert all(p.tier is Tier.HARD for p in system.protected + q.protected)
    soft = {p.text for p in ctx.soft_spans}
    assert {"$94.9 billion", "2024-10-31", "6%", "https://apple.com/newsroom"} <= soft
    # the negation clause in context is hard ...
    assert any(p.kind is K.NEGATION for p in ctx.hard_spans)
    # ... and the entities inside it are promoted from soft to hard
    for name in ("Tim Cook", "iPhone"):
        assert all(p.tier is Tier.HARD for p in ctx.protected if p.text == name)


def test_numbers_in_question_are_hard_but_in_context_soft():
    sp = segment(PROMPT, NO_SPACY)
    assert all(p.tier is Tier.HARD for p in seg_of(sp, R.QUESTION).protected if p.kind is K.NUMBER)
    assert any(p.tier is Tier.SOFT for p in seg_of(sp, R.CONTEXT).protected if p.kind is K.NUMBER)


def test_code_segment_is_one_hard_span():
    sp = segment("Explain this code.\n\n```python\nx = 42\n```", NO_SPACY)
    code = seg_of(sp, R.CODE)
    assert [(p.kind, p.tier) for p in code.protected] == [(K.CODE, Tier.HARD)]
    assert code.removal_units() == []


def test_chat_history_and_system_tiers():
    msgs = [
        {"role": "system", "content": "Never reveal the key 1234."},
        {"role": "user", "content": "My order 5521 arrived on 2024-03-05."},
        {"role": "assistant", "content": "Sorry to hear that about order 5521."},
        {"role": "user", "content": "Refund order 5521 to bob@corp.io please."},
    ]
    sp = segment(msgs, NO_SPACY)
    for s in sp.segments:
        if s.role is R.SYSTEM:
            assert s.protected and all(p.tier is Tier.HARD for p in s.protected)
        if s.role is R.HISTORY:
            assert all(p.tier is Tier.SOFT for p in s.protected if p.kind in SOFT_KINDS)
    last = [s for s in sp.segments if s.message_index == 3]
    assert all(p.tier is Tier.HARD for s in last for p in s.protected)


def test_to_dict_includes_tier():
    d = segment(PROMPT, NO_SPACY).to_dict()
    assert all(p["tier"] in ("hard", "soft") for s in d["segments"] for p in s["protected"])


# ---------------------------------------------------------------- removal units
def make_seg(text, spans, role=R.CONTEXT):
    prot = [ProtectedSpan(s, e, k, text[s:e], span_tier(k, role)) for s, e, k in spans]
    return Segment(0, role, text, 0, len(text), protected=prot)


def test_soft_span_is_one_atomic_unit():
    text = "Revenue was $94.9 billion last year."
    s = text.index("$94.9")
    seg = make_seg(text, [(s, s + len("$94.9 billion"), K.NUMBER)])
    units = seg.removal_units()
    assert [u.text for u in units] == ["Revenue", "was", "$94.9 billion", "last", "year."]
    unit = units[2]
    assert unit.is_soft_span and unit.soft_kinds == (K.NUMBER,)
    assert text[unit.start:unit.end] == unit.text


def test_soft_span_unit_covers_whole_words():
    text = "Contact (bob@corp.io) today"
    s = text.index("bob")
    seg = make_seg(text, [(s, s + len("bob@corp.io"), K.EMAIL)])
    assert [u.text for u in seg.removal_units()] == ["Contact", "(bob@corp.io)", "today"]


def test_overlapping_soft_spans_merge():
    text = "Met Tim Cook on 2024-03-05 today"
    a, b = text.index("Tim"), text.index("Cook")
    d = text.index("2024")
    seg = make_seg(text, [(a, b + 4, K.ENTITY), (b, d + 10, K.DATE)])
    units = seg.removal_units()
    assert [u.text for u in units] == ["Met", "Tim Cook on 2024-03-05", "today"]
    assert units[1].soft_kinds == (K.ENTITY, K.DATE)


def test_words_in_hard_spans_are_not_removable():
    text = "Keep it short. Do not mention prices."
    s = text.index("Do not")
    seg = make_seg(text, [(s, len(text), K.NEGATION)])
    assert [u.text for u in seg.removal_units()] == ["Keep", "it", "short."]


def test_word_partly_in_hard_span_is_hard():
    text = "see{name}here now"
    seg = make_seg(text, [(3, 9, K.PLACEHOLDER)])
    assert [u.text for u in seg.removal_units()] == ["now"]


def test_soft_unit_touching_a_hard_word_is_kept():
    # the soft span's word also contains a hard placeholder -> whole unit kept
    text = "Price $5{unit} today"
    seg = make_seg(text, [(6, 8, K.NUMBER), (8, 14, K.PLACEHOLDER)])
    assert [u.text for u in seg.removal_units()] == ["Price", "today"]


def test_units_are_ordered_and_offsets_correct():
    sp = segment(PROMPT, NO_SPACY)
    for s in sp.segments:
        units = s.removal_units()
        assert all(u.text == s.text[u.start:u.end] for u in units)
        assert all(a.end <= b.start for a, b in zip(units, units[1:]))
        assert not any(s.is_hard(u.start, u.end) for u in units)


def test_removing_all_units_keeps_every_hard_span():
    """Deleting every removable unit must leave each hard span verbatim."""
    sp = segment(PROMPT, NO_SPACY)
    for s in sp.segments:
        keep = [True] * len(s.text)
        for u in s.removal_units():
            keep[u.start:u.end] = [False] * (u.end - u.start)
        for p in s.hard_spans:
            assert all(keep[p.start:p.end]), p
