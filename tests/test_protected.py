from aspo.protected import detect_protected_spans
from aspo.types import SpanKind


def spans(text, kind=None, **kw):
    out = detect_protected_spans(text, use_spacy=False, **kw)
    return [s.text for s in out if kind is None or s.kind is kind]


def test_offsets_match_text():
    text = "Visit https://example.com/docs on 2024-03-05 and don't skip step 3."
    for sp in detect_protected_spans(text, use_spacy=False):
        assert text[sp.start:sp.end] == sp.text


def test_code_fences_and_inline_code():
    text = "Fix this:\n```python\ndef f(x):\n    return x + 1\n```\nand call `f(2)`."
    code = spans(text, SpanKind.CODE)
    assert any(c.startswith("```python") and c.endswith("```") for c in code)
    assert "`f(2)`" in code


def test_numbers_inside_code_are_not_duplicated():
    text = "```\nx = 42\n```"
    assert spans(text, SpanKind.NUMBER) == []


def test_urls_emails_paths():
    text = "Email bob@corp.io, read https://a.b/c?d=1. Edit src/app/main.py and config.yaml."
    assert "https://a.b/c?d=1" in spans(text, SpanKind.URL)
    assert "bob@corp.io" in spans(text, SpanKind.EMAIL)
    paths = spans(text, SpanKind.PATH)
    assert "src/app/main.py" in paths and "config.yaml" in paths


def test_numbers_money_units_versions():
    text = "Revenue was $94.9 billion in 2024, up 6%. Use v1.2.3 within 200 ms and 1,200 words."
    nums = spans(text, SpanKind.NUMBER)
    for expected in ["$94.9 billion", "2024", "6%", "v1.2.3", "1,200 words"]:
        assert expected in nums, (expected, nums)


def test_dates():
    text = "Due on November 14, 2024 or 2024-11-14 at 10:30 AM."
    dates = spans(text, SpanKind.DATE)
    assert "November 14, 2024" in dates and "2024-11-14" in dates


def test_placeholders():
    text = "Hello {name}, your code is {{otp}}. Wrap in <context></context>. Sign as [YOUR NAME] using $API_KEY."
    ph = spans(text, SpanKind.PLACEHOLDER)
    for expected in ["{name}", "{{otp}}", "<context>", "</context>", "[YOUR NAME]", "$API_KEY"]:
        assert expected in ph, (expected, ph)


def test_quoted_strings_ignore_apostrophes():
    text = "Reply with \"ACK\" only. It's fine, don't worry. Say 'hello world'."
    q = spans(text, SpanKind.QUOTED)
    assert '"ACK"' in q and "'hello world'" in q
    assert not any("s fine" in x for x in q)


def test_negation_protects_whole_clause():
    text = "Summarise the report. Do not mention competitors, and keep it short. Thanks."
    neg = spans(text, SpanKind.NEGATION)
    assert neg == ["Do not mention competitors, and keep it short."]


def test_contracted_negation():
    neg = spans("The drug isn't approved for children.", SpanKind.NEGATION)
    assert neg and "isn't approved" in neg[0]


def test_constraint_and_format_sentences():
    text = "Write a story about a dragon. It must be under 300 words. Respond in JSON with keys title and body."
    cons = spans(text, SpanKind.CONSTRAINT)
    fmt = spans(text, SpanKind.FORMAT)
    assert "It must be under 300 words." in cons
    assert any("Respond in JSON" in f for f in fmt)
    assert not any("dragon" in c for c in cons)


def test_kinds_filter():
    text = "You must answer. Paris is lovely."
    only_entities = detect_protected_spans(text, kinds={SpanKind.ENTITY}, use_spacy=False)
    assert all(s.kind is SpanKind.ENTITY for s in only_entities)
    assert any(s.text == "Paris" for s in only_entities)


def test_entities_skip_sentence_initial_verbs_and_stopwords():
    text = "Summarize Apple's report. The Eiffel Tower is in Paris. I like NASA and GPT-4o."
    ents = spans(text, SpanKind.ENTITY)
    assert "Summarize" not in " ".join(ents)
    assert "Eiffel Tower" in ents
    assert "Paris" in ents and "NASA" in ents and "GPT-4o" in ents
    assert "I" not in ents and "The" not in ents


def test_no_spans_in_plain_text():
    assert spans("hello there, how are you doing today", ) == []
