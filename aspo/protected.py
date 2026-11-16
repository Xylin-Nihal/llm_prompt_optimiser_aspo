"""Protected-span detection.

A *protected span* is a region of a prompt that later optimisation stages must
never delete or rewrite: code, literals, numbers, named entities, negated
clauses, hard constraints and output-format instructions. Dropping any of these
is the most common way prompt compressors silently change a prompt's meaning
(e.g. removing "not", or "in 3 bullet points").

Detection is rule-based (fast, CPU-only, deterministic). If spaCy and an English
model are installed, named-entity recognition is added on top.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

from .lexicon import ENTITY_STOPWORDS, IMPERATIVE_VERBS, SMALL_NUMBER_WORDS
from .types import ATOMIC_KINDS, ProtectedSpan, SpanKind

ALL_KINDS = frozenset(SpanKind)

# --------------------------------------------------------------------- patterns
_FENCE = re.compile(r"(?ms)^[ \t]*(```|~~~)[^\n]*\n.*?(?:^[ \t]*\1[ \t]*$|\Z)")
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_URL = re.compile(r"\b(?:https?://|ftp://|www\.)[^\s<>\"'`)\]}]+(?<![.,;:!?])")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_FILE_EXT = (
    r"py|ipynb|js|jsx|ts|tsx|json|jsonl|csv|tsv|txt|md|yaml|yml|toml|ini|cfg|html|css|"
    r"java|kt|cpp|cc|c|h|hpp|cs|go|rs|rb|php|swift|sql|xml|sh|bat|ps1|pdf|docx?|xlsx?|"
    r"pptx?|png|jpe?g|gif|svg|log|env|lock"
)
_PATH = re.compile(
    rf"(?:(?<![\w/\\])(?:[A-Za-z]:[\\/]|~/|\.{{1,2}}/|/)?(?:[\w.-]+[\\/])+[\w.-]+)"
    rf"|\b[\w-]+\.(?:{_FILE_EXT})\b"
)
_PLACEHOLDER = re.compile(
    r"\{\{[^{}\n]{1,60}\}\}"              # {{var}}
    r"|\{[A-Za-z_][\w.]{0,40}\}"          # {var}
    r"|</?[A-Za-z_][\w-]{0,40}(?:\s[^<>\n]{0,80})?/?>"  # <tag>, </tag>, <tag attr="x">
    r"|\[[A-Z][A-Z0-9_ ]{1,40}\]"         # [PLACEHOLDER]
    r"|\$\{?[A-Z_][A-Z0-9_]{1,40}\}?"     # $VAR / ${VAR}
)
_QUOTED = re.compile(
    r"\"[^\"\n]{1,200}\""
    r"|“[^”\n]{1,200}”"
    r"|(?<![\w'])'[^'\n]{1,120}'(?![\w'])"
)
_MONTHS = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
_DATE = re.compile(
    rf"\b\d{{4}}-\d{{1,2}}-\d{{1,2}}(?:[T ]\d{{1,2}}:\d{{2}}(?::\d{{2}})?)?\b"
    rf"|\b\d{{1,2}}[/.-]\d{{1,2}}[/.-]\d{{2,4}}\b"
    rf"|\b(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTHS})\.?(?:,?\s+\d{{4}})?\b"
    rf"|\b(?:{_MONTHS})\.?\s+\d{{4}}\b"
    rf"|\b\d{{1,2}}:\d{{2}}(?::\d{{2}})?\s?(?:[AaPp]\.?[Mm]\.?)?(?!\w)"
)
_UNITS = (
    r"%|percent|thousand|million|billion|trillion|crores?|lakhs?|bn|mn|[kKmMbB]|"
    r"bullet\s+points?|ms|s|sec|secs|seconds?|mins?|minutes?|h|hrs?|hours?|days?|"
    r"weeks?|months?|years?|yrs?|kb|mb|gb|tb|KB|MB|GB|TB|km|kg|g|mg|cm|mm|m|mi|lbs?|"
    r"°[CF]?|words?|tokens?|characters?|chars?|sentences?|paragraphs?|lines?|items?|"
    r"points?|bullets?|pages?|steps?|examples?|times|x"
)
_NUMBER = re.compile(
    rf"\bv?\d+(?:\.\d+){{2,3}}\b"                                  # versions 1.2.3
    rf"|(?<![\w.])[$€£₹¥]?\s?\d(?:[\d,]*\d)?(?:\.\d+)?(?:\s?(?:{_UNITS})(?![A-Za-z]))?"
    rf"|\b(?:{SMALL_NUMBER_WORDS})\s+(?:{_UNITS})\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(?:not|no|never|none|nothing|nobody|nowhere|nor|neither|without|cannot|"
    r"avoid(?:s|ing)?|except|excluding|unless|refrain|prohibited|forbidden|"
    r"disallowed|exclude|omit|don't|doesn't|didn't|won't|wouldn't|shouldn't|"
    r"mustn't|isn't|aren't|wasn't|weren't|can't|couldn't|haven't|hasn't)\b"
    r"|\b\w+n[’']t\b",
    re.IGNORECASE,
)
_CONSTRAINT = re.compile(
    r"\b(?:must|shall|should|required?|requires|mandatory|only|exactly|precisely|"
    r"at\s+(?:least|most)|no\s+(?:more|less|fewer|longer|shorter)\s+than|"
    r"(?:more|less|fewer|longer|shorter)\s+than|up\s+to|maximum|minimum|"
    r"limit(?:ed)?|always|ensure|make\s+sure|strictly|exceed|"
    r"(?:under|within|below|over|above|between)\s+\d)",
    re.IGNORECASE,
)
_FORMAT = re.compile(
    rf"\b(?:json|yaml|xml|csv|tsv|markdown|html|latex|table|tabular|bullet(?:ed)?\s+(?:points?|list)|"
    rf"bullets|numbered\s+list|headings?|code\s+block|format(?:ted|ting)?|schema|"
    rf"lowercase|uppercase|capitali[sz]e[ds]?|title\s+case|"
    rf"respond\s+(?:with|in|only|using)|reply\s+(?:with|in|only)|answer\s+(?:with|in|only)|"
    rf"return\s+(?:only|just)|output\s+(?:only|as|in|format)|in\s+the\s+form\s+of|template|"
    rf"(?:\d+|{SMALL_NUMBER_WORDS})[\s-]+(?:words?|sentences?|paragraphs?|bullets?|"
    rf"points?|lines?|items?|characters?|words?\s+or\s+less))\b",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY = re.compile(r"[.!?](?=[\s\"')\]]|$)|[;\n]")
_SENT_BOUNDARY = re.compile(r"[.!?](?=[\s\"')\]]|$)|\n")

_CAP_WORD = r"(?:[A-Z][A-Za-z0-9]*(?:[-.'’][A-Za-z0-9]+)*|[a-z]+[A-Z][A-Za-z0-9]*)"
_ENTITY = re.compile(
    rf"\b{_CAP_WORD}(?:\s+(?:(?:of|the|for|de|la|von|van|del|du)\s+)?{_CAP_WORD})*"
)


# ---------------------------------------------------------------- public API
def detect_protected_spans(
    text: str,
    kinds: Iterable[SpanKind] | None = None,
    use_spacy: bool | None = None,
) -> list[ProtectedSpan]:
    """Return de-duplicated protected spans in ``text`` sorted by position.

    ``kinds`` restricts which detectors run (default: all). ``use_spacy``: True =
    require spaCy NER, False = never use it, None = use it if available.
    """
    kinds = ALL_KINDS if kinds is None else frozenset(kinds)
    spans: list[ProtectedSpan] = []

    def add(start: int, end: int, kind: SpanKind) -> None:
        # trim surrounding whitespace so spans are tight
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append(ProtectedSpan(start, end, kind, text[start:end]))

    simple = (
        (SpanKind.CODE, _FENCE),
        (SpanKind.CODE, _INLINE_CODE),
        (SpanKind.URL, _URL),
        (SpanKind.EMAIL, _EMAIL),
        (SpanKind.PATH, _PATH),
        (SpanKind.PLACEHOLDER, _PLACEHOLDER),
        (SpanKind.QUOTED, _QUOTED),
        (SpanKind.DATE, _DATE),
        (SpanKind.NUMBER, _NUMBER),
    )
    for kind, pattern in simple:
        if kind in kinds:
            for m in pattern.finditer(text):
                if kind is SpanKind.NUMBER and not any(c.isdigit() for c in m.group()) \
                        and not re.match(SMALL_NUMBER_WORDS, m.group(), re.I):
                    continue
                add(m.start(), m.end(), kind)

    if SpanKind.NEGATION in kinds:
        for m in _NEGATION.finditer(text):
            s = _clause_start(text, m.start(), 60, _CLAUSE_BOUNDARY)
            e = _clause_end(text, m.end(), 150, _CLAUSE_BOUNDARY)
            add(s, e, SpanKind.NEGATION)

    for kind, pattern in ((SpanKind.CONSTRAINT, _CONSTRAINT), (SpanKind.FORMAT, _FORMAT)):
        if kind in kinds:
            for m in pattern.finditer(text):
                s = _clause_start(text, m.start(), 300, _SENT_BOUNDARY)
                e = _clause_end(text, m.end(), 300, _SENT_BOUNDARY)
                add(s, e, kind)

    if SpanKind.ENTITY in kinds:
        for s, e in _regex_entities(text):
            add(s, e, SpanKind.ENTITY)
        if use_spacy is not False:
            ner = _spacy_entities(text, required=bool(use_spacy))
            for s, e, kind in ner:
                if kind in kinds:
                    add(s, e, kind)

    return _dedupe(spans, text)


def spacy_available() -> bool:
    return _load_spacy() is not None


# ------------------------------------------------------------------- helpers
def _clause_start(text: str, pos: int, limit: int, boundary: re.Pattern) -> int:
    lo = max(0, pos - limit)
    last = None
    for m in boundary.finditer(text, lo, pos):
        last = m.end()
    if last is None:
        if lo == 0:
            last = 0
        else:  # no boundary nearby: start at the next word after ``lo``
            nxt = re.compile(r"\s").search(text, lo, pos)
            last = nxt.end() if nxt else pos
    while last < pos and text[last].isspace():
        last += 1
    # skip list markers like "- " or "2. "
    m = re.compile(r"(?:[-*•]|\d+[.)])\s+").match(text, last, pos)
    return m.end() if m else last


def _clause_end(text: str, pos: int, limit: int, boundary: re.Pattern) -> int:
    hi = min(len(text), pos + limit)
    m = boundary.search(text, pos, hi)
    if m:
        ch = text[m.start()]
        return m.start() + 1 if ch in ".!?;" else m.start()
    if hi == len(text):
        return hi
    nxt = re.compile(r"\s").search(text, hi)   # don't cut through a word
    return nxt.start() if nxt else len(text)


def _is_sentence_initial(text: str, start: int) -> bool:
    prefix = text[:start].rstrip(" \t")
    if not prefix or prefix[-1] in ".!?:\n\"'(“[":
        return True
    return re.search(r"(?:^|\n)\s*(?:[-*•>#]+|\d+[.)])$", prefix) is not None


def _regex_entities(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for m in _ENTITY.finditer(text):
        start, end = m.start(), m.end()
        words = list(re.finditer(r"\S+", m.group()))
        # strip leading stopwords / sentence-initial imperative verbs
        while words:
            w = words[0].group()
            sentence_initial = _is_sentence_initial(text, start + words[0].start())
            if w in ENTITY_STOPWORDS or (sentence_initial and w.lower() in IMPERATIVE_VERBS):
                words.pop(0)
            else:
                break
        while words and words[-1].group() in ENTITY_STOPWORDS:
            words.pop()
        if not words:
            continue
        s = start + words[0].start()
        e = start + words[-1].end()
        span = text[s:e]
        if len(words) == 1:
            w = span
            distinctive = (
                sum(c.isupper() for c in w) >= 2 or any(c.isdigit() for c in w) or w[0].islower()
            )
            if len(w) < 2 or (_is_sentence_initial(text, s) and not distinctive and len(w) < 4):
                continue
        out.append((s, e))
    return out


@lru_cache(maxsize=1)
def _load_spacy():
    try:
        import spacy  # type: ignore

        for name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
            try:
                return spacy.load(name, disable=["parser", "lemmatizer", "tagger"])
            except OSError:
                continue
    except ImportError:
        pass
    return None


_SPACY_ENTITY = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW",
                 "NORP", "FAC", "LANGUAGE"}
_SPACY_NUMBER = {"MONEY", "QUANTITY", "PERCENT", "CARDINAL", "ORDINAL"}


def _spacy_entities(text: str, required: bool = False) -> list[tuple[int, int, SpanKind]]:
    nlp = _load_spacy()
    if nlp is None:
        if required:
            raise RuntimeError(
                "use_spacy=True but spaCy / an English model is not installed. "
                "Run: pip install spacy && python -m spacy download en_core_web_sm"
            )
        return []
    out = []
    for ent in nlp(text).ents:
        if ent.label_ in _SPACY_ENTITY:
            out.append((ent.start_char, ent.end_char, SpanKind.ENTITY))
        elif ent.label_ in _SPACY_NUMBER:
            out.append((ent.start_char, ent.end_char, SpanKind.NUMBER))
        elif ent.label_ in {"DATE", "TIME"}:
            out.append((ent.start_char, ent.end_char, SpanKind.DATE))
    return out


def _dedupe(spans: list[ProtectedSpan], text: str) -> list[ProtectedSpan]:
    """Merge overlapping spans of the same kind and drop spans nested in atomic spans."""
    by_kind: dict[SpanKind, list[ProtectedSpan]] = {}
    for sp in spans:
        by_kind.setdefault(sp.kind, []).append(sp)

    merged: list[ProtectedSpan] = []
    for kind, items in by_kind.items():
        items.sort()
        cur_s, cur_e = items[0].start, items[0].end
        for sp in items[1:]:
            if sp.start < cur_e:
                cur_e = max(cur_e, sp.end)
            else:
                merged.append(ProtectedSpan(cur_s, cur_e, kind, text[cur_s:cur_e]))
                cur_s, cur_e = sp.start, sp.end
        merged.append(ProtectedSpan(cur_s, cur_e, kind, text[cur_s:cur_e]))

    atomic = [sp for sp in merged if sp.kind in ATOMIC_KINDS]
    result = []
    for sp in merged:
        nested = any(
            a is not sp and a.kind is not sp.kind and a.start <= sp.start and sp.end <= a.end
            for a in atomic
        )
        if not nested:
            result.append(sp)
    return sorted(result)

