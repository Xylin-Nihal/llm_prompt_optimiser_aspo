"""Module 1 - Prompt Segmenter.

Splits a prompt (plain string or chat ``messages``) into functional segments -
system / instruction / context / examples / question / format / code / data /
history / tool output - and attaches protected spans to each segment.

Why it matters for ASPO: every later stage is *segment-aware*. The budget
controller compresses context hard but leaves the question and format spec alone,
the validator checks that the question and the protected spans survived, and the
reconstruction is exact (``segment(p).to_text() == p``).

Signals used, strongest first:
1. explicit structure - ``` fences, XML-style tags (<context>...</context>),
   markdown headers and section labels ("Context:", "### Question");
2. few-shot pair patterns (Q:/A:, Input:/Output:, Text:/Label:);
3. content heuristics - imperative verbs, question marks, persona phrases,
   JSON/code shape, length;
4. position - the final unit of the prompt is the most likely question.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any

from .lexicon import (
    FORMAT_VERBS,
    IMPERATIVE_VERBS,
    PAIR_INPUT_LABELS,
    PAIR_OUTPUT_LABELS,
    PERSONA_PREFIXES,
    POLITE_QUESTION_REQUEST,
    QUESTION_WORDS,
    REQUEST_PREFIXES,
    SECTION_KEYWORDS,
)
from .protected import detect_protected_spans
from .types import (
    ProtectedSpan, Segment, SegmentedPrompt, SegmentRole, Source, SpanKind, Tier, merge_intervals,
    span_tier,
)

R = SegmentRole

# Which protected-span detectors run for each role. Constraint/format sentence
# detection only makes sense in instruction-like text; inside documents words
# like "should" or "table" are ordinary content.
_CONTENT_KINDS = frozenset(SpanKind) - {SpanKind.CONSTRAINT, SpanKind.FORMAT}
_ROLE_KINDS: dict[SegmentRole, frozenset[SpanKind]] = {
    R.SYSTEM: frozenset(SpanKind),
    R.INSTRUCTION: frozenset(SpanKind),
    R.QUESTION: frozenset(SpanKind),
    R.FORMAT: frozenset(SpanKind),
    R.OTHER: frozenset(SpanKind),
    R.EXAMPLE: _CONTENT_KINDS,
    R.CONTEXT: _CONTENT_KINDS,
    R.HISTORY: _CONTENT_KINDS,
    R.TOOL_OUTPUT: _CONTENT_KINDS,
    R.DATA: _CONTENT_KINDS,
}


@dataclass
class SegmenterConfig:
    long_context_words: int = 60       # unlabelled paragraphs this long are CONTEXT
    short_question_words: int = 60     # longer "?" paragraphs are context + question
    split_trailing_question: bool = True
    split_min_context_words: int = 20  # lead-in needed before a trailing "?" is split off
    use_spacy: bool | None = None      # None = auto-detect
    detect_protected: bool = True


# ------------------------------------------------------------------ patterns
_FENCE = re.compile(r"(?ms)^[ \t]*(```|~~~)[^\n]*\n.*?(?:^[ \t]*\1[ \t]*$|\Z)")
_TAG_BLOCK = re.compile(
    r"(?ms)^[ \t]*<(?P<tag>[A-Za-z_][\w-]*)(?:\s[^<>]*)?>(?P<body>.*?)</(?P=tag)\s*>[ \t]*$"
)
_PARA_SEP = re.compile(r"\n[ \t]*\n\s*")
_MD_HEADER = re.compile(r"^[ \t]*#{1,6}[ \t]+(?P<label>.+?)[ \t]*:?[ \t]*#*[ \t]*$")
_LABEL = re.compile(
    r"^[ \t]*(?:[-*][ \t]+)?(?:\*\*|__)?\[?(?P<label>[A-Za-z][A-Za-z0-9 _/()&-]{0,40}?)\]?"
    r"[ \t]*(?:\*\*|__)?[ \t]*:[ \t]*(?:\*\*|__)?[ \t]*(?P<rest>.*)$"
)
_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=\S)|\n+")
_PLEASANTRY = re.compile(r"(?i)^(?:thanks|thank you|thx|ty|cheers|regards|appreciate it)\b")
_LIST_MARKER = re.compile(r"^\s*(?:[-*•>]+|\d+[.)]|\(?[a-z]\))\s+")


@dataclass
class _Header:
    label: str
    role: SegmentRole | None
    rest_offset: int          # offset of content after the label within the line
    has_rest: bool
    markdown: bool


@dataclass
class _Unit:
    start: int
    end: int
    role: SegmentRole | None = None
    header: str | None = None
    body_start: int = 0        # absolute offset where content begins
    explicit: bool = False     # role came from explicit structure
    kind: str = "para"         # para | fence | tag | cue
    meta: dict[str, Any] = field(default_factory=dict)


# ================================================================ public API
class Segmenter:
    def __init__(self, config: SegmenterConfig | None = None):
        self.config = config or SegmenterConfig()

    # -------------------------------------------------------------- entry points
    def segment(self, prompt: str | list[dict[str, Any]]) -> SegmentedPrompt:
        if isinstance(prompt, str):
            return self.segment_text(prompt)
        return self.segment_messages(prompt)

    def segment_text(self, text: str) -> SegmentedPrompt:
        src = Source(text=text)
        units = self._segment_source(text, final=True)
        segments = self._build_segments([(0, src, units)])
        return SegmentedPrompt(sources=[src], segments=segments, messages=None)

    def segment_messages(self, messages: list[dict[str, Any]]) -> SegmentedPrompt:
        sources: list[Source] = []
        for mi, msg in enumerate(messages):
            role = str(msg.get("role", "user"))
            content = msg.get("content")
            if isinstance(content, str):
                sources.append(Source(content, mi, None, role))
            elif isinstance(content, list):
                for pi, part in enumerate(content):
                    if isinstance(part, dict) and part.get("type") == "text" \
                            and isinstance(part.get("text"), str):
                        sources.append(Source(part["text"], mi, pi, role))
            # None / non-text content (images, tool calls) is passed through untouched

        last_user = max(
            (mi for mi, m in enumerate(messages) if m.get("role") == "user"), default=None
        )
        last_user_src = max(
            (i for i, s in enumerate(sources) if s.message_index == last_user), default=None
        )

        batches = []
        for si, src in enumerate(sources):
            mrole = src.message_role
            if mrole in ("tool", "function", "ipython"):
                units = [self._whole_unit(src.text, R.TOOL_OUTPUT)] if src.text.strip() else []
            else:
                units = self._segment_source(src.text, final=(si == last_user_src))
                if mrole in ("system", "developer"):
                    for u in units:
                        if not u.explicit and u.role in (R.INSTRUCTION, R.OTHER, R.QUESTION):
                            u.role = R.SYSTEM
                elif mrole == "assistant" or (mrole == "user" and src.message_index != last_user):
                    for u in units:
                        if u.role is not R.CODE:
                            u.meta["inner_role"] = u.role.value if u.role else None
                            u.role = R.HISTORY
            batches.append((si, src, units))
        segments = self._build_segments(batches)
        return SegmentedPrompt(sources=sources, segments=segments, messages=messages)

    # ------------------------------------------------------------ segmentation
    def _segment_source(self, text: str, final: bool) -> list[_Unit]:
        if not text.strip():
            return []
        raw = self._raw_units(text)
        units = self._resolve_roles(text, raw, final)
        return units

    def _whole_unit(self, text: str, role: SegmentRole) -> _Unit:
        s = len(text) - len(text.lstrip())
        e = len(text.rstrip())
        return _Unit(s, e, role=role, body_start=s, explicit=True, kind="whole")

    def _raw_units(self, text: str) -> list[_Unit]:
        """Split into units and apply explicit structure (headers, labels, tags)."""
        blocks = self._find_blocks(text)
        units: list[_Unit] = []
        cursor = 0
        for b in blocks:
            units.extend(self._paragraph_units(text, cursor, b.start))
            units.append(b)
            cursor = b.end
        units.extend(self._paragraph_units(text, cursor, len(text)))

        # Attach header-only units to the next unit; carry section roles forward.
        out: list[_Unit] = []
        pending: _Unit | None = None
        section_role: SegmentRole | None = None
        for u in units:
            if u.kind == "header":
                if pending is not None:          # two headers in a row: keep the first as-is
                    out.append(pending)
                pending = u
                section_role = u.role
                continue
            if u.header is not None and u.kind == "para":   # inline "Label: content"
                section_role = u.role
            if pending is not None:
                u.start = pending.start
                u.header = pending.header
                if pending.role is not None and not (u.explicit and u.kind in ("tag", "fence")):
                    u.role = pending.role
                    u.explicit = True
                pending = None
            elif u.role is None and section_role is not None and u.kind == "para":
                u.role = section_role
                u.meta["inherited"] = True
            out.append(u)
        if pending is not None:   # trailing header with nothing after it: an answer cue
            pending.kind = "cue"
            pending.role = R.FORMAT
            pending.explicit = True
            pending.meta["cue"] = True
            out.append(pending)
        return out

    def _find_blocks(self, text: str) -> list[_Unit]:
        cands: list[_Unit] = []
        for m in _FENCE.finditer(text):
            s, e = _trim(text, m.start(), m.end())
            cands.append(_Unit(s, e, role=R.CODE, body_start=s, explicit=True, kind="fence"))
        for m in _TAG_BLOCK.finditer(text):
            s, e = _trim(text, m.start(), m.end())
            tag = m.group("tag")
            role = _label_role(tag)
            body_s, body_e = m.start("body"), m.end("body")
            if role is None:
                role = self._classify(text[body_s:body_e], is_last=False)
                explicit = False
            else:
                explicit = True
            cands.append(_Unit(s, e, role=role, header=tag, body_start=body_s,
                               explicit=explicit, kind="tag"))
        cands.sort(key=lambda u: (u.start, -(u.end - u.start)))
        blocks: list[_Unit] = []
        for c in cands:
            if not blocks or c.start >= blocks[-1].end:
                blocks.append(c)
        return blocks

    def _paragraph_units(self, text: str, lo: int, hi: int) -> list[_Unit]:
        units: list[_Unit] = []
        for ps, pe in _paragraphs(text, lo, hi):
            lines = _lines(text, ps, pe)
            if _is_example_pair(text, lines):
                units.append(self._pair_unit(text, ps, pe, lines))
                continue
            cur: _Unit | None = None
            for ls, le in lines:
                h = _parse_header(text[ls:le])
                if h is not None:
                    if cur is not None:
                        units.append(cur)
                    if h.has_rest:
                        cur = _Unit(ls, le, role=h.role, header=h.label,
                                    body_start=ls + h.rest_offset, explicit=h.role is not None)
                    else:
                        units.append(_Unit(ls, le, role=h.role, header=h.label,
                                           body_start=le, explicit=True, kind="header"))
                        cur = None
                else:
                    if cur is None:
                        cur = _Unit(ls, le, body_start=ls)
                    else:
                        cur.end = le
            if cur is not None:
                units.append(cur)
        return units

    def _pair_unit(self, text: str, ps: int, pe: int, lines: list[tuple[int, int]]) -> _Unit:
        # Is the final output label empty? Then this is the open query, not a demo.
        last_out = None
        for i, (ls, le) in enumerate(lines):
            m = _LABEL.match(text[ls:le])
            if m and _norm(m.group("label")) in PAIR_OUTPUT_LABELS:
                last_out = (i, m)
        open_query = False
        if last_out is not None:
            i, m = last_out
            open_query = not m.group("rest").strip() and i == len(lines) - 1
        if open_query:
            return _Unit(ps, pe, role=R.QUESTION, body_start=ps, explicit=True,
                         meta={"few_shot_query": True})
        return _Unit(ps, pe, role=R.EXAMPLE, body_start=ps, explicit=True,
                     meta={"few_shot_pair": True})

    def _resolve_roles(self, text: str, units: list[_Unit], final: bool) -> list[_Unit]:
        # index of the last "content" unit - the most likely place for the question
        last_idx = max((i for i, u in enumerate(units) if u.kind not in ("cue", "fence")),
                       default=None)
        out: list[_Unit] = []
        for i, u in enumerate(units):
            body = text[u.body_start:u.end]
            is_last = final and i == last_idx
            if u.role is None:
                u.role = self._classify(body, is_last)
            elif u.meta.get("inherited"):
                guess = self._classify(body, is_last)
                # strong content signals override an inherited section role
                if guess in (R.FORMAT, R.CODE, R.DATA) and u.role not in (R.CONTEXT, R.DATA):
                    u.role = guess
                elif guess is R.QUESTION and u.role is R.INSTRUCTION and is_last:
                    u.role = guess
            if is_last and self.config.split_trailing_question and not u.explicit \
                    and u.role in (R.CONTEXT, R.OTHER, R.INSTRUCTION, R.QUESTION) \
                    and u.kind == "para":
                split = self._split_trailing_request(text, u)
                if split:
                    out.extend(split)
                    continue
            out.append(u)
        return out

    def _split_trailing_request(self, text: str, u: _Unit) -> list[_Unit] | None:
        """Peel the trailing request (question / imperative / polite ask) off a long
        final paragraph, e.g. "<story about my situation> Could you write X? Keep it short."
        -> CONTEXT + INSTRUCTION."""
        body = text[u.body_start:u.end]
        starts = [0] + [m.end() for m in _SENT_END.finditer(body)]
        if len(starts) < 2:
            return None
        sentences = [body[s:e].strip() for s, e in zip(starts, starts[1:] + [len(body)])]
        cut_idx = None
        for i in range(len(sentences) - 1, 0, -1):
            sent = sentences[i]
            if _PLEASANTRY.match(sent):
                continue
            if self._classify(sent, is_last=True) in (R.QUESTION, R.INSTRUCTION, R.FORMAT):
                cut_idx = i
            else:
                break
        if cut_idx is None:
            return None
        cut = u.body_start + starts[cut_idx]
        head = _Unit(u.start, _rtrim(text, u.start, cut), header=u.header,
                     body_start=u.body_start, meta=dict(u.meta))
        head_body = text[head.body_start:head.end]
        if len(head_body.split()) < self.config.split_min_context_words:
            return None
        head.role = self._classify(head_body, is_last=False)
        if head.role in (R.OTHER, R.QUESTION):
            head.role = R.CONTEXT
        tail_role = self._classify(sentences[cut_idx], is_last=True)
        tail = _Unit(cut, u.end, role=tail_role, body_start=cut, meta={"split_from": "trailing"})
        return [head, tail]

    # ---------------------------------------------------------- classification
    def _classify(self, body: str, is_last: bool) -> SegmentRole:
        t = body.strip()
        if not t:
            return R.OTHER
        words = len(t.split())
        if _looks_like_data(t):
            return R.DATA
        if _looks_like_code(t):
            return R.CODE
        low = _LIST_MARKER.sub("", t, count=1).lower()
        if re.match(r"(?:examples?|e\.g\.|for example|for instance|sample)\b", low):
            return R.EXAMPLE
        if low.startswith(PERSONA_PREFIXES):
            return R.SYSTEM

        first = _first_word(low)
        format_hit = re.search(
            r"\b(?:json|yaml|xml|csv|markdown|table|bullet|numbered list|format|"
            r"schema|\d+\s+words?|one (?:word|sentence|paragraph))\b", low
        )
        if first in FORMAT_VERBS or re.match(
            r"(?:the |your )?(?:output|answer|response|reply|result)s?\s+(?:should|must|"
            r"will|needs? to|has to|is to)\b", low
        ):
            return R.FORMAT

        ends_q = t.rstrip("\"')] ").endswith("?")
        if ends_q and low.startswith(POLITE_QUESTION_REQUEST):
            return R.INSTRUCTION
        if ends_q and words <= self.config.short_question_words:
            return R.QUESTION
        if first in QUESTION_WORDS and words <= 25 and is_last and "\n" not in t:
            return R.QUESTION   # question missing its "?"

        if first in IMPERATIVE_VERBS or low.startswith(REQUEST_PREFIXES):
            return R.INSTRUCTION
        if words < self.config.long_context_words:
            # a later sentence may carry the request: "Mostly food. Give me a plan..."
            for sent in _SENT_END.split(low)[1:]:
                sent = _LIST_MARKER.sub("", sent.strip(), count=1)
                if _first_word(sent) in IMPERATIVE_VERBS - FORMAT_VERBS \
                        or sent.startswith(REQUEST_PREFIXES + POLITE_QUESTION_REQUEST):
                    return R.INSTRUCTION
        if format_hit and words <= 30:
            return R.FORMAT
        if words >= self.config.long_context_words:
            return R.CONTEXT
        if ends_q:
            return R.CONTEXT if not is_last else R.QUESTION
        return R.CONTEXT if words >= 12 else R.OTHER

    # ------------------------------------------------------------ build output
    def _build_segments(self, batches) -> list[Segment]:
        segments: list[Segment] = []
        for si, src, units in batches:
            for u in units:
                seg_text = src.text[u.start:u.end]
                seg = Segment(
                    id=len(segments),
                    role=u.role or R.OTHER,
                    text=seg_text,
                    start=u.start,
                    end=u.end,
                    source_index=si,
                    message_index=src.message_index,
                    message_role=src.message_role,
                    header=u.header,
                    body_start=max(0, min(len(seg_text), u.body_start - u.start)),
                    meta=dict(u.meta),
                )
                if u.kind == "cue":
                    seg.body_start = len(seg_text)
                seg.protected = self._protect(seg)
                segments.append(seg)
        return segments

    def _protect(self, seg: Segment) -> list[ProtectedSpan]:
        return _assign_tiers(self._detect(seg), seg.role)

    def _detect(self, seg: Segment) -> list[ProtectedSpan]:
        if not self.config.detect_protected:
            return []
        if seg.role is R.CODE or seg.meta.get("cue"):
            kind = SpanKind.CODE if seg.role is R.CODE else SpanKind.FORMAT
            return [ProtectedSpan(0, len(seg.text), kind, seg.text)]
        kinds = _ROLE_KINDS.get(seg.role, frozenset(SpanKind))
        spans = detect_protected_spans(seg.text, kinds=kinds, use_spacy=self.config.use_spacy)
        if seg.role is R.QUESTION and seg.meta.get("few_shot_query"):
            # keep the trailing answer cue (e.g. "A:") of an open few-shot query
            m = re.search(r"\n[^\n]*:\s*$", seg.text)
            if m:
                s = m.start() + 1
                spans.append(ProtectedSpan(s, len(seg.text.rstrip()), SpanKind.FORMAT,
                                           seg.text[s:len(seg.text.rstrip())]))
                spans.sort()
        return spans


# ======================================================== module-level helpers
def _assign_tiers(spans: list[ProtectedSpan], role: SegmentRole) -> list[ProtectedSpan]:
    """Set each span's tier from (kind, role). A soft span that overlaps a hard span
    becomes hard: removing it as a unit would cut into the hard span."""
    tiered = [replace(p, tier=span_tier(p.kind, role)) for p in spans]
    hard = merge_intervals((p.start, p.end) for p in tiered if p.tier is Tier.HARD)
    return [
        replace(p, tier=Tier.HARD)
        if p.tier is Tier.SOFT and any(s < p.end and p.start < e for s, e in hard) else p
        for p in tiered
    ]


_DEFAULT = Segmenter()


def segment(prompt: str | list[dict[str, Any]], config: SegmenterConfig | None = None
            ) -> SegmentedPrompt:
    """Segment a prompt string or a list of chat messages."""
    return (Segmenter(config) if config else _DEFAULT).segment(prompt)


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    return s, _rtrim(text, s, e)


def _rtrim(text: str, s: int, e: int) -> int:
    while e > s and text[e - 1].isspace():
        e -= 1
    return e


def _paragraphs(text: str, lo: int, hi: int):
    cursor = lo
    for m in _PARA_SEP.finditer(text, lo, hi):
        s, e = _trim(text, cursor, m.start())
        if e > s:
            yield s, e
        cursor = m.end()
    s, e = _trim(text, cursor, hi)
    if e > s:
        yield s, e


def _lines(text: str, ps: int, pe: int) -> list[tuple[int, int]]:
    out = []
    pos = ps
    while pos < pe:
        nl = text.find("\n", pos, pe)
        end = pe if nl == -1 else nl
        s, e = _trim(text, pos, end)
        if e > s:
            out.append((pos if text[pos:s].strip() == "" else s, e))
        pos = end + 1
    return out


def _norm(label: str) -> str:
    label = re.sub(r"[\d#_()&/-]+", " ", label.lower())
    return " ".join(label.split())


def _label_role(label: str) -> SegmentRole | None:
    norm = _norm(label)
    if not norm:
        return None
    if norm in SECTION_KEYWORDS:
        return SegmentRole(SECTION_KEYWORDS[norm])
    words = norm.split()
    if len(words) > 4:
        return None
    # "output format" -> format, "user question" -> question, "few shot examples" -> example
    for w in reversed(words):
        if w in SECTION_KEYWORDS and len(w) > 1:
            return SegmentRole(SECTION_KEYWORDS[w])
    return None


def _parse_header(line: str) -> _Header | None:
    m = _MD_HEADER.match(line)
    if m:
        label = m.group("label").strip("*_ ")
        return _Header(label, _label_role(label), len(line), False, True)
    m = _LABEL.match(line)
    if m:
        label = m.group("label").strip()
        role = _label_role(label)
        if role is None:
            return None        # "Note: ..." etc. is just prose
        if len(_norm(label)) == 1 and role is not R.QUESTION:
            return None        # bare "A:" only matters inside few-shot pairs
        rest = m.group("rest")
        return _Header(label, role, m.start("rest"), bool(rest.strip()), False)
    return None


def _is_example_pair(text: str, lines: list[tuple[int, int]]) -> bool:
    if len(lines) < 2:
        return False
    first = _LABEL.match(text[lines[0][0]:lines[0][1]])
    if not first or _norm(first.group("label")) not in PAIR_INPUT_LABELS:
        return False
    for ls, le in lines[1:]:
        m = _LABEL.match(text[ls:le])
        if m and _norm(m.group("label")) in PAIR_OUTPUT_LABELS:
            return True
    return False


def _first_word(low: str) -> str:
    low = re.sub(r"^(?:please|kindly|now|then|also|so|ok(?:ay)?|and)[,\s]+", "", low)
    m = re.match(r"[a-z']+", low)
    return m.group() if m else ""


def _looks_like_data(t: str) -> bool:
    if t[0] in "{[" and t[-1] in "}]" and len(t) > 2:
        try:
            json.loads(t)
            return True
        except ValueError:
            pass
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if len(lines) >= 3:
        for sep in (",", "\t", "|"):
            counts = [ln.count(sep) for ln in lines]
            if min(counts) >= 2 and max(counts) - min(counts) <= 1:
                return True
    return False


_CODE_LINE = re.compile(
    r"^\s*(?:def |class |import |from \S+ import|return\b|if .*:$|for .*:$|while .*:$|"
    r"#include|public |private |function\b|const |let |var |print\(|console\.|"
    r"SELECT\b|INSERT\b|UPDATE\b|@\w+)"
    r"|[;{}]\s*$|^\s*}|=>|==|!=|\w+\(.*\)\s*;?\s*$"
)


def _looks_like_code(t: str) -> bool:
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    hits = sum(bool(_CODE_LINE.search(ln)) for ln in lines)
    indented = sum(ln.startswith(("    ", "\t")) for ln in lines)
    return hits / len(lines) >= 0.5 or (indented == len(lines) and hits >= 1)


# ================================================================== CLI demo
def _describe(sp: SegmentedPrompt) -> str:
    rows = []
    for s in sp.segments:
        preview = s.text.replace("\n", " ⏎ ")
        preview = preview if len(preview) <= 70 else preview[:67] + "..."
        where = f"m{s.message_index}" if s.message_index is not None else ""
        hard = ", ".join(sorted({p.kind.value for p in s.hard_spans}))
        soft = ", ".join(sorted({p.kind.value for p in s.soft_spans}))
        rows.append(f"[{s.id:>2}] {where:<3} {s.role.value:<11} {s.word_count:>4}w "
                    f"prot={s.protected_ratio:>4.0%}  {preview}\n"
                    f"{'':<24}hard: {hard or '-'}  |  soft: {soft or '-'}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="ASPO Module 1: segment a prompt.")
    ap.add_argument("file", nargs="?", help="prompt file (.txt, or .json chat messages); stdin if omitted")
    ap.add_argument("--json", action="store_true", help="print full JSON output")
    args = ap.parse_args(argv)

    raw = open(args.file, encoding="utf-8").read() if args.file else sys.stdin.read()
    prompt: Any = raw
    if args.file and args.file.endswith(".json"):
        prompt = json.loads(raw)
    sp = segment(prompt)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(sp.to_dict(), indent=2, ensure_ascii=False) if args.json else _describe(sp))


if __name__ == "__main__":
    main()
