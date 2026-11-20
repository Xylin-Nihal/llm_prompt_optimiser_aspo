"""Core data types shared by every ASPO module.

Offsets convention
------------------
* ``Segment.start`` / ``Segment.end`` are character offsets into the *source*
  text the segment came from (``SegmentedPrompt.sources[seg.source_index].text``).
* ``ProtectedSpan.start`` / ``.end`` are offsets **relative to the segment's own
  text**, so downstream optimizers can work on one segment at a time.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SegmentRole(str, Enum):
    """Functional role of a piece of a prompt."""

    SYSTEM = "system"            # persona / system-level instructions
    INSTRUCTION = "instruction"  # the task the model must perform
    CONTEXT = "context"          # documents, background, retrieved passages
    EXAMPLE = "example"          # few-shot demonstrations
    QUESTION = "question"        # the actual query
    FORMAT = "format"            # output-format spec or answer cue ("A:")
    CODE = "code"                # code blocks
    DATA = "data"                # JSON / CSV / tables
    HISTORY = "history"          # earlier conversation turns
    TOOL_OUTPUT = "tool_output"  # tool / function results in agent loops
    OTHER = "other"


class SpanKind(str, Enum):
    """Why a span must survive optimisation untouched."""

    CODE = "code"
    URL = "url"
    EMAIL = "email"
    PATH = "path"
    PLACEHOLDER = "placeholder"   # {var}, {{var}}, <tag>, [NAME], $VAR
    QUOTED = "quoted"             # literal strings in quotes
    DATE = "date"
    NUMBER = "number"             # numbers, units, money, percentages, versions
    ENTITY = "entity"             # named entities / proper nouns / acronyms
    NEGATION = "negation"         # clause containing a negation
    CONSTRAINT = "constraint"     # sentence with must/only/at most/...
    FORMAT = "format"             # sentence specifying output format


#: Kinds whose spans are indivisible tokens - anything nested inside them is redundant.
ATOMIC_KINDS = frozenset(
    {
        SpanKind.CODE,
        SpanKind.URL,
        SpanKind.EMAIL,
        SpanKind.PATH,
        SpanKind.PLACEHOLDER,
        SpanKind.QUOTED,
        SpanKind.DATE,
    }
)


@dataclass(frozen=True, order=True)
class ProtectedSpan:
    start: int
    end: int
    kind: SpanKind
    text: str = field(compare=False)

    def __len__(self) -> int:
        return self.end - self.start


@dataclass
class Segment:
    id: int
    role: SegmentRole
    text: str
    start: int
    end: int
    source_index: int = 0
    message_index: int | None = None
    message_role: str | None = None
    header: str | None = None          # section label that introduced this segment, if any
    body_start: int = 0                # offset in ``text`` where content starts (after header)
    protected: list[ProtectedSpan] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def body(self) -> str:
        return self.text[self.body_start:]

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def protected_intervals(self) -> list[tuple[int, int]]:
        """Merged, sorted (start, end) intervals covered by protected spans."""
        return merge_intervals((s.start, s.end) for s in self.protected)

    @property
    def protected_ratio(self) -> float:
        if not self.text:
            return 0.0
        covered = sum(e - s for s, e in self.protected_intervals())
        return covered / len(self.text)

    def is_protected(self, start: int, end: int | None = None) -> bool:
        """True if any character in [start, end) lies inside a protected span."""
        end = start + 1 if end is None else end
        return any(s < end and start < e for s, e in self.protected_intervals())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role.value,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "source_index": self.source_index,
            "message_index": self.message_index,
            "message_role": self.message_role,
            "header": self.header,
            "body_start": self.body_start,
            "protected": [
                {"start": p.start, "end": p.end, "kind": p.kind.value, "text": p.text}
                for p in self.protected
            ],
            "meta": self.meta,
        }


@dataclass
class Source:
    """One contiguous piece of input text (a whole prompt, or one chat message part)."""

    text: str
    message_index: int | None = None
    part_index: int | None = None   # index into a list-style message ``content``
    message_role: str | None = None


@dataclass
class SegmentedPrompt:
    sources: list[Source]
    segments: list[Segment]
    messages: list[dict[str, Any]] | None = None   # original chat messages, if chat input

    # ------------------------------------------------------------------ queries
    @property
    def is_chat(self) -> bool:
        return self.messages is not None

    def by_role(self, *roles: SegmentRole) -> list[Segment]:
        return [s for s in self.segments if s.role in roles]

    def role_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.segments:
            counts[s.role.value] = counts.get(s.role.value, 0) + 1
        return counts

    # ----------------------------------------------------------- reconstruction
    def reconstruct_sources(self, replacements: dict[int, str] | None = None) -> list[str]:
        """Rebuild every source text, optionally swapping segment texts.

        ``replacements`` maps segment id -> new text. An empty string drops the
        segment *and* the whitespace gap that followed it. With no replacements the
        output is byte-identical to the input.
        """
        replacements = replacements or {}
        out: list[str] = []
        for si, src in enumerate(self.sources):
            segs = sorted((s for s in self.segments if s.source_index == si), key=lambda s: s.start)
            items: list[list[str]] = []   # [gap_before, text]
            cursor = 0
            for seg in segs:
                items.append([src.text[cursor:seg.start], replacements.get(seg.id, seg.text)])
                cursor = seg.end
            tail = src.text[cursor:]
            kept = [it for it in items if it[1] != ""]
            if not items:
                out.append(src.text)
                continue
            if not kept:
                out.append("")
                continue
            # A dropped segment takes its preceding gap with it; if the first segment
            # was dropped, the first kept one inherits the original leading gap.
            if items[0][1] == "":
                kept[0][0] = items[0][0]
            out.append("".join(g + t for g, t in kept) + tail)
        return out

    def to_text(self, replacements: dict[int, str] | None = None) -> str:
        texts = self.reconstruct_sources(replacements)
        if self.is_chat:
            raise ValueError("Chat prompt: use to_messages() instead of to_text().")
        return texts[0] if texts else ""

    def to_messages(self, replacements: dict[int, str] | None = None) -> list[dict[str, Any]]:
        if not self.is_chat:
            raise ValueError("Not a chat prompt: use to_text().")
        texts = self.reconstruct_sources(replacements)
        messages = copy.deepcopy(self.messages)
        for src, new_text in zip(self.sources, texts):
            msg = messages[src.message_index]
            if src.part_index is None:
                msg["content"] = new_text
            else:
                msg["content"][src.part_index]["text"] = new_text
        return messages

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_chat": self.is_chat,
            "role_counts": self.role_counts(),
            "segments": [s.to_dict() for s in self.segments],
        }


def merge_intervals(intervals) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


# --------------------------------------------------------------------------- Module 2
class TaskType(str, Enum):
    """What the prompt asks the model to do (output of the intent analyzer)."""

    QA = "qa"                          # open / closed-book question answering
    RAG_QA = "rag_qa"                  # question answered from supplied context
    SUMMARIZATION = "summarization"
    CODE = "code"                      # write / explain / debug code
    REASONING = "reasoning"            # math, logic, multi-step problems
    EXTRACTION = "extraction"          # pull structured facts out of text
    CLASSIFICATION = "classification"  # label / categorise / sentiment
    TRANSFORMATION = "transformation"  # translate / rewrite / proofread (input must survive)
    CREATIVE = "creative"              # open-ended generation: stories, copy, plans, ideas
    CHITCHAT = "chitchat"              # greetings, small talk


@dataclass
class IntentAnalysis:
    task: TaskType
    confidence: float                       # probability of ``task``
    scores: dict[TaskType, float]           # probability per task type (sums to 1)
    complexity: float                       # 0 (trivial) .. 1 (very demanding)
    complexity_factors: dict[str, float]    # each factor in [0, 1]
    features: dict[str, float]              # raw feature vector (for a learned model later)
    signals: list[str] = field(default_factory=list)   # human-readable evidence

    @property
    def complexity_level(self) -> str:
        return "low" if self.complexity < 0.33 else "medium" if self.complexity < 0.66 else "high"

    def top(self, k: int = 3) -> list[tuple[TaskType, float]]:
        return sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)[:k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "confidence": round(self.confidence, 4),
            "scores": {t.value: round(p, 4) for t, p in self.top(len(self.scores))},
            "complexity": round(self.complexity, 4),
            "complexity_level": self.complexity_level,
            "complexity_factors": {k: round(v, 4) for k, v in self.complexity_factors.items()},
            "features": {k: round(v, 4) for k, v in self.features.items()},
            "signals": self.signals,
        }
