"""ASPO - Adaptive Semantic Prompt Optimization Framework."""

from .protected import detect_protected_spans
from .segmenter import Segmenter, SegmenterConfig, segment
from .types import ProtectedSpan, Segment, SegmentedPrompt, SegmentRole, SpanKind

__all__ = [
    "Segmenter",
    "SegmenterConfig",
    "segment",
    "detect_protected_spans",
    "Segment",
    "SegmentedPrompt",
    "SegmentRole",
    "SpanKind",
    "ProtectedSpan",
]
__version__ = "0.1.0"
