"""ASPO - Adaptive Semantic Prompt Optimization Framework."""

from .analyzer import AnalyzerConfig, IntentAnalyzer, analyze
from .protected import detect_protected_spans
from .segmenter import Segmenter, SegmenterConfig, segment
from .tokenizers import (
    ModelPrice,
    TokenizerInfo,
    count_tokens,
    price_per_token,
    savings,
    tokenizer_info,
)
from .types import (
    IntentAnalysis,
    ProtectedSpan,
    Segment,
    SegmentedPrompt,
    SegmentRole,
    SpanKind,
    TaskType,
)

__all__ = [
    "count_tokens",
    "tokenizer_info",
    "price_per_token",
    "savings",
    "TokenizerInfo",
    "ModelPrice",
    "Segmenter",
    "SegmenterConfig",
    "segment",
    "detect_protected_spans",
    "IntentAnalyzer",
    "AnalyzerConfig",
    "analyze",
    "IntentAnalysis",
    "TaskType",
    "Segment",
    "SegmentedPrompt",
    "SegmentRole",
    "SpanKind",
    "ProtectedSpan",
]
__version__ = "0.1.0"
