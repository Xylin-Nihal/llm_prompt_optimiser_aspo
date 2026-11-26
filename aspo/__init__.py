"""ASPO - Adaptive Semantic Prompt Optimization Framework."""

from .analyzer import AnalyzerConfig, IntentAnalyzer, analyze
from .intake import EmptyPromptError, IntakeResult, check_intake
from .protected import detect_protected_spans
from .segmenter import Segmenter, SegmenterConfig, segment
from .tokenizers import (
    ModelPrice,
    TokenizerInfo,
    context_window,
    count_tokens,
    price_per_token,
    savings,
    tokenizer_info,
)
from .types import (
    IntentAnalysis,
    ProtectedSpan,
    RemovalUnit,
    Segment,
    SegmentedPrompt,
    SegmentRole,
    SpanKind,
    TaskType,
    Tier,
    span_tier,
)
from .validator import CheckResult, ValidationResult, Validator, ValidatorConfig, tau

__all__ = [
    "Validator",
    "ValidatorConfig",
    "ValidationResult",
    "CheckResult",
    "tau",
    "check_intake",
    "IntakeResult",
    "EmptyPromptError",
    "context_window",
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
    "Tier",
    "span_tier",
    "RemovalUnit",
]
__version__ = "0.1.0"
