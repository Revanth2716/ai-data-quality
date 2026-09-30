"""AI explanation providers (protocol, factory, deterministic fallback)."""

from .mock import build_mock_explanation
from .openai_compat import OpenAICompatProvider
from .provider import ExplanationProvider, explain_safely, get_provider

__all__ = [
    "ExplanationProvider",
    "OpenAICompatProvider",
    "build_mock_explanation",
    "explain_safely",
    "get_provider",
]
