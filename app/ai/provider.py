"""Provider protocol, factory and safe-explanation wrapper.

Contract: a provider turns the *computed* analysis (plain dict) into a short
human-readable explanation. Providers never compute scores themselves — they
may only reference numbers already present in the analysis.
"""

from __future__ import annotations

from typing import Any, Protocol

from .mock import build_mock_explanation
from .openai_compat import OpenAICompatProvider


class ExplanationProvider(Protocol):
    name: str

    def explain(self, analysis: dict[str, Any]) -> str: ...


def get_provider(settings) -> ExplanationProvider:
    """Factory: pick the provider named by settings (mock is the default)."""
    if settings.ai_provider == "openai":
        return OpenAICompatProvider(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            timeout_secs=settings.openai_timeout_secs,
        )
    return MockExplanationProvider()


def explain_safely(provider: ExplanationProvider, analysis: dict[str, Any]) -> tuple[str, str]:
    """Call the provider; on ANY failure fall back to the deterministic template.

    The data-quality report must never fail because its explainer did.
    Returns (provider_label, explanation_text).
    """
    try:
        text = provider.explain(analysis).strip()
        if not text:
            raise ValueError("provider returned an empty explanation")
        return provider.name, text
    except Exception as exc:  # noqa: BLE001 — deliberate: degrade, never raise
        return f"mock (fallback: {type(exc).__name__})", build_mock_explanation(analysis)


class MockExplanationProvider:
    """Deterministic, offline explanation built from the computed analysis.

    Same input → byte-identical output, no network, no API key.
    """

    name = "mock"

    def explain(self, analysis: dict[str, Any]) -> str:
        return build_mock_explanation(analysis)
