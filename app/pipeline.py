"""Pipeline orchestrator: profile → issues → anomalies → score → explanation.

Pure domain flow — HTTP, decoding and validation live in the routes layer.
"""

from __future__ import annotations

import pandas as pd

from .ai.provider import explain_safely, get_provider
from .anomaly import detect_anomalies
from .config import Settings, load_settings
from .profiler import profile_dataset
from .quality import detect_quality_issues
from .schemas import AnalysisResponse, Explanation, FileMeta
from .scoring import compute_quality_score


def analyze_dataframe(df: pd.DataFrame, file_meta: FileMeta, explain: bool) -> AnalysisResponse:
    """Run the full analysis pipeline over an already-parsed DataFrame."""
    profile = profile_dataset(df)
    issues = detect_quality_issues(df, profile)
    anomalies = detect_anomalies(df, profile)
    score = compute_quality_score(df, profile, issues, anomalies)

    explanation: Explanation | None = None
    if explain:
        settings: Settings = load_settings()
        provider = get_provider(settings)
        analysis = AnalysisResponse(
            file=file_meta,
            profile=profile,
            issues=issues,
            anomalies=anomalies,
            quality_score=score,
        )
        provider_label, text = explain_safely(provider, analysis.model_dump())
        explanation = Explanation(provider=provider_label, text=text)

    return AnalysisResponse(
        file=file_meta,
        profile=profile,
        issues=issues,
        anomalies=anomalies,
        quality_score=score,
        explanation=explanation,
    )
