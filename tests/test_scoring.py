"""Scoring tests — deterministic penalties, grade bands, breakdown sanity."""

from __future__ import annotations

import pandas as pd

from app.anomaly import detect_anomalies
from app.profiler import profile_dataset
from app.quality import detect_quality_issues
from app.scoring import compute_quality_score, grade_for


def _score(df: pd.DataFrame):
    """Run the full analysis exactly as the pipeline does, then score it."""
    profile = profile_dataset(df)
    issues = detect_quality_issues(df, profile)
    anomalies = detect_anomalies(df, profile)
    return compute_quality_score(df, profile, issues, anomalies)


def test_clean_sample_scores_high(clean_df):
    result = _score(clean_df)
    assert result.score >= 90
    assert result.grade == "A"


def test_dirty_sample_scores_far_lower(dirty_df):
    result = _score(dirty_df)
    assert result.score <= 70
    factors = {f.factor for f in result.breakdown}
    assert factors == {"completeness", "uniqueness", "validity", "consistency", "anomalies"}


def test_clean_outranks_dirty(clean_df, dirty_df):
    assert _score(clean_df).score > _score(dirty_df).score


def test_grade_bands():
    assert grade_for(100) == "A"
    assert grade_for(90) == "A"
    assert grade_for(89) == "B"
    assert grade_for(75) == "B"
    assert grade_for(74) == "C"
    assert grade_for(60) == "C"
    assert grade_for(59) == "D"
    assert grade_for(40) == "D"
    assert grade_for(39) == "F"
    assert grade_for(0) == "F"


def test_score_is_bounded_zero_to_hundred(clean_df, dirty_df):
    for df in (clean_df, dirty_df):
        score = _score(df).score
        assert 0 <= score <= 100
