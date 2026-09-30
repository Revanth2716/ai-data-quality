"""Quality-analyzer tests — one per issue type plus the severity ladder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.profiler import profile_dataset
from app.quality import detect_quality_issues, severity_for


def issues_of(df: pd.DataFrame):
    return detect_quality_issues(df, profile_dataset(df))


def by_type(issues, type_: str):
    return [issue for issue in issues if issue.type == type_]


def test_missing_values_detected_with_severity():
    df = pd.DataFrame({"a": [1, np.nan, np.nan, 4]})
    found = by_type(issues_of(df), "missing_values")
    assert len(found) == 1
    assert found[0].column == "a" and found[0].count == 2
    assert found[0].severity == "high"  # 50% of rows > 10% threshold


def test_duplicate_rows_detected_with_first_occurrence_index():
    df = pd.DataFrame({"a": [1, 2, 2, 3], "b": ["x", "y", "y", "x"]})
    found = by_type(issues_of(df), "duplicate_rows")
    assert found[0].count == 1
    assert found[0].row_indices == [1]


def test_inconsistent_types_detected_with_examples():
    df = pd.DataFrame({"v": ["1", "2", "3", "4", "oops"]})  # 4/5 >= 80% numeric
    found = by_type(issues_of(df), "inconsistent_types")
    assert found[0].column == "v"
    assert found[0].count == 1
    assert found[0].examples == ["oops"]
    assert found[0].severity == "high"


def test_sentinel_values_detected():
    df = pd.DataFrame({"v": ["N/A", "?", "unknown", "-", "ok"]})
    found = by_type(issues_of(df), "sentinel_values")
    assert found[0].count == 4
    assert found[0].column == "v"


def test_empty_and_constant_columns_detected():
    df = pd.DataFrame({"e": [np.nan, np.nan], "c": ["a", "a"]})
    issues = issues_of(df)
    assert by_type(issues, "empty_column")[0].column == "e"
    assert by_type(issues, "constant_column")[0].column == "c"


def test_inf_values_detected():
    df = pd.DataFrame({"v": [1.0, np.inf, 3.0]})
    found = by_type(issues_of(df), "inf_values")
    assert found[0].count == 1 and found[0].column == "v"


def test_clean_frame_has_no_issues(clean_df):
    assert issues_of(clean_df) == []


def test_severity_ladder_boundaries():
    assert severity_for(0, 100) == "low"
    assert severity_for(2, 100) == "low"  # exactly 2% stays low
    assert severity_for(3, 100) == "medium"
    assert severity_for(10, 100) == "medium"  # exactly 10% stays medium
    assert severity_for(11, 100) == "high"
    assert severity_for(5, 0) == "low"  # degenerate input never crashes
