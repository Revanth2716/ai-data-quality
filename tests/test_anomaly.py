"""Anomaly-detector tests: IQR, Z-score, IsolationForest — incl. their gates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.anomaly import detect_anomalies
from app.profiler import profile_dataset


def _anoms(df: pd.DataFrame):
    return detect_anomalies(df, profile_dataset(df))


def _by_method(anoms, method: str):
    return [a for a in anoms if a.method == method]


def test_iqr_catches_planted_outlier():
    values = [float(v) for v in range(1, 21)] + [1000.0]
    df = pd.DataFrame({"v": values})
    found = _by_method(_anoms(df), "iqr")
    assert found and found[0].column == "v" and found[0].count == 1
    assert found[0].bounds["upper"] < 1000


def test_zscore_catches_outlier_with_enough_rows():
    rng = np.random.default_rng(7)
    values = list(rng.normal(50, 5, 60)) + [500.0]
    df = pd.DataFrame({"v": values})
    found = _by_method(_anoms(df), "zscore")
    assert found and found[0].count == 1
    assert found[0].threshold == 3.0


def test_zscore_gated_on_small_samples():
    df = pd.DataFrame({"v": [float(v) for v in range(1, 20)] + [1000.0]})  # 20 rows
    assert _by_method(_anoms(df), "zscore") == []  # n < 30
    found = _by_method(_anoms(df), "iqr")
    assert found and found[0].count == 1  # IQR still works on small data


def test_zero_iqr_column_is_skipped():
    df = pd.DataFrame({"v": [5.0] * 20})
    assert _by_method(_anoms(df), "iqr") == []


def test_isolation_forest_gated_on_small_or_narrow_data():
    two_rows = pd.DataFrame({"v": [1.0, 2.0], "w": [3.0, 4.0]})  # too few rows
    single_numeric = pd.DataFrame({"v": list(range(60))})  # only 1 numeric col
    assert _by_method(_anoms(two_rows), "isolation_forest") == []
    assert _by_method(_anoms(single_numeric), "isolation_forest") == []


def test_isolation_forest_flags_planted_multivariate_outliers():
    rng = np.random.default_rng(11)
    base = rng.normal(0, 1, (100, 2))
    outliers = np.array([[10.0, -10.0], [-12.0, 12.0]])
    df = pd.DataFrame(np.vstack([base, outliers]), columns=["x", "y"])
    found = _by_method(_anoms(df), "isolation_forest")
    assert found and found[0].column is None
    # The two planted outliers must be flagged; the exact count can wobble by
    # a row or two when a borderline inlier ties with the contamination cut.
    assert {100, 101} <= set(found[0].row_indices)
    assert 2 <= found[0].count <= 4


def test_dirty_sample_triggers_all_three_methods(dirty_df):
    methods = {a.method for a in detect_anomalies(dirty_df, profile_dataset(dirty_df))}
    assert methods == {"iqr", "zscore", "isolation_forest"}
