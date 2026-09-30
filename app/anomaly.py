"""Statistical anomaly detection: IQR fences, Z-score, IsolationForest.

Pure functions: DataFrame in, Pydantic models out.

- IQR: per numeric column, values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
- Z-score: per numeric column, |z| > 3 (needs n >= 30 and std > 0).
- IsolationForest: multivariate pass over the numeric matrix at a fixed 2%
  contamination; runs only when the dataset is big enough (>= 50 rows and
  >= 2 numeric columns) to be meaningful. Deterministic via random_state=42.

Infinite values are treated as missing here — they are already reported as
data-quality issues and would poison every statistic below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .profiler import coerce_numeric
from .schemas import Anomaly, DatasetProfile

IQR_K = 1.5
ZSCORE_THRESHOLD = 3.0
ZSCORE_MIN_ROWS = 30
ISOLATION_MIN_ROWS = 50
ISOLATION_MIN_COLS = 2
ISOLATION_CONTAMINATION = 0.02  # expected multivariate-outlier share (~2%)
ISOLATION_RANDOM_STATE = 42
MAX_EXAMPLES = 5


def _positions_of(mask: np.ndarray) -> tuple[list[int], int]:
    positions = np.flatnonzero(mask)
    count = int(positions.size)
    # Full index list: consumers (scoring) need the exact set; serialization caps it.
    return [int(i) for i in positions], count


def _numeric_series(df: pd.DataFrame, name: str) -> pd.Series:
    """Numeric view of a column: junk coerced to NaN, infinities to NaN."""
    numeric = coerce_numeric(df[name])
    return numeric.replace([np.inf, -np.inf], np.nan)


def _iqr_anomalies(df: pd.DataFrame, profile: DatasetProfile) -> list[Anomaly]:
    anomalies: list[Anomaly] = []
    for col in profile.columns:
        if col.inferred_type != "numeric":
            continue
        full = _numeric_series(df, col.name)
        values = full.dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        if iqr == 0:
            continue  # zero-IQR column: the fence collapses, nothing to test
        lower = float(q1 - IQR_K * iqr)
        upper = float(q3 + IQR_K * iqr)
        mask = (full < lower) | (full > upper)
        row_indices, count = _positions_of(mask.to_numpy())
        if count == 0:
            continue
        anomalies.append(
            Anomaly(
                method="iqr",
                column=col.name,
                count=count,
                ratio=round(count / len(df), 4),
                bounds={"lower": round(lower, 4), "upper": round(upper, 4)},
                row_indices=row_indices,
                examples=[float(v) for v in full[mask].head(MAX_EXAMPLES)],
            )
        )
    return anomalies


def _zscore_anomalies(df: pd.DataFrame, profile: DatasetProfile) -> list[Anomaly]:
    anomalies: list[Anomaly] = []
    for col in profile.columns:
        if col.inferred_type != "numeric":
            continue
        full = _numeric_series(df, col.name)
        values = full.dropna()
        if len(values) < ZSCORE_MIN_ROWS:
            continue
        std = float(values.std())
        if std == 0 or not np.isfinite(std):
            continue
        mean = float(values.mean())
        mask = (full - mean).abs() > ZSCORE_THRESHOLD * std
        mask = mask & full.notna()
        row_indices, count = _positions_of(mask.to_numpy())
        if count == 0:
            continue
        anomalies.append(
            Anomaly(
                method="zscore",
                column=col.name,
                count=count,
                ratio=round(count / len(df), 4),
                threshold=ZSCORE_THRESHOLD,
                row_indices=row_indices,
                examples=[float(v) for v in full[mask].head(MAX_EXAMPLES)],
            )
        )
    return anomalies


def _isolation_forest_anomalies(df: pd.DataFrame, profile: DatasetProfile) -> list[Anomaly]:
    numeric_cols = [c.name for c in profile.columns if c.inferred_type == "numeric"]
    if len(df) < ISOLATION_MIN_ROWS or len(numeric_cols) < ISOLATION_MIN_COLS:
        return []
    matrix = df[numeric_cols].apply(coerce_numeric).replace([np.inf, -np.inf], np.nan)
    matrix = matrix.dropna(axis=1, how="all")
    if matrix.shape[1] < ISOLATION_MIN_COLS:
        return []
    filled = matrix.fillna(matrix.median())  # median-impute gaps for the model
    model = IsolationForest(
        contamination=ISOLATION_CONTAMINATION, random_state=ISOLATION_RANDOM_STATE
    )
    labels = model.fit_predict(filled.to_numpy())  # -1 outlier, 1 inlier
    row_indices, count = _positions_of(labels == -1)
    if count == 0:
        return []
    return [
        Anomaly(
            method="isolation_forest",
            column=None,
            count=count,
            ratio=round(count / len(df), 4),
            row_indices=row_indices,
            examples=[],
        )
    ]


def detect_anomalies(df: pd.DataFrame, profile: DatasetProfile) -> list[Anomaly]:
    """Run all three detectors; results are ordered IQR, Z-score, IsolationForest."""
    anomalies: list[Anomaly] = []
    anomalies.extend(_iqr_anomalies(df, profile))
    anomalies.extend(_zscore_anomalies(df, profile))
    anomalies.extend(_isolation_forest_anomalies(df, profile))
    return anomalies
