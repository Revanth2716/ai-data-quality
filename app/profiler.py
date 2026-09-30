"""Dataset profiler: shape, duplicates, missingness, per-column types and stats.

Pure functions: DataFrame in, Pydantic models out. No HTTP or I/O concerns.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_string_dtype,
)

from .schemas import ColumnProfile, ColumnStats, DatasetProfile

CATEGORICAL_MAX_UNIQUE = 20
NUMERIC_INFERENCE_THRESHOLD = 0.8  # share of values that must parse as numbers


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Best-effort numeric coercion; anything unparseable becomes NaN."""
    return pd.to_numeric(series, errors="coerce")


def finite_or_none(value: float) -> float | None:
    """JSON-safe float: non-finite values (inf/nan) become None."""
    value = float(value)
    return value if math.isfinite(value) else None


def is_stringy(series: pd.Series) -> bool:
    """True for string or object columns under both pandas 2.x and 3.x."""
    return bool(is_string_dtype(series) or series.dtype == object)


def infer_type(series: pd.Series) -> str:
    if is_bool_dtype(series):
        return "boolean"
    if is_numeric_dtype(series):
        return "numeric"
    if is_datetime64_any_dtype(series):
        return "datetime"
    if is_stringy(series):
        non_null = series.dropna()
        if non_null.empty:
            return "text"
        coerced = coerce_numeric(non_null)
        if float(coerced.notna().mean()) >= NUMERIC_INFERENCE_THRESHOLD:
            # Mostly-numeric column polluted with a few junk strings.
            return "numeric"
        n_unique = int(non_null.nunique())
        if 0 < n_unique <= CATEGORICAL_MAX_UNIQUE:
            return "categorical"
        return "text"
    return "text"


def numeric_stats(series: pd.Series) -> ColumnStats | None:
    """Seven-number summary over finite values only (NaN/inf excluded)."""
    numeric = coerce_numeric(series).replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return None
    q1, median, q3 = numeric.quantile([0.25, 0.5, 0.75])
    return ColumnStats(
        mean=finite_or_none(numeric.mean()),
        std=finite_or_none(numeric.std()) if len(numeric) > 1 else None,
        min=finite_or_none(numeric.min()),
        p25=finite_or_none(q1),
        median=finite_or_none(median),
        p75=finite_or_none(q3),
        max=finite_or_none(numeric.max()),
    )


def profile_column(name: str, series: pd.Series, row_count: int) -> ColumnProfile:
    missing = int(series.isna().sum())
    try:
        unique = int(series.nunique(dropna=True))
    except TypeError:  # unhashable values (should not happen for CSV data)
        unique = int(series.astype(str).nunique(dropna=True))
    inferred = infer_type(series)
    return ColumnProfile(
        name=name,
        inferred_type=inferred,
        missing=missing,
        missing_ratio=round(missing / row_count, 4) if row_count else 0.0,
        unique=unique,
        constant=unique <= 1,
        stats=numeric_stats(series) if inferred == "numeric" else None,
    )


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    row_count, column_count = df.shape
    total_cells = row_count * column_count
    missing_cell_count = int(df.isna().sum().sum())
    return DatasetProfile(
        row_count=row_count,
        column_count=column_count,
        duplicate_row_count=int(df.duplicated().sum()),
        missing_cell_count=missing_cell_count,
        missing_cell_ratio=round(missing_cell_count / total_cells, 4) if total_cells else 0.0,
        columns=[profile_column(name, df[name], row_count) for name in df.columns],
    )
