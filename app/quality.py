"""Data-quality analyzer: deterministic issue detection over the dataset.

Issue types (7):
    missing_values, duplicate_rows, inconsistent_types, sentinel_values,
    empty_column, constant_column, inf_values

All functions are pure: DataFrame in, Pydantic models out. Each issue carries a
count, a ratio, a human-readable message and (where useful) row indices or
example values so callers can jump straight to the offending data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .profiler import NUMERIC_INFERENCE_THRESHOLD, coerce_numeric, is_stringy
from .schemas import DatasetProfile, Issue

# Tokens that mean "missing" to humans but survive default CSV parsing as strings.
SENTINEL_TOKENS = {"n/a", "na", "null", "none", "-", "?", "unknown"}
_SENTINEL_CASEFOLD = {token.casefold() for token in SENTINEL_TOKENS}

# Severity ladder: <=2% low, <=10% medium, else high (ratio = affected rows).
SEVERITY_MEDIUM_RATIO = 0.02
SEVERITY_HIGH_RATIO = 0.10

ISSUE_TYPES = (
    "missing_values",
    "duplicate_rows",
    "inconsistent_types",
    "sentinel_values",
    "empty_column",
    "constant_column",
    "inf_values",
)


def severity_for(count: int, total: int) -> str:
    if total <= 0:
        return "low"
    ratio = count / total
    if ratio > SEVERITY_HIGH_RATIO:
        return "high"
    if ratio > SEVERITY_MEDIUM_RATIO:
        return "medium"
    return "low"


def _mask_row_indices(mask: pd.Series, limit: int = 5) -> list[int]:
    return [int(i) for i in mask[mask].head(limit).index]


def _missing_issues(df: pd.DataFrame, profile: DatasetProfile) -> list[Issue]:
    issues: list[Issue] = []
    for col in profile.columns:
        if col.missing == 0:
            continue
        mask = df[col.name].isna()
        issues.append(
            Issue(
                type="missing_values",
                severity=severity_for(col.missing, len(df)),
                column=col.name,
                count=col.missing,
                ratio=col.missing_ratio,
                message=(
                    f"Column '{col.name}' has {col.missing} missing value(s) "
                    f"({col.missing_ratio:.1%} of rows)."
                ),
                row_indices=_mask_row_indices(mask),
            )
        )
    return issues


def _duplicate_issues(df: pd.DataFrame) -> list[Issue]:
    dup_mask = df.duplicated(keep="first")
    count = int(dup_mask.sum())
    if count == 0:
        return []
    # Point at the first occurrence of each duplicated group.
    first_occurrences = df[df.duplicated(keep=False)].drop_duplicates().index.tolist()
    return [
        Issue(
            type="duplicate_rows",
            severity=severity_for(count, len(df)),
            count=count,
            ratio=round(count / len(df), 4),
            message=(
                f"{count} duplicate row(s) detected "
                f"({count / len(df):.1%} of rows repeat an earlier row exactly)."
            ),
            row_indices=[int(i) for i in first_occurrences[:5]],
        )
    ]


def _inconsistent_type_issues(df: pd.DataFrame) -> list[Issue]:
    """String columns that are >=80% numeric but contain unparseable values."""
    issues: list[Issue] = []
    for column in df.columns:
        series = df[column]
        if not is_stringy(series):
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        coerced = coerce_numeric(non_null)
        numeric_share = float(coerced.notna().mean())
        if numeric_share < NUMERIC_INFERENCE_THRESHOLD or not coerced.isna().any():
            continue
        bad_values = [str(v) for v in non_null[coerced.isna()].tolist()]
        bad_count = len(bad_values)
        issues.append(
            Issue(
                type="inconsistent_types",
                severity="high",
                column=column,
                count=bad_count,
                ratio=round(bad_count / len(df), 4),
                message=(
                    f"Column '{column}' mixes numeric strings with {bad_count} value(s) "
                    f"that do not parse as numbers, e.g. {bad_values[:3]}."
                ),
                examples=bad_values[:5],
            )
        )
    return issues


def _sentinel_issues(df: pd.DataFrame) -> list[Issue]:
    """Human missing-markers ('N/A', '?', 'unknown', ...) stored as strings."""
    issues: list[Issue] = []
    for column in df.columns:
        series = df[column]
        if not is_stringy(series):
            continue
        normalized = series.dropna().astype(str).str.strip().str.casefold()
        hit = normalized.isin(_SENTINEL_CASEFOLD)
        count = int(hit.sum())
        if count == 0:
            continue
        hit_indices = normalized[hit].index
        issues.append(
            Issue(
                type="sentinel_values",
                severity=severity_for(count, len(df)),
                column=column,
                count=count,
                ratio=round(count / len(df), 4),
                message=(
                    f"Column '{column}' contains {count} placeholder value(s) such as "
                    "'N/A', '?' or 'unknown' that should be treated as missing."
                ),
                examples=[str(series.loc[i]) for i in hit_indices[:5]],
            )
        )
    return issues


def _empty_constant_issues(df: pd.DataFrame, profile: DatasetProfile) -> list[Issue]:
    issues: list[Issue] = []
    for col in profile.columns:
        if len(df) > 0 and col.missing == len(df):
            issues.append(
                Issue(
                    type="empty_column",
                    severity="high",
                    column=col.name,
                    count=len(df),
                    ratio=1.0,
                    message=f"Column '{col.name}' is completely empty.",
                )
            )
        elif col.unique == 1 and not df[col.name].dropna().empty:
            value = df[col.name].dropna().iloc[0]
            issues.append(
                Issue(
                    type="constant_column",
                    severity="low",
                    column=col.name,
                    count=len(df),
                    ratio=1.0,
                    message=(
                        f"Column '{col.name}' has a single distinct value ({value!r}); "
                        "it adds no information."
                    ),
                )
            )
    return issues


def _inf_issues(df: pd.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    for column in df.columns:
        series = df[column]
        if not is_numeric_dtype(series):
            continue
        values = series.to_numpy(dtype=float, na_value=np.nan)
        mask = np.isinf(values)
        count = int(mask.sum())
        if count == 0:
            continue
        positions = np.flatnonzero(mask)[:5]
        issues.append(
            Issue(
                type="inf_values",
                severity="high",
                column=column,
                count=count,
                ratio=round(count / len(df), 4),
                message=(
                    f"Column '{column}' contains {count} infinite value(s); "
                    "infinities break most downstream statistics and models."
                ),
                examples=[float(values[i]) for i in positions],
            )
        )
    return issues


def detect_quality_issues(df: pd.DataFrame, profile: DatasetProfile) -> list[Issue]:
    """Run all detectors and merge their findings into one ordered list."""
    issues: list[Issue] = []
    issues.extend(_missing_issues(df, profile))
    issues.extend(_duplicate_issues(df))
    issues.extend(_inconsistent_type_issues(df))
    issues.extend(_sentinel_issues(df))
    issues.extend(_empty_constant_issues(df, profile))
    issues.extend(_inf_issues(df))
    return issues
