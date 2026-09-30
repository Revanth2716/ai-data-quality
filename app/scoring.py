"""Quality score: 100 minus capped deterministic penalties across five factors.

The score is computed entirely in code — the AI layer only ever *explains* it,
never recalculates it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .profiler import coerce_numeric
from .schemas import Anomaly, DatasetProfile, Issue, QualityScore, ScoreFactor

# Start from 100; each factor can cost at most its cap.
WEIGHTS: dict[str, int] = {
    "completeness": 25,
    "uniqueness": 20,
    "validity": 15,
    "consistency": 10,
    "anomalies": 30,
}

GRADE_BANDS: tuple[tuple[int, str], ...] = (
    (90, "A"),
    (75, "B"),
    (60, "C"),
    (40, "D"),
)


def grade_for(score: int) -> str:
    for threshold, grade in GRADE_BANDS:
        if score >= threshold:
            return grade
    return "F"


def _factor(factor: str, penalty: float, detail: str) -> ScoreFactor | None:
    return ScoreFactor(factor=factor, penalty=round(penalty, 2), detail=detail) if penalty else None


def _numeric_matrix(df: pd.DataFrame, profile: DatasetProfile) -> pd.DataFrame:
    """Numeric view used for the validity ratio: coerced, inf preserved as inf."""
    cols = [c.name for c in profile.columns if c.inferred_type == "numeric"]
    if not cols:
        return pd.DataFrame(index=df.index)
    return df[cols].apply(coerce_numeric)


def compute_quality_score(
    df: pd.DataFrame,
    profile: DatasetProfile,
    issues: list[Issue],
    anomalies: list[Anomaly],
) -> QualityScore:
    row_count = max(len(df), 1)
    breakdown: list[ScoreFactor] = []

    # Completeness — missing cells across the whole dataset.
    completeness = min(profile.missing_cell_ratio * 250.0, WEIGHTS["completeness"])
    factor = _factor(
        "completeness",
        completeness,
        f"{profile.missing_cell_count} missing cells ({profile.missing_cell_ratio:.1%} of all cells).",
    )
    if factor:
        breakdown.append(factor)

    # Uniqueness — exact duplicate rows.
    duplicate_ratio = profile.duplicate_row_count / row_count
    uniqueness = min(duplicate_ratio * 200.0, WEIGHTS["uniqueness"])
    factor = _factor(
        "uniqueness",
        uniqueness,
        f"{profile.duplicate_row_count} duplicate rows ({duplicate_ratio:.1%} of rows).",
    )
    if factor:
        breakdown.append(factor)

    # Validity — infinite values and structurally useless columns.
    numeric = _numeric_matrix(df, profile)
    total_cells = max(numeric.size, 1)
    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    inf_ratio = float(np.isinf(values).sum()) / total_cells
    empty_columns = sum(1 for issue in issues if issue.type == "empty_column")
    constant_columns = sum(1 for issue in issues if issue.type == "constant_column")
    validity = min(inf_ratio * 100.0 + 5.0 * (empty_columns + constant_columns), WEIGHTS["validity"])
    factor = _factor(
        "validity",
        validity,
        "; ".join(
            part
            for part in (
                f"{inf_ratio:.2%} infinite numeric cells" if inf_ratio else "",
                f"{empty_columns} empty column(s)" if empty_columns else "",
                f"{constant_columns} constant column(s)" if constant_columns else "",
            )
            if part
        ),
    )
    if factor:
        breakdown.append(factor)

    # Consistency — type mismatches and placeholder sentinel cells.
    consistency_cells = sum(
        issue.count
        for issue in issues
        if issue.type in {"inconsistent_types", "sentinel_values"}
    )
    consistency = min(consistency_cells / row_count * 50.0, WEIGHTS["consistency"])
    factor = _factor(
        "consistency",
        consistency,
        f"{consistency_cells} cell(s) with type mismatches or sentinel values.",
    )
    if factor:
        breakdown.append(factor)

    # Anomalies — union of rows flagged by any detection method.
    flagged_rows = {int(i) for anomaly in anomalies for i in anomaly.row_indices}
    anomaly_penalty = min(len(flagged_rows) / row_count * 200.0, WEIGHTS["anomalies"])
    factor = _factor(
        "anomalies",
        anomaly_penalty,
        f"{len(flagged_rows)} row(s) flagged by at least one anomaly method.",
    )
    if factor:
        breakdown.append(factor)

    score = int(round(max(0.0, min(100.0, 100.0 - sum(f.penalty for f in breakdown)))))
    return QualityScore(score=score, grade=grade_for(score), breakdown=breakdown)
