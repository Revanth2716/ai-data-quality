"""Pydantic v2 models — the single source of truth for the response contract."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_serializer


class FileMeta(BaseModel):
    name: str
    size_bytes: int


class ColumnStats(BaseModel):
    """Seven-number summary for a numeric column (None when not computable)."""

    mean: float | None = None
    std: float | None = None
    min: float | None = None
    p25: float | None = None
    median: float | None = None
    p75: float | None = None
    max: float | None = None


class ColumnProfile(BaseModel):
    name: str
    inferred_type: str  # numeric | categorical | text | boolean | datetime
    missing: int
    missing_ratio: float
    unique: int
    constant: bool
    stats: ColumnStats | None = None


class DatasetProfile(BaseModel):
    row_count: int
    column_count: int
    duplicate_row_count: int
    missing_cell_count: int
    missing_cell_ratio: float
    columns: list[ColumnProfile]


class Issue(BaseModel):
    """A detected data-quality problem."""

    type: str
    severity: str  # low | medium | high
    column: str | None = None
    count: int
    ratio: float
    message: str
    examples: list[Any] = Field(default_factory=list)
    row_indices: list[int] = Field(default_factory=list)


class Anomaly(BaseModel):
    """A statistical anomaly cluster found by one detection method."""

    method: str  # iqr | zscore | isolation_forest
    column: str | None = None  # None for dataset-wide methods
    count: int
    ratio: float
    bounds: dict[str, float] | None = None  # iqr lower/upper
    threshold: float | None = None  # zscore |z| cutoff
    examples: list[Any] = Field(default_factory=list)
    row_indices: list[int] = Field(default_factory=list)

    @field_serializer("row_indices")
    def _cap_row_indices(self, value: list[int]) -> list[int]:
        """Cap the full flagged-row list to 10 example indices in JSON output."""
        return value[:10]


class ScoreFactor(BaseModel):
    factor: str  # completeness | uniqueness | validity | consistency | anomalies
    penalty: float
    detail: str


class QualityScore(BaseModel):
    score: int  # 0-100
    grade: str  # A | B | C | D | F
    breakdown: list[ScoreFactor]


class Explanation(BaseModel):
    provider: str  # e.g. "mock" or "mock (fallback: <reason>)"
    text: str


class AnalysisResponse(BaseModel):
    file: FileMeta
    profile: DatasetProfile
    issues: list[Issue]
    anomalies: list[Anomaly]
    quality_score: QualityScore
    explanation: Explanation | None = None
