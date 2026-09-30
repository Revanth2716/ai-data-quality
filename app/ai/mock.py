"""Deterministic mock explanation — the offline default.

Renders a template strictly from the computed analysis dict. No randomness,
no network, no API key: the same analysis always yields the same text, which
makes the behaviour testable and the demo reproducible.
"""

from __future__ import annotations

from typing import Any

SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}
MAX_ISSUES = 3
MAX_ANOMALIES = 3
MAX_MISSING_COLUMNS = 3

METHOD_LABELS = {
    "iqr": "IQR fences",
    "zscore": "Z-score (|z| > 3)",
    "isolation_forest": "IsolationForest",
}

RECOMMENDATIONS = {
    "missing_values": "Impute or drop missing values, and check upstream collection for systematic gaps.",
    "duplicate_rows": "De-duplicate on load and add a uniqueness constraint at the source system.",
    "inconsistent_types": "Normalise column types at ingestion (e.g. explicit numeric parsing with error handling).",
    "sentinel_values": "Map placeholder tokens such as 'N/A', '?' or 'unknown' to real nulls during parsing.",
    "empty_column": "Drop empty columns before analysis — they carry no information.",
    "constant_column": "Drop constant columns — a single distinct value adds no signal.",
    "inf_values": "Replace infinities with NaN or domain-capped values before modelling.",
}

GRADE_VERDICTS = {
    "A": "This dataset is in good shape — only minor polish is needed.",
    "B": "This dataset is broadly usable, with some fixable quality debt.",
    "C": "This dataset needs attention before it can be trusted for analysis.",
    "D": "This dataset is in poor shape — fix the issues below before any downstream use.",
    "F": "This dataset is critically compromised and should not be used as-is.",
}


def _format_pct(value: float) -> str:
    return f"{value:.1%}"


def explain(analysis: dict[str, Any]) -> str:
    score = analysis["quality_score"]
    profile = analysis["profile"]
    issues: list[dict[str, Any]] = analysis["issues"]
    anomalies: list[dict[str, Any]] = analysis["anomalies"]

    lines: list[str] = []
    lines.append(
        f"Quality verdict: the dataset scores {score['score']}/100 (grade {score['grade']}). "
        f"{GRADE_VERDICTS[score['grade']]}"
    )
    lines.append(
        f"It holds {profile['row_count']} rows across {profile['column_count']} columns; "
        f"{profile['missing_cell_count']} cells are missing "
        f"({_format_pct(profile['missing_cell_ratio'])}) and "
        f"{profile['duplicate_row_count']} rows are exact duplicates."
    )

    if issues:
        ranked = sorted(issues, key=lambda i: (SEVERITY_RANK.get(i["severity"], 3), -i["count"]))
        lines.append("What the checks found:")
        for issue in ranked[:MAX_ISSUES]:
            where = f" in '{issue['column']}'" if issue.get("column") else ""
            lines.append(
                f"- [{issue['severity']}] {issue['type'].replace('_', ' ')}{where}: "
                f"{issue['count']} affected ({_format_pct(issue['ratio'])})."
            )
    else:
        lines.append("No data-quality issues were detected by the deterministic checks.")

    missing_columns = sorted(
        (c for c in profile["columns"] if c["missing"] > 0),
        key=lambda c: -c["missing_ratio"],
    )[:MAX_MISSING_COLUMNS]
    if missing_columns:
        worst = ", ".join(
            f"'{c['name']}' ({_format_pct(c['missing_ratio'])})" for c in missing_columns
        )
        lines.append(f"Missingness is concentrated in: {worst}.")

    if anomalies:
        parts = []
        for anomaly in anomalies[:MAX_ANOMALIES]:
            where = f" in '{anomaly['column']}'" if anomaly.get("column") else " across the dataset"
            label = METHOD_LABELS.get(anomaly["method"], anomaly["method"])
            parts.append(f"{label}{where} flagged {anomaly['count']} row(s)")
        lines.append("Statistical anomalies: " + "; ".join(parts) + ".")

    unique_types = list(dict.fromkeys(issue["type"] for issue in issues))  # dedupe, keep order
    if unique_types:
        lines.append("Recommended fixes:")
        for issue_type in unique_types:
            lines.append(f"- {RECOMMENDATIONS.get(issue_type, 'Review this issue manually.')}")

    lines.append(
        "This explanation was generated deterministically from the computed analysis; "
        "no AI model was involved."
    )
    return "\n".join(lines)


# Canonical entry point used by the provider module.
build_mock_explanation = explain
