"""Regenerate the synthetic demo CSVs (seeded — output is deterministic).

    uv run python samples/generate_samples.py

customers_clean.csv  — 80 tidy rows, expect a high quality score (A/B).
customers_dirty.csv  — 119 rows with planted defects, expect a low score (D/F):
    ~8% missing cells, 6 exact duplicates, 'N/A'/'?'/'unknown' sentinels,
    mixed-type ages, a 9,999,999 salary outlier, and a constant column.

Every value is synthetic; no real personal data is used anywhere.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

COLUMNS = [
    "customer_id",
    "name",
    "age",
    "email",
    "city",
    "signup_date",
    "salary",
    "plan",
]

CITIES = ["Springfield", "Riverton", "Fairview", "Georgetown", "Clinton", "Salem"]
PLANS = ["basic", "standard", "premium"]

FIRST_NAMES = [
    "Alex", "Blake", "Casey", "Drew", "Ellis", "Frankie", "Gale", "Harper",
    "Indigo", "Jamie", "Kelly", "Logan", "Morgan", "Noel", "Oakley", "Peyton",
    "Quinn", "Riley", "Sage", "Terry",
]
LAST_INITIALS = list("ABCDEFGHIJ")


def base_frame(rng: random.Random, n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "customer_id": f"C{i + 1:04d}",
                "name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_INITIALS)}.",
                "age": rng.randint(18, 75),
                "email": f"user{i + 1}@example.com",
                "city": rng.choice(CITIES),
                "signup_date": (
                    pd.Timestamp("2024-01-01") + pd.Timedelta(days=rng.randint(0, 700))
                ).date().isoformat(),
                "salary": rng.randint(30_000, 140_000),
                "plan": rng.choice(PLANS),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def make_clean(n: int = 80) -> pd.DataFrame:
    rng = random.Random(42)
    return base_frame(rng, n)


def make_dirty(n: int = 113) -> pd.DataFrame:
    """113 base rows; 6 exact duplicates are appended to reach 119 rows."""
    rng = random.Random(42)
    df = base_frame(rng, n).copy()

    # pandas 3 refuses lossy setitem: make the age column object-typed up front
    # so it can hold numbers, NaN and junk strings side by side.
    df["age"] = df["age"].astype(object)

    # 1. ~8% missing cells scattered over numeric and string columns.
    for column in ["age", "email", "salary", "city"]:
        for row in rng.sample(range(len(df)), 12):
            df.loc[row, column] = np.nan  # type: ignore[call-overload]

    # 2. Placeholder sentinels that survive default CSV parsing as real strings.
    #    ('N/A' is consumed by pandas' NA parser, so '-' stands in for city.)
    for column, token in [("city", "-"), ("plan", "?"), ("name", "unknown")]:
        for row in rng.sample(range(len(df)), 5):
            df.loc[row, column] = token

    # 3. Mixed-type ages: junk strings in an otherwise numeric column.
    for row, text in zip(rng.sample(range(len(df)), 3), ["twenty-five", "??", "n/a"], strict=False):
        df.loc[row, "age"] = text

    # 4. A glaring salary outlier plus an infinite value.
    df.loc[10, "salary"] = 9_999_999
    df["salary"] = df["salary"].astype(float)
    df.loc[11, "salary"] = np.inf

    # 5. A constant column that carries no information.
    df["region_code"] = "EU-WEST"

    # 6. A completely empty column.
    df["notes"] = np.nan

    # 7. Six exact duplicates (appended at the end).
    duplicates = df.sample(n=6, random_state=rng.randint(0, 10_000))
    return pd.concat([df, duplicates], ignore_index=True)


def main() -> None:
    clean = make_clean()
    dirty = make_dirty()
    clean.to_csv(HERE / "customers_clean.csv", index=False)
    dirty.to_csv(HERE / "customers_dirty.csv", index=False)
    print(f"wrote customers_clean.csv ({clean.shape[0]} rows)")
    print(f"wrote customers_dirty.csv ({dirty.shape[0]} rows)")


if __name__ == "__main__":
    main()
