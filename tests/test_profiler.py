"""Profiler tests — inline DataFrames, no HTTP."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.profiler import profile_dataset


def test_shape_and_missing_totals():
    df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": ["x", "y", "z", np.nan]})
    profile = profile_dataset(df)
    assert profile.row_count == 4
    assert profile.column_count == 2
    assert profile.missing_cell_count == 2
    assert profile.missing_cell_ratio == 0.25


def test_duplicate_row_count():
    df = pd.DataFrame({"a": [1, 2, 2, 2], "b": ["x", "y", "y", "y"]})
    assert profile_dataset(df).duplicate_row_count == 2


def test_numeric_inference_and_stats():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    col = profile_dataset(df).columns[0]
    assert col.inferred_type == "numeric"
    assert col.stats.mean == 3
    assert col.stats.median == 3
    assert col.stats.min == 1 and col.stats.max == 5
    assert col.unique == 5 and col.constant is False


def test_categorical_inference_and_constant_flag():
    df = pd.DataFrame({"c": ["a", "a", "b"]})
    col = profile_dataset(df).columns[0]
    assert col.inferred_type == "categorical"
    assert col.unique == 2

    constant = profile_dataset(pd.DataFrame({"c": ["a", "a", "a"]})).columns[0]
    assert constant.unique == 1 and constant.constant is True


def test_mostly_numeric_strings_reported_numeric():
    df = pd.DataFrame({"a": ["1", "2", "3", "4", "5", "6", "7", "oops"]})
    assert profile_dataset(df).columns[0].inferred_type == "numeric"


def test_all_nan_numeric_column_has_no_stats():
    col = profile_dataset(pd.DataFrame({"n": [np.nan, np.nan]})).columns[0]
    assert col.inferred_type == "numeric"
    assert col.stats is None
