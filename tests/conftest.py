"""Shared pytest fixtures: app client and the two synthetic datasets."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import create_app  # noqa: E402
from samples.generate_samples import make_clean, make_dirty  # noqa: E402


@pytest.fixture(scope="session")
def clean_df() -> pd.DataFrame:
    return make_clean()


@pytest.fixture(scope="session")
def dirty_df() -> pd.DataFrame:
    return make_dirty()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())
