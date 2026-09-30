"""API integration tests — TestClient drives the real FastAPI stack."""

from __future__ import annotations

from pathlib import Path

SAMPLES = Path(__file__).resolve().parent.parent / "samples"

DIRTY = SAMPLES / "customers_dirty.csv"
CLEAN = SAMPLES / "customers_clean.csv"


def _upload(client, name: str, path: Path, **params):
    with open(path, "rb") as fh:
        return client.post(
            "/api/v1/analyze",
            files={"file": (name, fh, "text/csv")},
            params=params,
        )


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["ai_provider"] == "mock"
    assert "version" in body


def test_analyze_clean_sample(client):
    response = _upload(client, "customers_clean.csv", CLEAN)
    assert response.status_code == 200
    body = response.json()
    assert body["file"]["name"] == "customers_clean.csv"
    assert body["file"]["size_bytes"] > 0
    assert body["profile"]["row_count"] == 80
    assert body["profile"]["column_count"] == 8
    assert body["quality_score"]["score"] >= 90
    assert body["explanation"] is None  # explain defaults to false


def test_analyze_dirty_sample_full_contract_and_determinism(client):
    first = _upload(client, "customers_dirty.csv", DIRTY, explain=True).json()
    second = _upload(client, "customers_dirty.csv", DIRTY, explain=True).json()

    assert first["quality_score"]["score"] <= 70
    assert first["quality_score"]["grade"] in {"C", "D", "F"}

    # All seven issue types and all three anomaly methods fire on the dirty sample.
    assert {issue["type"] for issue in first["issues"]} == {
        "missing_values",
        "duplicate_rows",
        "inconsistent_types",
        "sentinel_values",
        "empty_column",
        "constant_column",
        "inf_values",
    }
    assert {a["method"] for a in first["anomalies"]} == {
        "iqr",
        "zscore",
        "isolation_forest",
    }

    # Deterministic AI: identical input → byte-identical explanation.
    assert first["explanation"]["provider"] == "mock"
    assert first["explanation"]["text"] == second["explanation"]["text"]


def test_rejects_non_csv_extension(client):
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("data.txt", b"a,b\n1,2\n", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_rejects_empty_file(client):
    response = client.post(
        "/api/v1/analyze", files={"file": ("empty.csv", b"", "text/csv")}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_csv"


def test_rejects_header_only_csv(client):
    response = client.post(
        "/api/v1/analyze", files={"file": ("headers.csv", b"a,b\n", "text/csv")}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_csv"


def test_rejects_malformed_csv(client):
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("bad.csv", b"a,b\n1,2,3\n", "text/csv")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_csv"


def test_rejects_non_utf8_file(client):
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("bad.csv", b"\xff\xfe\x00bad", "text/csv")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_csv"


def test_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0")  # cap of 0 bytes → everything is too big
    response = _upload(client, "big.csv", CLEAN)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_missing_file_part_returns_error_envelope(client):
    response = client.post("/api/v1/analyze")
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"


def test_unknown_route_uses_error_envelope(client):
    response = client.get("/nope")
    assert response.status_code == 404
    assert "error" in response.json()
