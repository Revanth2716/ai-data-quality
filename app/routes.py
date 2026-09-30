"""HTTP layer: health probe and the CSV analysis endpoint."""

from __future__ import annotations

import csv
import io

import pandas as pd
from fastapi import APIRouter, File, Query, Request, UploadFile

from .config import APP_VERSION, load_settings
from .errors import (
    InvalidCsvError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)
from .pipeline import analyze_dataframe
from .schemas import AnalysisResponse, FileMeta

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    settings = load_settings()
    return {
        "status": "ok",
        "version": APP_VERSION,
        "ai_provider": settings.ai_provider,
    }


def _validate_extension(filename: str | None, content_type: str | None) -> None:
    name = (filename or "").strip().lower()
    allowed_types = {
        "text/csv",
        "application/csv",
        "application/vnd.ms-excel",
        "text/plain",
        "application/octet-stream",
    }
    if not name.endswith(".csv") or (content_type is not None and content_type not in allowed_types):
        raise UnsupportedMediaTypeError(
            "Only .csv uploads are supported (send multipart/form-data with a 'file' part)."
        )


def _validate_rectangular(text: str) -> None:
    """Reject ragged CSVs: pandas 3 pads mismatched rows silently, so we check first."""
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise InvalidCsvError("The uploaded file contains no data.") from exc
    expected = len(header)
    for line_no, row in enumerate(reader, start=2):
        if not row:  # blank line
            continue
        if len(row) != expected:
            raise InvalidCsvError(
                f"Malformed CSV: line {line_no} has {len(row)} field(s), expected {expected}."
            )


def parse_csv(filename: str | None, content_type: str | None, raw: bytes) -> pd.DataFrame:
    """Validate and parse an uploaded CSV into a DataFrame; raise ApiError on problems."""
    _validate_extension(filename, content_type)
    if not raw:
        raise InvalidCsvError("The uploaded file is empty.")

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InvalidCsvError(f"The file is not valid UTF-8 text: {exc}") from exc
    if not text.strip():
        raise InvalidCsvError("The uploaded file contains no data.")

    _validate_rectangular(text)

    try:
        df = pd.read_csv(io.StringIO(text))
    except pd.errors.EmptyDataError as exc:
        raise InvalidCsvError("The uploaded file contains no data rows.") from exc
    except pd.errors.ParserError as exc:
        raise InvalidCsvError(f"The file could not be parsed as CSV: {exc}") from exc

    if df.empty:
        raise InvalidCsvError("The uploaded CSV has headers but no data rows.")
    return df


@router.post("/api/v1/analyze", response_model=AnalysisResponse)
def analyze(
    request: Request,
    file: UploadFile = File(..., description="CSV file to profile and analyze."),
    explain: bool = Query(
        False, description="Attach a natural-language explanation of the computed analysis."
    ),
) -> AnalysisResponse:
    settings = load_settings()
    max_bytes = settings.max_upload_bytes

    raw_bytes = file.file.read(max_bytes + 1)
    if len(raw_bytes) > max_bytes:
        raise PayloadTooLargeError(
            f"Upload exceeds the {settings.max_upload_mb} MB limit; "
            "adjust MAX_UPLOAD_MB to accept larger files."
        )

    df = parse_csv(file.filename, file.content_type, raw_bytes)
    size_bytes = len(raw_bytes)
    file.file.close()

    file_meta = FileMeta(name=file.filename or "upload.csv", size_bytes=size_bytes)
    return analyze_dataframe(df, file_meta, explain=explain)
