"""FastAPI application factory.

create_app() builds the full stack so tests drive exactly what production runs.
Run locally with:
    uv run uvicorn app.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import APP_VERSION, load_settings
from .errors import (
    ApiError,
    api_error_handler,
    http_error_handler,
    unhandled_error_handler,
    validation_error_handler,
)
from .routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Data Quality & Anomaly Detection API",
        version=APP_VERSION,
        description=(
            "Upload a CSV, get a quality profile, detected issues, statistical anomalies, "
            "a 0-100 quality score, and an optional AI-generated explanation of the "
            "computed analysis. Works fully offline with the deterministic mock provider."
        ),
    )
    app.include_router(router)

    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
    return app


app = create_app()
