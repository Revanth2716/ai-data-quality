"""Uniform error envelope and typed API errors.

Every error the client can see uses the same shape:

    {"error": {"code": "<machine_readable>", "message": "<human readable>"}}
"""

from __future__ import annotations

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class ApiError(Exception):
    """Base class for expected, user-facing API errors."""

    status_code = 400
    code = "bad_request"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidCsvError(ApiError):
    status_code = 400
    code = "invalid_csv"


class UnsupportedMediaTypeError(ApiError):
    status_code = 400
    code = "unsupported_media_type"


class PayloadTooLargeError(ApiError):
    status_code = 413
    code = "payload_too_large"


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    return error_response(exc.status_code, exc.code, exc.message)


def http_error_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    return error_response(exc.status_code, "http_error", str(exc.detail))


def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    parts = [
        f"{'.'.join(str(loc) for loc in err.get('loc', []))}: {err.get('msg', 'invalid')}"
        for err in exc.errors()
    ]
    return error_response(422, "validation_error", "; ".join(parts) or "Invalid request.")


def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    return error_response(500, "internal_error", f"Unexpected server error: {exc}")
