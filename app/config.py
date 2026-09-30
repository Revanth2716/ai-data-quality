"""Environment-driven configuration (stdlib only — no config library)."""

from __future__ import annotations

import os
from dataclasses import dataclass

APP_VERSION = "1.0.0"


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """All runtime knobs; loaded fresh per request so tests can monkeypatch env."""

    ai_provider: str = "mock"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_timeout_secs: float = 20.0
    max_upload_mb: int = 10

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def load_settings() -> Settings:
    return Settings(
        ai_provider=(os.getenv("AI_PROVIDER", "mock").strip().lower() or "mock"),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
        openai_timeout_secs=float(os.getenv("OPENAI_TIMEOUT_SECS", "20").strip() or 20),
        max_upload_mb=_int_env("MAX_UPLOAD_MB", 10),
    )
