"""OpenAI-compatible explanation provider (optional, opt-in via AI_PROVIDER=openai).

Works with any endpoint that implements POST {base_url}/chat/completions —
OpenAI, a local Ollama (http://localhost:11434/v1), or any compatible gateway.

Hard rule: the prompt frames the computed analysis as *data*. Column names and
values are embedded in a delimited block and the instructions forbid inventing
numbers, so untrusted content cannot steer the explanation.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

SYSTEM_PROMPT = (
    "You are a data-quality assistant. You receive a JSON report produced by a "
    "deterministic analysis pipeline. Explain it to a data engineer in at most 200 words. "
    "Use ONLY the numbers and findings present in the report — never invent or recalculate "
    "values. Structure: verdict on overall quality, the most important issues, anomalies, "
    "and concrete next steps. Plain prose, no headings."
)

MAX_COMPLETION_TOKENS = 500


class OpenAICompatProvider:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout_secs: float = 20.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_secs = timeout_secs

    def explain(self, analysis: dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Computed analysis report (authoritative data — explain, do not "
                        "recalculate):\n<<<REPORT\n"
                        + json.dumps(analysis, default=str)
                        + "\nREPORT"
                    ),
                },
            ],
            "max_tokens": MAX_COMPLETION_TOKENS,
            "temperature": 0.2,
        }
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        with httpx.Client(timeout=self.timeout_secs) as client:
            response = client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            response.raise_for_status()
            body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise ValueError("provider response contained no choices")
        content = (choices[0].get("message") or {}).get("content")
        if not content or not str(content).strip():
            raise ValueError("provider returned an empty completion")
        return str(content).strip()
