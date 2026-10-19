"""Gemini implementation of extractor.LLMClient.

DATA FLOW: calling generate_json() sends the prompt (chat message text, sender
names, timestamps) to Google's Gemini API. Callers must make that clear to the
user before sending a real chat. Nothing is sent at import or construction time.

Errors are reported by type/status code only, never with prompt or response text.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Callable

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from gcl.config import Settings, get_settings
from gcl.extractor import LLMError, QuotaExhausted

logger = logging.getLogger(__name__)

_RETRYABLE_CLIENT_CODES = {408, 429}  # timeout, rate limit
# A 429 asking us to wait longer than this is a daily quota, not a per-minute limit.
_MAX_RATE_LIMIT_WAIT = 120.0


def _retry_delay(exc: errors.APIError) -> float | None:
    """Seconds Google asks us to wait before retrying, if it says."""
    details = exc.details if isinstance(exc.details, dict) else {}
    for item in (details.get("error") or {}).get("details") or []:
        if isinstance(item, dict) and str(item.get("@type", "")).endswith("RetryInfo"):
            match = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(item.get("retryDelay", "")))
            if match:
                return float(match.group(1))
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", exc.message or "")
    return float(match.group(1)) if match else None


class GeminiClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: int = 120,
        max_attempts: int = 3,
        backoff_seconds: float = 2.0,
        thinking_level: str | None = None,   # e.g. "LOW"; None = model default
        client: genai.Client | None = None,          # injectable for tests
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.model = model
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.thinking_level = thinking_level
        self._sleep = sleep
        self._client = client or genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_seconds * 1000),  # milliseconds
        )

    def generate_json(self, *, system: str, user: str, schema: type[BaseModel]) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,   # Gemini constrains its output to this schema
            temperature=0,            # extraction, not creativity
            # No tools are ever given to the model; make that explicit.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=(
                types.ThinkingConfig(thinking_level=self.thinking_level)
                if self.thinking_level else None
            ),
        )
        last_problem = "unknown error"
        for attempt in range(1, self.max_attempts + 1):
            wait = self.backoff_seconds * attempt
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=user, config=config
                )
                text = response.text
                if not text:
                    raise _Retryable("empty response (possibly blocked)")
                schema.model_validate_json(text)  # retry once on malformed output
                return text
            except errors.ClientError as exc:
                if exc.code not in _RETRYABLE_CLIENT_CODES:
                    # 400 bad request, 401/403 bad key, 404 unknown model: retrying won't help.
                    raise LLMError(f"Gemini rejected the request ({exc.code} {exc.status})") from None
                last_problem = f"{exc.code} {exc.status}"
                delay = _retry_delay(exc) if exc.code == 429 else None
                if delay is not None and delay > _MAX_RATE_LIMIT_WAIT:
                    raise QuotaExhausted(
                        f"Gemini daily quota used up ({last_problem}); resets in about {delay / 3600:.1f} h"
                    ) from None
                if delay is not None:  # per-minute limit: wait as long as Google asks
                    wait = max(wait, delay + 1)
            except errors.ServerError as exc:
                last_problem = f"{exc.code} {exc.status}"
            except httpx.TransportError as exc:  # timeouts, connection failures
                last_problem = type(exc).__name__
            except ValidationError:
                last_problem = "response did not match schema"
            except _Retryable as exc:
                last_problem = str(exc)

            logger.warning("Gemini attempt %d/%d failed: %s", attempt, self.max_attempts, last_problem)
            if attempt < self.max_attempts:
                if wait > self.backoff_seconds * attempt:
                    logger.warning("Gemini rate limit; waiting %.0f s before retrying", wait)
                self._sleep(wait)

        error = QuotaExhausted if last_problem.startswith("429") else LLMError
        raise error(f"Gemini failed after {self.max_attempts} attempts: {last_problem}")


class _Retryable(Exception):
    pass


def client_from_settings(settings: Settings | None = None) -> GeminiClient:
    settings = settings or get_settings()
    return GeminiClient(
        api_key=settings.require_api_key(),
        model=settings.gemini_model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_attempts=settings.llm_max_attempts,
        thinking_level=settings.gemini_thinking_level,
    )