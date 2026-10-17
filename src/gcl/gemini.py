"""Gemini implementation of extractor.LLMClient.

DATA FLOW: calling generate_json() sends the prompt (chat message text, sender
names, timestamps) to Google's Gemini API. Callers must make that clear to the
user before sending a real chat. Nothing is sent at import or construction time.

Errors are reported by type/status code only, never with prompt or response text.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from gcl.config import Settings, get_settings
from gcl.extractor import LLMError

logger = logging.getLogger(__name__)

_RETRYABLE_CLIENT_CODES = {408, 429}  # timeout, rate limit


class GeminiClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: int = 60,
        max_attempts: int = 2,
        backoff_seconds: float = 2.0,
        client: genai.Client | None = None,          # injectable for tests
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.model = model
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
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
        )
        last_problem = "unknown error"
        for attempt in range(1, self.max_attempts + 1):
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
                self._sleep(self.backoff_seconds * attempt)

        raise LLMError(f"Gemini failed after {self.max_attempts} attempts: {last_problem}")


class _Retryable(Exception):
    pass


def client_from_settings(settings: Settings | None = None) -> GeminiClient:
    settings = settings or get_settings()
    return GeminiClient(
        api_key=settings.require_api_key(),
        model=settings.gemini_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )