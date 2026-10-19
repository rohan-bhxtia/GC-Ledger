"""Talk to Google Gemini.

DATA: calling generate_json() sends the prompt (message text, sender names, times) to Google.
The app asks the user before the first send. Nothing is sent when this file is imported or
when the client is created.

What happens when something goes wrong:
  - Bad request, bad key, unknown model (400/401/403/404): stop at once. Retrying won't help.
  - Server error, timeout, network problem, or a reply in the wrong shape: try again,
    waiting a little longer each time (2 s, then 4 s, ...).
  - Too many requests (429):
      * per-minute limit: Google says how long to wait (e.g. 30 s). We wait that long, then retry.
      * daily limit: Google says to wait hours. We stop at once with QuotaExhausted.

Error messages contain only status codes, never the prompt or the reply.
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

RETRYABLE_CLIENT_CODES = {408, 429}  # 408 = timeout, 429 = too many requests
# If a 429 asks us to wait longer than this, it's the daily quota, not the per-minute limit.
MAX_RATE_LIMIT_WAIT_SECONDS = 120.0


class _TryAgain(Exception):
    """An odd reply (e.g. empty) that's worth retrying."""


def _wait_google_asks_for(error: errors.APIError) -> float | None:
    """How many seconds Google tells us to wait before retrying, if it says."""
    # Usually it's in the error details as {"@type": "...RetryInfo", "retryDelay": "33s"}.
    details = error.details if isinstance(error.details, dict) else {}
    for item in (details.get("error") or {}).get("details") or []:
        if isinstance(item, dict) and str(item.get("@type", "")).endswith("RetryInfo"):
            match = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(item.get("retryDelay", "")))
            if match:
                return float(match.group(1))

    # Sometimes only the message says it: "... Please retry in 33.5s."
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", error.message or "")
    if match:
        return float(match.group(1))
    return None


class GeminiClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: int = 120,
        max_attempts: int = 3,
        backoff_seconds: float = 2.0,
        thinking_level: str | None = None,          # e.g. "LOW"; None = the model's default
        client: genai.Client | None = None,          # the tests pass a fake one
        sleep: Callable[[float], None] = time.sleep,  # the tests pass a fake one
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
            http_options=types.HttpOptions(timeout=timeout_seconds * 1000),  # in milliseconds
        )

    def generate_json(self, *, system: str, user: str, schema: type[BaseModel]) -> str:
        thinking = None
        if self.thinking_level:
            thinking = types.ThinkingConfig(thinking_level=self.thinking_level)

        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,   # Gemini must reply with JSON in this shape
            temperature=0,            # same input -> same answer
            thinking_config=thinking,
            # We never give the model tools; say so explicitly.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        last_problem = "unknown error"
        for attempt in range(1, self.max_attempts + 1):
            normal_wait = self.backoff_seconds * attempt   # 2 s, 4 s, 6 s, ...
            wait = normal_wait

            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=user, config=config
                )
                text = response.text
                if not text:
                    raise _TryAgain("empty response (possibly blocked)")
                schema.model_validate_json(text)  # is it really the right shape?
                return text

            except errors.ClientError as error:
                if error.code not in RETRYABLE_CLIENT_CODES:
                    raise LLMError(f"Gemini rejected the request ({error.code} {error.status})") from None
                last_problem = f"{error.code} {error.status}"

                if error.code == 429:
                    asked_wait = _wait_google_asks_for(error)
                    if asked_wait is not None and asked_wait > MAX_RATE_LIMIT_WAIT_SECONDS:
                        hours = asked_wait / 3600
                        raise QuotaExhausted(
                            f"Gemini daily quota used up ({last_problem}); resets in about {hours:.1f} h"
                        ) from None
                    if asked_wait is not None:
                        wait = max(wait, asked_wait + 1)

            except errors.ServerError as error:
                last_problem = f"{error.code} {error.status}"
            except httpx.TransportError as error:  # timeouts, connection failures
                last_problem = type(error).__name__
            except ValidationError:
                last_problem = "response did not match schema"
            except _TryAgain as error:
                last_problem = str(error)

            logger.warning("Gemini attempt %d/%d failed: %s", attempt, self.max_attempts, last_problem)
            if attempt < self.max_attempts:
                if wait > normal_wait:
                    logger.warning("Gemini rate limit; waiting %.0f s before retrying", wait)
                self._sleep(wait)

        # Every attempt failed. If the last problem was 429, treat it as the quota.
        if last_problem.startswith("429"):
            raise QuotaExhausted(f"Gemini failed after {self.max_attempts} attempts: {last_problem}")
        raise LLMError(f"Gemini failed after {self.max_attempts} attempts: {last_problem}")


def client_from_settings(settings: Settings | None = None) -> GeminiClient:
    settings = settings or get_settings()
    return GeminiClient(
        api_key=settings.require_api_key(),
        model=settings.gemini_model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_attempts=settings.llm_max_attempts,
        thinking_level=settings.gemini_thinking_level,
    )
