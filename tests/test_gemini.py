"""GeminiClient tests with a fake SDK client (offline). The real API is only hit by
the `live` test at the bottom, which is skipped unless you run `pytest -m live`."""

import os
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx
import pytest
from google.genai import errors

from gcl.config import Settings
from gcl.extractor import LLMError, SYSTEM_PROMPT, extract
from gcl.gemini import GeminiClient, client_from_settings
from gcl.models import RawExtraction
from gcl.parser import parse_chat_file

VALID = '{"expenses": [], "settlements": [], "commitments": [], "events": []}'


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)


def make(outcomes, **kwargs):
    models = FakeModels(outcomes)
    client = GeminiClient(
        api_key="unused", model="test-model", client=SimpleNamespace(models=models),
        sleep=lambda _: None, **kwargs,
    )
    return client, models


def call(client):
    return client.generate_json(system=SYSTEM_PROMPT, user="chat payload", schema=RawExtraction)


def api_error(cls, code, status):
    return cls(code, {"error": {"code": code, "message": "secret chat text", "status": status}})


def test_request_is_configured_for_structured_output():
    client, models = make([VALID])
    assert call(client) == VALID
    sent = models.calls[0]
    assert sent.model == "test-model"
    assert sent.contents == "chat payload"
    assert sent.config.response_mime_type == "application/json"
    assert sent.config.response_schema is RawExtraction
    assert sent.config.temperature == 0
    assert sent.config.system_instruction == SYSTEM_PROMPT


@pytest.mark.parametrize("first_failure", [
    "not json",
    '{"expenses": [{"payer": 1}]}',
    "",
    api_error(errors.ServerError, 503, "UNAVAILABLE"),
    api_error(errors.ClientError, 429, "RESOURCE_EXHAUSTED"),
    httpx.ReadTimeout("timed out"),
])
def test_transient_failures_are_retried(first_failure):
    client, models = make([first_failure, VALID])
    assert call(client) == VALID
    assert len(models.calls) == 2


def test_gives_up_after_max_attempts():
    client, models = make(["bad", "still bad"])
    with pytest.raises(LLMError, match="after 2 attempts"):
        call(client)
    assert len(models.calls) == 2


@pytest.mark.parametrize("code,status", [(400, "INVALID_ARGUMENT"), (403, "PERMISSION_DENIED"), (404, "NOT_FOUND")])
def test_permanent_errors_are_not_retried(code, status):
    client, models = make([api_error(errors.ClientError, code, status), VALID])
    with pytest.raises(LLMError, match=str(code)):
        call(client)
    assert len(models.calls) == 1


def test_errors_never_include_api_message():
    client, _ = make([api_error(errors.ClientError, 400, "INVALID_ARGUMENT")])
    with pytest.raises(LLMError) as info:
        call(client)
    assert "secret chat text" not in str(info.value)


def test_backoff_between_attempts():
    pauses = []
    models = FakeModels(["bad", "bad", VALID])
    client = GeminiClient(api_key="x", model="m", client=SimpleNamespace(models=models),
                          max_attempts=3, backoff_seconds=2, sleep=pauses.append)
    call(client)
    assert pauses == [2, 4]


def test_missing_key_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("GCL_GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GCL_GEMINI_API_KEY"):
        client_from_settings(Settings(_env_file=None))


# --- Live smoke test (real API, fictional fixture only) -------------------------------------

@pytest.mark.live
def test_live_extraction_on_fixture():
    settings = Settings()
    if settings.gemini_api_key is None:
        pytest.skip("GCL_GEMINI_API_KEY not set")
    parsed = parse_chat_file(Path(__file__).parent / "fixtures" / "android_dmy_12h.txt", settings=settings)
    result = extract(parsed, client_from_settings(settings), tz=ZoneInfo("Asia/Kolkata"))
    assert result.failed_windows == []
    assert any(e.payer == "Karan" and e.amount_paise == 1_200_000 for e in result.expenses)