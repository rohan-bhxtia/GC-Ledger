"""OllamaClient tests with a fake HTTP transport (no Ollama needed)."""

import json

import httpx
import pytest

from gcl.config import Settings
from gcl.extractor import LLMError
from gcl.llm import client_from_settings
from gcl.models import RawExtraction
from gcl.ollama import OllamaClient, strict_schema

VALID = '{"expenses": [], "settlements": [], "commitments": [], "events": []}'


def make(handler, **kwargs):
    sent = []

    def record(request):
        sent.append(json.loads(request.content))
        return handler(request)

    http = httpx.Client(transport=httpx.MockTransport(record))
    return OllamaClient(model="qwen3:8b", http=http, **kwargs), sent


def reply(content, status=200):
    return lambda _: httpx.Response(status, json={"message": {"role": "assistant", "content": content}})


def call(client):
    return client.generate_json(system="sys", user="chat payload", schema=RawExtraction)


def test_request_uses_schema_and_large_context():
    client, sent = make(reply(VALID), num_ctx=8192)
    assert call(client) == VALID
    body = sent[0]
    assert body["model"] == "qwen3:8b" and body["stream"] is False and body["think"] is False
    assert body["format"] == strict_schema(RawExtraction)
    assert body["options"] == {"temperature": 0, "num_ctx": 8192}
    assert body["messages"][1] == {"role": "user", "content": "chat payload"}


def test_malformed_output_is_retried():
    replies = iter(["not json", VALID])
    client, sent = make(lambda r: reply(next(replies))(r))
    assert call(client) == VALID
    assert len(sent) == 2


def test_missing_model_says_how_to_install_it():
    client, sent = make(reply("", status=404))
    with pytest.raises(LLMError, match="ollama pull qwen3:8b"):
        call(client)
    assert len(sent) == 1


def test_ollama_not_running_is_a_clear_error():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)
    client, _ = make(refuse)
    with pytest.raises(LLMError, match="not running"):
        call(client)


def test_provider_setting_picks_ollama(monkeypatch):
    monkeypatch.setenv("GCL_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("GCL_OLLAMA_MODEL", "llama3:8b")
    settings = Settings(_env_file=None)
    client = client_from_settings(settings)
    assert isinstance(client, OllamaClient) and client.model == "llama3:8b"
    assert settings.qa_max_chars is not None


def test_strict_schema_requires_every_field():
    schema = strict_schema(RawExtraction)
    assert schema["required"] == ["expenses", "settlements", "commitments", "events"]
    event = schema["$defs"]["RawEvent"]
    assert set(event["required"]) == set(event["properties"])  # incl. on_date / end_date
    assert "required" not in RawExtraction.model_json_schema()  # original left untouched
