"""Ollama implementation of extractor.LLMClient: runs the model locally, nothing leaves the machine.

Uses Ollama's structured outputs (`format` = JSON schema) so the reply is constrained to the
schema, like Gemini's response_schema. Errors never include prompt or response text.
"""

from __future__ import annotations

import copy
import logging

import httpx
from pydantic import BaseModel, ValidationError

from gcl.extractor import LLMError

logger = logging.getLogger(__name__)


def strict_schema(schema: type[BaseModel]) -> dict:
    """The JSON schema with every field required. Small local models under Ollama's JSON
    constraint otherwise stop after the first list and skip optional fields (like dates);
    being made to write each one, even as [] or null, makes them consider it."""
    def require_all(node):
        if isinstance(node, dict):
            if "properties" in node:
                node["required"] = list(node["properties"])
            for value in node.values():
                require_all(value)
        elif isinstance(node, list):
            for value in node:
                require_all(value)
        return node

    return require_all(copy.deepcopy(schema.model_json_schema()))


class OllamaClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 600,
        max_attempts: int = 2,
        num_ctx: int = 16384,
        http: httpx.Client | None = None,  # injectable for tests
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_attempts = max_attempts
        self.num_ctx = num_ctx
        self._http = http or httpx.Client(timeout=timeout_seconds)

    def generate_json(self, *, system: str, user: str, schema: type[BaseModel]) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": strict_schema(schema),
            "stream": False,
            "think": False,  # reading a chat needs no long reasoning; much faster (ignored by older models)
            # Ollama's default context is small and silently cuts long prompts.
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        }
        last_problem = "unknown error"
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._http.post(f"{self.base_url}/api/chat", json=body)
            except httpx.ConnectError:
                raise LLMError(f"Ollama is not running at {self.base_url}. Start the Ollama app "
                               "(or run `ollama serve`) and try again.") from None
            except httpx.TimeoutException:
                last_problem = "timed out"
            else:
                if response.status_code == 404:
                    raise LLMError(f"Ollama model {self.model!r} is not installed. "
                                   f"Run: ollama pull {self.model}") from None
                if response.status_code != 200:
                    last_problem = f"HTTP {response.status_code}"
                else:
                    text = ((response.json().get("message") or {}).get("content") or "").strip()
                    try:
                        schema.model_validate_json(text)
                        return text
                    except ValidationError:
                        last_problem = "response did not match schema"
            logger.warning("Ollama attempt %d/%d failed: %s", attempt, self.max_attempts, last_problem)

        raise LLMError(f"Ollama failed after {self.max_attempts} attempts: {last_problem}")
