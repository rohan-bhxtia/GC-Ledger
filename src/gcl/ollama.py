"""Talk to a local model through Ollama. Nothing leaves your computer.

Ollama runs on this machine at http://localhost:11434. We send it the prompt plus a JSON
schema ("format"), and Ollama makes the model reply with JSON in exactly that shape.

Error messages never contain the prompt or the reply, only what went wrong.
"""

from __future__ import annotations

import copy
import logging

import httpx
from pydantic import BaseModel, ValidationError

from gcl.extractor import LLMError

logger = logging.getLogger(__name__)


def strict_schema(schema: type[BaseModel]) -> dict:
    """The JSON schema of `schema`, with every field marked as required.

    Why: small local models, when forced to write JSON, tend to stop after the first list and
    skip optional fields like dates. If every field is required, they must write each one
    (even as [] or null), which makes them actually think about it.
    """
    result = copy.deepcopy(schema.model_json_schema())  # don't change pydantic's own copy

    def require_every_field(node) -> None:
        if isinstance(node, dict):
            if "properties" in node:
                node["required"] = list(node["properties"])
            for value in node.values():
                require_every_field(value)
        elif isinstance(node, list):
            for value in node:
                require_every_field(value)

    require_every_field(result)
    return result


class OllamaClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 600,
        max_attempts: int = 2,
        num_ctx: int = 16384,
        http: httpx.Client | None = None,  # the tests pass a fake one
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_attempts = max_attempts
        self.num_ctx = num_ctx
        self._http = http or httpx.Client(timeout=timeout_seconds)

    def generate_json(self, *, system: str, user: str, schema: type[BaseModel]) -> str:
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": strict_schema(schema),  # the reply must be JSON in this shape
            "stream": False,                  # one complete reply, not word by word
            "think": False,                   # no long "thinking": much faster for this job
            "options": {
                "temperature": 0,             # same input -> same answer
                "num_ctx": self.num_ctx,      # Ollama's default is small and cuts long prompts
            },
        }

        last_problem = "unknown error"
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._http.post(f"{self.base_url}/api/chat", json=request)
            except httpx.ConnectError:
                # Retrying won't help if Ollama isn't running.
                raise LLMError(
                    f"Ollama is not running at {self.base_url}. "
                    "Start the Ollama app (or run `ollama serve`) and try again."
                ) from None
            except httpx.TimeoutException:
                last_problem = "timed out"
            else:
                if response.status_code == 404:
                    # Retrying won't help if the model isn't downloaded.
                    raise LLMError(
                        f"Ollama model {self.model!r} is not installed. Run: ollama pull {self.model}"
                    ) from None

                if response.status_code != 200:
                    last_problem = f"HTTP {response.status_code}"
                else:
                    message = response.json().get("message") or {}
                    text = (message.get("content") or "").strip()
                    try:
                        schema.model_validate_json(text)  # is it really the right shape?
                        return text
                    except ValidationError:
                        last_problem = "response did not match schema"

            logger.warning("Ollama attempt %d/%d failed: %s", attempt, self.max_attempts, last_problem)

        raise LLMError(f"Ollama failed after {self.max_attempts} attempts: {last_problem}")
