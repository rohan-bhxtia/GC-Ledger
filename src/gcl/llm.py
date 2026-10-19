"""Pick the LLM backend from settings (GCL_LLM_PROVIDER = gemini | ollama)."""

from __future__ import annotations

from gcl.config import Settings, get_settings
from gcl.extractor import LLMClient


def client_from_settings(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    if settings.llm_provider == "ollama":
        from gcl.ollama import OllamaClient

        return OllamaClient(
            model=settings.ollama_model,
            base_url=settings.ollama_url,
            timeout_seconds=settings.ollama_timeout_seconds,
            num_ctx=settings.ollama_num_ctx,
        )
    from gcl.gemini import client_from_settings as gemini_client

    return gemini_client(settings)
