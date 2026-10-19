"""Central configuration. All settings come from environment variables / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_THINKING_LEVELS = {"MINIMAL", "LOW", "MEDIUM", "HIGH"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GCL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Which LLM does the reading: "gemini" (Google API) or "ollama" (local, nothing leaves the machine).
    llm_provider: Literal["gemini", "ollama"] = "gemini"

    # Ollama (local)
    ollama_model: str = "qwen3:8b"
    ollama_url: str = "http://localhost:11434"
    ollama_timeout_seconds: int = Field(default=600, gt=0, le=3600)  # local models are slower
    ollama_num_ctx: int = Field(default=16384, ge=2048, le=131072)  # context window in tokens

    # LLM (Gemini). Optional here so parsing/ledger commands work without a key.
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.8-flash"
    # Sent to Google as the request deadline, so it must cover the model's thinking time.
    llm_timeout_seconds: int = Field(default=120, gt=0, le=600)
    llm_max_attempts: int = Field(default=3, ge=1, le=10)
    # LOW keeps extraction fast; empty means the model's default (for models without it).
    gemini_thinking_level: str | None = "LOW"

    # Storage
    db_path: Path = Path("data/gcl.db")

    # Parsing / dates
    default_timezone: str = "Asia/Kolkata"
    max_import_mb: int = Field(default=10, gt=0, le=100)

    log_level: str = "INFO"

    @field_validator("default_timezone")
    @classmethod
    def _valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"Unknown timezone: {value!r}") from exc
        return value

    @field_validator("gemini_thinking_level")
    @classmethod
    def _valid_thinking_level(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        value = value.strip().upper()
        if value not in _THINKING_LEVELS:
            raise ValueError(f"gemini_thinking_level must be one of {sorted(_THINKING_LEVELS)} or empty")
        return value

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, value: str) -> str:
        value = value.upper()
        if value not in _LOG_LEVELS:
            raise ValueError(f"log_level must be one of {sorted(_LOG_LEVELS)}")
        return value

    @property
    def is_local_llm(self) -> bool:
        return self.llm_provider == "ollama"

    @property
    def llm_label(self) -> str:
        if self.is_local_llm:
            return f"Ollama on this computer ({self.ollama_model})"
        return f"Google Gemini ({self.gemini_model})"

    @property
    def qa_max_chars(self) -> int | None:
        """Prompt budget for Q&A: a local model gets what fits its context (~3 chars per token)."""
        return self.ollama_num_ctx * 3 - 6000 if self.is_local_llm else None

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.default_timezone)

    @property
    def max_import_bytes(self) -> int:
        return self.max_import_mb * 1024 * 1024

    def require_api_key(self) -> str:
        """Return the Gemini key, or fail loudly. Call only where the LLM is needed."""
        if self.gemini_api_key is None or not self.gemini_api_key.get_secret_value().strip():
            raise RuntimeError(
                "GCL_GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return self.gemini_api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    return Settings()