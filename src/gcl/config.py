"""All settings, read from environment variables or the .env file.

Every setting has the prefix GCL_. For example, `ollama_model` below is set with
GCL_OLLAMA_MODEL=qwen3:8b in .env. Anything not set uses the default written here.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
THINKING_LEVELS = {"MINIMAL", "LOW", "MEDIUM", "HIGH"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GCL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",   # unknown GCL_ variables are ignored, not an error
    )

    # --- Which model reads the chat ---------------------------------------------
    # "gemini" = Google's API.  "ollama" = a model on this computer (nothing leaves it).
    llm_provider: Literal["gemini", "ollama"] = "gemini"

    # --- Ollama (local) ----------------------------------------------------------
    ollama_model: str = "qwen3:8b"
    ollama_url: str = "http://localhost:11434"
    ollama_timeout_seconds: int = Field(default=600, gt=0, le=3600)  # local models are slow
    ollama_num_ctx: int = Field(default=16384, ge=2048, le=131072)  # how many tokens it can read

    # --- Gemini (Google) ---------------------------------------------------------
    gemini_api_key: SecretStr | None = None  # only needed when llm_provider is "gemini"
    gemini_model: str = "gemini-3.8-flash"
    # Google uses this as the deadline for the request, so it must cover thinking time.
    llm_timeout_seconds: int = Field(default=120, gt=0, le=600)
    llm_max_attempts: int = Field(default=3, ge=1, le=10)
    # LOW keeps extraction fast. Empty = the model's own default (some models need that).
    gemini_thinking_level: str | None = "LOW"

    # --- Storage -----------------------------------------------------------------
    db_path: Path = Path("data/gcl.db")

    # --- Reading chats -----------------------------------------------------------
    default_timezone: str = "Asia/Kolkata"  # used for chat times and words like "kal"
    max_import_mb: int = Field(default=10, gt=0, le=100)

    log_level: str = "INFO"

    # --- Checks on the values ------------------------------------------------------

    @field_validator("default_timezone")
    @classmethod
    def _check_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError(f"Unknown timezone: {value!r}") from error
        return value

    @field_validator("gemini_thinking_level")
    @classmethod
    def _check_thinking_level(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None  # empty means "use the model's default"
        value = value.strip().upper()
        if value not in THINKING_LEVELS:
            raise ValueError(f"gemini_thinking_level must be one of {sorted(THINKING_LEVELS)} or empty")
        return value

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, value: str) -> str:
        value = value.upper()
        if value not in LOG_LEVELS:
            raise ValueError(f"log_level must be one of {sorted(LOG_LEVELS)}")
        return value

    # --- Handy values worked out from the settings ---------------------------------

    @property
    def is_local_llm(self) -> bool:
        return self.llm_provider == "ollama"

    @property
    def llm_label(self) -> str:
        """A name for the model, to show to the user."""
        if self.is_local_llm:
            return f"Ollama on this computer ({self.ollama_model})"
        return f"Google Gemini ({self.gemini_model})"

    @property
    def qa_max_chars(self) -> int | None:
        """How much chat text a question may send. A local model gets what fits its context
        (about 3 characters per token, minus room for the instructions and the answer).
        Gemini: None, meaning no character limit."""
        if self.is_local_llm:
            return self.ollama_num_ctx * 3 - 6000
        return None

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.default_timezone)

    @property
    def max_import_bytes(self) -> int:
        return self.max_import_mb * 1024 * 1024

    def require_api_key(self) -> str:
        """The Gemini key, or a clear error if it isn't set."""
        key = self.gemini_api_key
        if key is None or not key.get_secret_value().strip():
            raise RuntimeError(
                "GCL_GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    """The settings, read once and then reused."""
    return Settings()
