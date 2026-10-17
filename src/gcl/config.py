"""Central configuration. All settings come from environment variables / .env."""

from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GCL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM (Gemini). Optional here so parsing/ledger commands work without a key.
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash"
    llm_timeout_seconds: int = Field(default=60, gt=0, le=600)

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

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, value: str) -> str:
        value = value.upper()
        if value not in _LOG_LEVELS:
            raise ValueError(f"log_level must be one of {sorted(_LOG_LEVELS)}")
        return value

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