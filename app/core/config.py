"""Service configuration loaded from environment variables / .env file."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

DRAFT_ONLY_WARNING = (
    "Draft-only Braille OCR. All output is an unverified draft and must be "
    "checked by a QTVI or Braille-literate specialist before any use in "
    "teacher feedback or export. This engine never claims certified Braille "
    "accuracy."
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    service_name: str = "insighted-braille-ocr-engine"
    service_version: str = "0.4.0"
    api_version: str = "v1"
    draft_warning: str = DRAFT_ONLY_WARNING

    # Optional API key. When set, POST /ocr requires the X-API-Key header.
    ocr_engine_api_key: str | None = None

    max_image_mb: float = 10.0
    max_image_pixels: int = 40_000_000

    log_level: str = "INFO"

    liblouis_enabled: bool = True
    liblouis_table: str = "en-ueb-g1.ctb"

    @property
    def max_image_bytes(self) -> int:
        return int(self.max_image_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
