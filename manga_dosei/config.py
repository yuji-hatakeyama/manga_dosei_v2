"""Single source of truth for env-driven configuration.

All tools go through `get_settings()`. New code must use `get_settings()`
instead of touching `os.environ` directly.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# NOTE: `env_file=".env"` is intentionally *relative* for the pipeline path.
# The package also calls `load_dotenv(<repo>/.env)` at import time, but that
# runs once with `override=False`, so shell env > dotenv-loaded env > pydantic
# env_file. Resolving relative to the CWD lets tests `monkeypatch.chdir(tmp_path)`
# to avoid picking up the real repo `.env`.
#
# `manga_dosei-publish` (publish.py) intentionally bypasses this dotenv lookup
# by instantiating `Settings(_env_file=None)` so the publisher only ever reads
# from the process env. The publisher is launched from arbitrary CI / dev CWDs,
# and a relative `.env` lookup would otherwise let a stray `.env` (publish-dir
# mounting a dev home, CI cache, sibling project, ...) inject
# `GITHUB_OUTPUT_TOKEN`. See AGENTS.md publisher section.


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
        frozen=True,
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # NOTE: secrets are SecretStr so `repr(settings)` / logging / exception
    # chains print `**********` instead of the live value. Call sites that
    # need the raw string use `.get_secret_value()`.
    gemini_api_key: SecretStr | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_text_model: str = Field(
        default="gemini-3.1-pro-preview", alias="GEMINI_TEXT_MODEL"
    )
    gemini_image_model: str = Field(
        default="gemini-3-pro-image-preview", alias="GEMINI_IMAGE_MODEL"
    )

    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_image_model: str = Field(default="gpt-image-2", alias="OPENAI_IMAGE_MODEL")

    tavily_api_key: SecretStr | None = Field(default=None, alias="TAVILY_API_KEY")
    wikimedia_contact_email: str | None = Field(
        default=None, alias="WIKIMEDIA_CONTACT_EMAIL"
    )

    github_output_token: SecretStr | None = Field(
        default=None, alias="GITHUB_OUTPUT_TOKEN"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
