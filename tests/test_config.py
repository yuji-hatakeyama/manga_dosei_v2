"""Unit tests for `manga_dosei.config` (refactor unit U1).

Covers: default values when env is unset, `.env` file loading via
pydantic-settings' `env_file`, process env precedence over `.env`,
`get_settings()` singleton caching, and SecretStr repr safety.
"""

from __future__ import annotations

import pytest

from manga_dosei.config import (
    Settings,
    get_settings,
)


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    # `get_settings` is module-level @lru_cache — clear before/after so test
    # ordering cannot leak a cached Settings built under a different env.
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_settings_uses_defaults_when_env_unset(isolate_env) -> None:
    settings = Settings()

    assert settings.gemini_text_model == "gemini-3.1-pro-preview"
    assert settings.gemini_image_model == "gemini-3-pro-image-preview"
    assert settings.openai_image_model == "gpt-image-2"

    assert settings.gemini_api_key is None
    assert settings.openai_api_key is None
    assert settings.tavily_api_key is None
    assert settings.wikimedia_contact_email is None
    assert settings.github_output_token is None


def test_settings_reads_from_dotenv_file_when_env_unset(isolate_env, tmp_path) -> None:
    # `isolate_env` already chdirs into tmp_path, so a relative `.env` here is
    # picked up by pydantic-settings' env_file.
    (tmp_path / ".env").write_text(
        "GEMINI_API_KEY=dotenv-gemini\n"
        "GEMINI_TEXT_MODEL=dotenv-text-model\n"
        "WIKIMEDIA_CONTACT_EMAIL=dotenv@example.invalid\n",
        encoding="utf-8",
    )

    settings = Settings()

    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == "dotenv-gemini"
    assert settings.gemini_text_model == "dotenv-text-model"
    assert settings.wikimedia_contact_email == "dotenv@example.invalid"


def test_process_env_overrides_dotenv_file(
    isolate_env, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        "GEMINI_API_KEY=dotenv-gemini\nGEMINI_TEXT_MODEL=dotenv-text-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "process-gemini")
    monkeypatch.setenv("GEMINI_TEXT_MODEL", "process-text-model")

    settings = Settings()

    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == "process-gemini"
    assert settings.gemini_text_model == "process-text-model"


def test_get_settings_returns_cached_singleton(isolate_env) -> None:
    first = get_settings()
    second = get_settings()

    assert first is second


def test_settings_instance_is_frozen(isolate_env) -> None:
    settings = Settings()

    # Pydantic v2 frozen models raise ValidationError (a subclass of ValueError)
    # on attribute mutation; older pydantic raises TypeError. Accept either so
    # the assertion does not pin a specific pydantic version.
    with pytest.raises((TypeError, ValueError)):
        settings.gemini_text_model = "mutated"  # type: ignore[misc]


def test_secret_str_fields_are_masked_in_repr(
    isolate_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-gemini")
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret-openai")
    monkeypatch.setenv("TAVILY_API_KEY", "super-secret-tavily")
    monkeypatch.setenv("GITHUB_OUTPUT_TOKEN", "super-secret-github")

    settings = Settings()

    rendered = repr(settings) + "\n" + str(settings)
    assert "super-secret-gemini" not in rendered
    assert "super-secret-openai" not in rendered
    assert "super-secret-tavily" not in rendered
    assert "super-secret-github" not in rendered

    # Sanity: the raw value is still accessible through the documented API.
    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == "super-secret-gemini"


def test_extra_env_vars_are_ignored(isolate_env, monkeypatch: pytest.MonkeyPatch) -> None:
    # `extra="ignore"` — unrelated env vars must not raise.
    monkeypatch.setenv("UNRELATED_VAR", "value")

    Settings()


def test_publisher_settings_ignores_cwd_dotenv(isolate_env, tmp_path) -> None:
    # NOTE: the publisher contract (AGENTS.md) requires that
    # `manga_dosei-publish` read `GITHUB_OUTPUT_TOKEN` only from the process
    # env — never from a `.env` in the CWD. `isolate_env` chdirs into
    # `tmp_path`, so a `.env` written here is what `get_settings()` would
    # pick up; `Settings(_env_file=None)` (publisher path) must not.
    (tmp_path / ".env").write_text(
        "GITHUB_OUTPUT_TOKEN=stray-dotenv-token\nGEMINI_API_KEY=stray-gemini\n",
        encoding="utf-8",
    )

    publish_settings = Settings(_env_file=None)
    assert publish_settings.github_output_token is None
    assert publish_settings.gemini_api_key is None

    # Sanity: the pipeline's `get_settings()` *does* still read the same `.env`,
    # so the test isolation is asserting the divergence, not a no-op.
    pipeline_settings = get_settings()
    assert pipeline_settings.github_output_token is not None
    assert (
        pipeline_settings.github_output_token.get_secret_value() == "stray-dotenv-token"
    )


def test_publisher_settings_still_reads_process_env(
    isolate_env, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Disabling dotenv must not break the documented "shell env" path —
    # exporting the token in the process env is the supported way to feed it.
    (tmp_path / ".env").write_text(
        "GITHUB_OUTPUT_TOKEN=stray-dotenv-token\n", encoding="utf-8"
    )
    monkeypatch.setenv("GITHUB_OUTPUT_TOKEN", "process-token")

    publish_settings = Settings(_env_file=None)

    assert publish_settings.github_output_token is not None
    assert publish_settings.github_output_token.get_secret_value() == "process-token"
