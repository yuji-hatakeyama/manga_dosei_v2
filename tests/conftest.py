"""Shared pytest fixtures for the manga_dosei test suite.

Keeps tests hermetic: pre-seeds placeholder API keys *before* importing
`manga_dosei` (its top-level `load_dotenv(..., override=False)` only fills
missing keys, so placeholders block the repo `.env` from leaking developer
secrets into `os.environ`), and provides a lightweight ToolContext stub so
direct-tool tests do not need a real ADK runner.
"""

from __future__ import annotations

import os
import types
from collections.abc import Callable
from typing import Any

import pytest

# NOTE: pre-import scrub. `manga_dosei/__init__.py` calls
# `load_dotenv(repo/.env, override=False)` at import time; tool modules then
# snapshot `get_settings().*_model` at *their* import time
# (e.g. `model=get_settings().gemini_text_model` at module top of every
# LlmAgent-backed tool). If we wait for `safe_env` to scrub env per-test,
# those snapshots have already captured the developer-local values from the
# repo `.env`. Two-step defense before importing `manga_dosei` below:
#   1. Seed placeholders for credential keys so `load_dotenv(override=False)`
#      finds them already set and skips the repo `.env` value.
#   2. Drop any model-override env vars that *might* be inherited from the
#      developer's shell or repo `.env`, so tools snapshot the in-repo
#      `Settings` field defaults rather than a locally-pinned preview model.
_PRE_IMPORT_PLACEHOLDERS = {
    "GEMINI_API_KEY": "test-gemini-key",
    "OPENAI_API_KEY": "test-openai-key",
    "TAVILY_API_KEY": "test-tavily-key",
    "WIKIMEDIA_CONTACT_EMAIL": "test@example.invalid",
}
for _name, _placeholder in _PRE_IMPORT_PLACEHOLDERS.items():
    os.environ.setdefault(_name, _placeholder)
for _name in ("GEMINI_TEXT_MODEL", "GEMINI_IMAGE_MODEL", "OPENAI_IMAGE_MODEL"):
    os.environ.pop(_name, None)

from manga_dosei.config import Settings, get_settings  # noqa: E402

# Derived from Settings so adding a new env var to `Settings` automatically
# extends the scrub list — single source of truth for env-var aliases.
_ENV_VARS = tuple(f.alias or name for name, f in Settings.model_fields.items() if f.alias)


@pytest.fixture(autouse=True)
def safe_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Isolate tests from real credentials and the repo `.env`.

    Scrubs every env var the package reads, then sets placeholder values for
    the keys that are required at *import time* (Tavily / Wikimedia) or that
    tools commonly assert on before doing real I/O. Also chdirs into tmp_path
    so test code that resolves paths relative to `cwd` lands in a clean dir.

    NOTE: chdir does *not* defeat the repo `.env`. `manga_dosei/__init__.py`
    resolves the dotenv path from `__file__` (absolute) and loads it once at
    package import time, long before this fixture runs. The pre-import scrub
    at the top of this file is what actually blocks developer-local secrets
    from leaking into `os.environ` for the test session.

    `get_settings` is module-level `@lru_cache` — clear before/after so a
    Settings instance built under the developer-local repo `.env` cannot
    leak into `get_settings()` calls inside the test body, and so per-test
    env overrides via `monkeypatch.setenv` are not silently ignored.
    NOTE: tool modules capture `get_settings().*_model` at *import time*
    (e.g. `LlmAgent(model=...)`), so cache_clear here does not retroactively
    rewrite already-constructed agents — tests that need to vary those
    values must reimport the tool module after setting env.
    """
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    monkeypatch.setenv("WIKIMEDIA_CONTACT_EMAIL", "test@example.invalid")

    monkeypatch.chdir(tmp_path)

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Stricter variant of `safe_env` that leaves *all* env vars unset.

    Use in tests that exercise "missing env" error paths.
    """
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def sample_target_date() -> str:
    return "20260101"


@pytest.fixture
def tmp_artifacts_dir(tmp_path):
    """Directory layout matching `.adk/artifacts/` for FileArtifactService tests."""
    path = tmp_path / ".adk" / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


class _StubState(dict):
    """`dict` subclass mirroring the subset of ADK `State` tools actually call.

    `inspect_artifacts` invokes `state.to_dict()`; bare `dict` lacks it.
    """

    def to_dict(self) -> dict[str, Any]:
        return dict(self)


@pytest.fixture
def stub_tool_context() -> Callable[..., types.SimpleNamespace]:
    """Factory for a minimal ADK ToolContext stand-in.

    Returns a `SimpleNamespace` exposing:
      - `state`: `_StubState` (dict subclass with `to_dict()`)
      - `list_artifacts() -> list[str]`
      - `load_artifact(filename) -> google.genai.types.Part | None`
      - `save_artifact(filename, artifact) -> int` (version number)

    Artifacts live in an in-memory dict so tests can pre-seed inputs and
    assert on what tools wrote, without spinning up `FileArtifactService`.
    """

    def _factory(
        *,
        initial_state: dict[str, Any] | None = None,
        initial_artifacts: dict[str, Any] | None = None,
    ) -> types.SimpleNamespace:
        store: dict[str, list[Any]] = {
            name: [artifact] for name, artifact in (initial_artifacts or {}).items()
        }

        async def list_artifacts() -> list[str]:
            return sorted(store.keys())

        async def load_artifact(filename: str, version: int | None = None):
            versions = store.get(filename)
            if not versions:
                return None
            if version is None:
                return versions[-1]
            return versions[version]

        async def save_artifact(filename: str, artifact: Any) -> int:
            versions = store.setdefault(filename, [])
            versions.append(artifact)
            return len(versions) - 1

        return types.SimpleNamespace(
            state=_StubState(initial_state or {}),
            list_artifacts=list_artifacts,
            load_artifact=load_artifact,
            save_artifact=save_artifact,
            _artifact_store=store,
        )

    return _factory
