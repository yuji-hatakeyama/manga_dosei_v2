"""Public-surface tests for the top-level `manga_dosei` package.

Covers refactor unit U13: the documented public exports
(`get_settings`, `Settings`, `ArtifactName`, `StateKey`) plus the
back-compat aliases (`APP_NAME`, `DEFAULT_USER_ID`) must all be
importable. Importing the package must also work with no
API keys set in the environment, because plain `import manga_dosei`
runs before any tool is exercised.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

# Subset of env vars we want every subprocess to start without, mirroring
# the scrub list `conftest.py::safe_env` builds from `Settings.model_fields`.
# Hard-coded here (not derived) so the subprocess does not need to import
# manga_dosei just to compute the scrub list — that would defeat the point
# of running module-level side effects (load_dotenv, etc.) in a clean env.
_SCRUB_ENV_VARS = (
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "TAVILY_API_KEY",
    "WIKIMEDIA_CONTACT_EMAIL",
    "GEMINI_TEXT_MODEL",
    "GEMINI_IMAGE_MODEL",
    "OPENAI_IMAGE_MODEL",
)


def _run_in_subprocess(
    body: str,
    *,
    extra_env: dict[str, str] | None = None,
    stub_dotenv: bool = False,
) -> dict:
    """Run `body` in a fresh interpreter and return the JSON it prints.

    A subprocess is the only reliable way to re-trigger module-level side
    effects (`load_dotenv`, `lru_cache` priming, downstream tool modules'
    `from manga_dosei.config import get_settings` captures) without leaving
    dangling old-module references in sibling tests' already-imported
    `manga_dosei.*` modules.

    `body` must end by printing one JSON object on a line prefixed with
    `RESULT:` so we can pluck it out of stdout even if the package emits
    its own logging during import.
    """
    env = {k: v for k, v in os.environ.items() if k not in _SCRUB_ENV_VARS}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # chdir into a tmp-ish path: use the repo root's parent so python-dotenv
    # cannot find the repo `.env`. The test runner's cwd is unpredictable;
    # explicit cwd via subprocess avoids leaking the developer-local `.env`.
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
    cwd = os.path.dirname(cwd)  # one level above repo root

    prelude = ""
    if stub_dotenv:
        # Stub `load_dotenv` only — pydantic_settings imports `dotenv_values`
        # from the same module, so replacing the whole module breaks it.
        # Patching the attribute in place leaves the rest of `dotenv` intact.
        prelude = textwrap.dedent(
            """
            import dotenv
            dotenv.load_dotenv = lambda *a, **kw: False
            """
        )
    if extra_env:
        env.update(extra_env)

    script = prelude + body
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"subprocess failed (rc={proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line[len("RESULT:") :])
    raise AssertionError(
        f"subprocess produced no RESULT line\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )


def test_package_exports_documented_public_symbols():
    result = _run_in_subprocess(
        textwrap.dedent(
            """
            import json
            import manga_dosei as pkg
            from manga_dosei import ArtifactName, Settings, StateKey, get_settings
            print("RESULT:" + json.dumps({
                "get_settings_is_pkg": get_settings is pkg.get_settings,
                "Settings_is_pkg": Settings is pkg.Settings,
                "ArtifactName_is_pkg": ArtifactName is pkg.ArtifactName,
                "StateKey_is_pkg": StateKey is pkg.StateKey,
            }))
            """
        ),
    )
    assert result["get_settings_is_pkg"]
    assert result["Settings_is_pkg"]
    assert result["ArtifactName_is_pkg"]
    assert result["StateKey_is_pkg"]


def test_package_exports_back_compat_aliases():
    result = _run_in_subprocess(
        textwrap.dedent(
            """
            import json
            from manga_dosei import APP_NAME, DEFAULT_USER_ID
            print("RESULT:" + json.dumps({
                "APP_NAME": APP_NAME,
                "DEFAULT_USER_ID": DEFAULT_USER_ID,
            }))
            """
        ),
    )
    assert result["APP_NAME"] == "manga_dosei"
    assert result["DEFAULT_USER_ID"] == "daily"


def test_dunder_all_lists_only_resolvable_names():
    result = _run_in_subprocess(
        textwrap.dedent(
            """
            import json
            import manga_dosei as pkg
            missing = [name for name in pkg.__all__ if not hasattr(pkg, name)]
            print("RESULT:" + json.dumps({"missing": missing, "all": list(pkg.__all__)}))
            """
        ),
    )
    assert result["missing"] == [], (
        f"__all__ advertises unresolvable names: {result['missing']!r}"
    )


def test_import_does_not_require_api_keys():
    """Bare `import manga_dosei` must not raise when no API keys are set.

    Runs in a fresh interpreter with the package's env vars scrubbed and
    `dotenv.load_dotenv` stubbed *before* manga_dosei is imported, so the
    repo's real `.env` cannot sneak credentials back in via the package's
    import-time `load_dotenv(...)` call.

    Tool submodules require keys at *their* import time, but the top-level
    package itself must stay import-safe so CLIs that only need
    `get_settings()` / names can load.
    """
    result = _run_in_subprocess(
        textwrap.dedent(
            """
            import json
            import manga_dosei as pkg
            print("RESULT:" + json.dumps({
                "get_settings_not_none": pkg.get_settings is not None,
                "DOSEI": pkg.ArtifactName.DOSEI,
                "TARGET_DATE": pkg.StateKey.TARGET_DATE,
            }))
            """
        ),
        stub_dotenv=True,
    )
    assert result["get_settings_not_none"]
    assert result["DOSEI"] == "dosei.md"
    assert result["TARGET_DATE"] == "target_date"
