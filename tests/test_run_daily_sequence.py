"""Unit tests for the `run_daily.STEPS` sequencing table (refactor unit U11).

Locks the canonical step order documented in AGENTS.md so the deterministic
CLI sequence cannot drift independently of the spec (the same order is also
encoded in `root_agent`'s instruction string).
"""

from __future__ import annotations

import inspect

import pytest
from google.genai import types as genai_types

from manga_dosei.run_daily import (
    _DIRECT_TOOLS,
    STEPS,
    StepSpec,
    _extract_part_bytes,
    _validate_session_id,
)
from manga_dosei.tools.generate_page_gemini import (
    PAGE_VARIANT_COUNT as GEMINI_PAGE_VARIANT_COUNT,
)


def test_steps_match_canonical_order_from_agents_md() -> None:
    expected = [
        "fetch_dosei",
        "enrich_news",
        "generate_scenario",
        "collect_assets",
        "resize_assets",
        "define_layout",
        "compose_image_brief",
        *(["generate_page_gemini"] * GEMINI_PAGE_VARIANT_COUNT),
    ]
    assert [step.tool_name for step in STEPS] == expected


def test_page_steps_carry_one_based_page_number() -> None:
    page_steps = [s for s in STEPS if s.tool_name == "generate_page_gemini"]
    assert [s.extra_args for s in page_steps] == [
        {"page_number": n} for n in range(1, GEMINI_PAGE_VARIANT_COUNT + 1)
    ]


def test_non_page_steps_have_no_extra_args() -> None:
    for step in STEPS:
        if step.tool_name == "generate_page_gemini":
            continue
        assert step.extra_args == {}, step


def test_every_step_is_a_stepspec() -> None:
    # Guard the typing: a stray bare tuple would defeat the structural rename.
    for step in STEPS:
        assert isinstance(step, StepSpec)


def test_no_step_is_marked_retry_exempt_by_default() -> None:
    # `RETRY_EXEMPT` was empty before U11 — the default rollover must preserve
    # that so retry semantics do not silently change.
    assert all(step.retry_exempt is False for step in STEPS)


def test_direct_tools_share_one_signature() -> None:
    # Direct tools are normalised to `(target_date, tool_context)` so the
    # dispatcher has a single call shape — no boolean discriminator, no
    # untyped callable. Lock the table's shape and the signature.
    assert set(_DIRECT_TOOLS) == {"inspect_artifacts", "resize_assets"}
    for name, fn in _DIRECT_TOOLS.items():
        assert callable(fn), name
        params = list(inspect.signature(fn).parameters)
        assert params[:2] == ["target_date", "tool_context"], (name, params)


def test_resize_assets_is_in_both_steps_and_direct_tools() -> None:
    # resize_assets is sequenced by the CLI (STEPS) and dispatched directly
    # (no LLM round-trip) via _DIRECT_TOOLS. Both contracts must hold.
    assert "resize_assets" in {step.tool_name for step in STEPS}
    assert "resize_assets" in _DIRECT_TOOLS


@pytest.mark.parametrize(
    "suffix",
    [
        "_../escape",
        "_a/b",
        "_a\\b",
        "_ ",
        "_\t",
        "_a.b",
    ],
)
def test_session_id_rejects_unsafe_suffix_chars(suffix: str) -> None:
    # session_id is propagated to the FileArtifactService root as a directory
    # component, so the regex must restrict the suffix to a filesystem-safe
    # alphabet rather than ".+".
    with pytest.raises(SystemExit):
        _validate_session_id(f"20260315{suffix}", "20260315")


def test_session_id_accepts_safe_suffix() -> None:
    _validate_session_id("20260315", "20260315")
    _validate_session_id("20260315_retry", "20260315")
    _validate_session_id("20260315_retry-2", "20260315")
    _validate_session_id("20260315_a_b_c", "20260315")


def test_extract_part_bytes_returns_inline_data_when_present() -> None:
    part = genai_types.Part(
        inline_data=genai_types.Blob(data=b"raw bytes", mime_type="image/png")
    )
    assert _extract_part_bytes(part, name="x.png") == b"raw bytes"


def test_extract_part_bytes_falls_back_to_text_when_inline_data_empty() -> None:
    # F4 regression: ADK can round-trip a text artifact with an empty
    # inline_data Blob and the real payload in `part.text`. Falling through to
    # the text branch prevents `--publish-dir` from silently dropping the file.
    part = genai_types.Part(
        text="hello world",
        inline_data=genai_types.Blob(data=b"", mime_type=""),
    )
    assert _extract_part_bytes(part, name="scenario.md") == b"hello world"


def test_main_loads_env_before_resolving_settings(monkeypatch) -> None:
    # F6 regression: main() must reload `.env` and invalidate the get_settings
    # lru_cache so a per-run CWD override (CI rotation, debug) lands in
    # Settings before the first tool call. Verify by setting an env var via
    # monkeypatch.setenv after get_settings has been cached.
    from manga_dosei.config import Settings, get_settings
    from manga_dosei.run_daily import main

    # Pre-cache get_settings with the current GEMINI_TEXT_MODEL absence.
    monkeypatch.delenv("GEMINI_TEXT_MODEL", raising=False)
    get_settings.cache_clear()
    cached = get_settings()
    assert cached.gemini_text_model == Settings.model_fields["gemini_text_model"].default

    # Simulate a shell-exported override and a `--publish-dir` not present.
    monkeypatch.setenv("GEMINI_TEXT_MODEL", "override-from-shell")
    monkeypatch.setattr(
        "sys.argv",
        ["manga_dosei", "20260101"],
    )

    # Short-circuit asyncio.run so we don't actually drive the pipeline;
    # close the coroutine to silence the unawaited-coroutine warning.
    def _consume(coro):
        coro.close()

    monkeypatch.setattr("manga_dosei.run_daily.asyncio.run", _consume)

    main()

    # After main(), get_settings() must reflect the shell-exported override.
    assert get_settings().gemini_text_model == "override-from-shell"
