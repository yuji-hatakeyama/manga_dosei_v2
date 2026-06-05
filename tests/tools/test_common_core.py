"""Unit tests for `manga_dosei.tools._common` pure-core helpers (refactor unit U6).

Locks the contract of the CallbackContext-free helpers extracted out of
`save_step_output` / `prepare_step` so the surrounding ADK-callback wiring
keeps the same observable behavior (and `state["last_error"]` stays stable)
after the pure/IO split.
"""

from __future__ import annotations

from manga_dosei.tools._common import (
    StepOutput,
    decide_missing,
    parse_step_output,
)


def test_parse_step_output_returns_body_when_error_is_empty() -> None:
    assert parse_step_output({"body": "x", "error": ""}) == ("x", None)


def test_parse_step_output_returns_error_when_body_is_empty() -> None:
    assert parse_step_output({"body": "", "error": "boom"}) == (None, "boom")


def test_parse_step_output_prefers_error_when_both_present() -> None:
    # Ambiguous LLM responses (body AND error set) must not silently succeed —
    # error wins so save_step_output records the failure instead of saving stale text.
    assert parse_step_output({"body": "x", "error": "boom"}) == (None, "boom")


def test_parse_step_output_accepts_raw_string() -> None:
    assert parse_step_output("raw string") == ("raw string", None)


def test_parse_step_output_treats_none_as_error() -> None:
    body, error = parse_step_output(None)
    assert body is None
    assert error  # non-empty error message — exact wording is implementation detail


def test_parse_step_output_accepts_step_output_instance() -> None:
    assert parse_step_output(StepOutput(body="hello", error="")) == ("hello", None)


def test_parse_step_output_step_output_error_instance() -> None:
    assert parse_step_output(StepOutput(body="", error="nope")) == (None, "nope")


def test_parse_step_output_strips_whitespace_only_body() -> None:
    # Whitespace-only body is treated as empty (matches existing rstrip behavior).
    body, error = parse_step_output({"body": "   \n", "error": ""})
    assert body is None
    assert error  # falls back to "agent reported failure" sentinel


def test_parse_step_output_empty_dict_returns_failure_sentinel() -> None:
    body, error = parse_step_output({})
    assert body is None
    assert error


def test_parse_step_output_unexpected_type_returns_error() -> None:
    body, error = parse_step_output(123)  # type: ignore[arg-type]
    assert body is None
    assert error


def test_decide_missing_returns_names_absent_from_available() -> None:
    missing = decide_missing(("dosei.md", "news.md"), {"dosei.md"})
    assert missing == ["news.md"]


def test_decide_missing_preserves_required_order() -> None:
    # Order must match `required` (callers surface this list to the LLM /
    # last_error, and a stable order keeps diffs and logs deterministic).
    missing = decide_missing(("a", "b", "c", "d"), {"b"})
    assert missing == ["a", "c", "d"]


def test_decide_missing_returns_empty_when_all_available() -> None:
    assert decide_missing(("dosei.md",), {"dosei.md", "news.md"}) == []


def test_decide_missing_accepts_list_available() -> None:
    # `list_artifact_keys` returns set, but accept any iterable so callers
    # don't have to pre-convert.
    assert decide_missing(("a", "b"), ["a"]) == ["b"]


def test_decide_missing_with_empty_required_returns_empty() -> None:
    assert decide_missing((), {"a"}) == []
