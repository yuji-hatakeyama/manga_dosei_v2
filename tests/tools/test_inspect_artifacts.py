"""Unit tests for `inspect_artifacts` (refactor U12).

The tool is a plain `async` FunctionTool — not an `LlmAgent`-wrapped step —
so these tests drive it directly with a stub `ToolContext`. Targets:

  - malformed `target_date` returns an error result dict.
  - the success path reads available artifacts via `tool_context.list_artifacts`.
  - the success-path payload matches the ok_result shape.
"""

from __future__ import annotations

import asyncio

from manga_dosei.tools.inspect_artifacts import inspect_artifacts


def test_inspect_artifacts_invalid_target_date_preserves_prior_last_error(
    stub_tool_context,
) -> None:
    # Regression: a typoed target_date must not overwrite an upstream step's
    # last_error (F1). The CLI retry path keys off the prior failure.
    ctx = stub_tool_context(
        initial_state={"last_error": {"step": "fetch_dosei", "message": "boom"}},
    )
    result = asyncio.run(inspect_artifacts("bad-date", ctx))

    assert result["status"] == "error"
    assert ctx.state["last_error"] == {"step": "fetch_dosei", "message": "boom"}


def test_inspect_artifacts_reads_available_artifacts_from_tool_context(
    stub_tool_context,
) -> None:
    # The tool must surface whatever `tool_context.list_artifacts()` returns,
    # sorted — that is the whole point of the "inspection" step.
    ctx = stub_tool_context(
        initial_artifacts={"news.md": "x", "dosei.md": "y", "scenario.md": "z"},
    )

    result = asyncio.run(inspect_artifacts("20260101", ctx))

    assert result["status"] == "success"
    assert result["artifacts"] == ["dosei.md", "news.md", "scenario.md"]


def test_inspect_artifacts_success_path_matches_step_result_shape(
    stub_tool_context,
) -> None:
    # Success payload is ok_result ("success"/step/message) plus the
    # inspection-specific fields target_date/artifacts/state.
    ctx = stub_tool_context(
        initial_state={"session_id": "20260101"},
        initial_artifacts={"dosei.md": "x"},
    )

    result = asyncio.run(inspect_artifacts("20260101", ctx))

    assert result["status"] == "success"
    assert result["step"] == "inspect_artifacts"
    # Message wording is informational; only assert presence to avoid brittle
    # exact-string coupling (Q4).
    assert isinstance(result["message"], str) and result["message"]
    assert result["target_date"] == "20260101"
    assert result["artifacts"] == ["dosei.md"]
    assert result["state"] == {"session_id": "20260101"}


def test_inspect_artifacts_excludes_temp_state_keys(
    stub_tool_context,
) -> None:
    # `temp:` keys are ADK-ephemeral and showing them as persistent state
    # would mislead the LLM into thinking they survive across turns.
    ctx = stub_tool_context(
        initial_state={"temp:target_date": "20260101", "persistent_key": "kept"},
    )

    result = asyncio.run(inspect_artifacts("20260101", ctx))

    assert result["state"] == {"persistent_key": "kept"}


def test_inspect_artifacts_does_not_touch_last_error_on_success(
    stub_tool_context,
) -> None:
    # Read-only inspection must not clobber a prior step's last_error — the
    # CLI relies on it to decide retry/abort for the *previous* failing step.
    ctx = stub_tool_context(
        initial_state={"last_error": {"step": "prev", "message": "stale"}},
    )

    result = asyncio.run(inspect_artifacts("20260101", ctx))

    assert result["status"] == "success"
    assert ctx.state["last_error"] == {"step": "prev", "message": "stale"}
