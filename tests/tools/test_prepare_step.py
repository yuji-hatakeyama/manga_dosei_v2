"""Regression tests for `prepare_step` wiring-vs-step-failure distinction (F3).

When AgentTool input plumbing fails (missing/invalid target_date), that is a
wiring problem — not a workflow-step failure — so `state["last_error"]` must
stay intact to preserve the upstream step's failure context for the CLI's
retry/abort decision. Missing required artifacts is still a step-level
failure and DOES populate last_error.
"""

from __future__ import annotations

import asyncio
import json
import types as _types
from typing import Any

from google.genai import types as genai_types

from manga_dosei.tools._common import prepare_step, save_step_output


class _StubState(dict):
    def to_dict(self) -> dict[str, Any]:
        return dict(self)


def _stub_callback_context(
    *,
    user_text: str | None,
    initial_state: dict[str, Any] | None = None,
    initial_artifacts: tuple[str, ...] = (),
) -> _types.SimpleNamespace:
    user_content: genai_types.Content | None
    if user_text is None:
        user_content = None
    else:
        user_content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_text)],
        )

    async def list_artifacts() -> list[str]:
        return list(initial_artifacts)

    async def load_artifact(filename: str, version: int | None = None):
        return None

    return _types.SimpleNamespace(
        user_content=user_content,
        state=_StubState(initial_state or {}),
        list_artifacts=list_artifacts,
        load_artifact=load_artifact,
    )


def test_prepare_step_missing_user_content_does_not_touch_last_error() -> None:
    ctx = _stub_callback_context(
        user_text=None,
        initial_state={"last_error": {"step": "prev", "message": "stale"}},
    )
    result = asyncio.run(prepare_step(ctx, step="fetch_dosei"))
    assert result is not None  # error_content returned
    assert ctx.state["last_error"] == {"step": "prev", "message": "stale"}


def test_prepare_step_invalid_target_date_does_not_touch_last_error() -> None:
    ctx = _stub_callback_context(
        user_text='{"target_date": "bad-date"}',
        initial_state={"last_error": {"step": "prev", "message": "stale"}},
    )
    result = asyncio.run(prepare_step(ctx, step="fetch_dosei"))
    assert result is not None
    assert ctx.state["last_error"] == {"step": "prev", "message": "stale"}


def test_prepare_step_validation_error_text_propagates_to_error_content() -> None:
    # Malformed JSON triggers the parse_target_date_input failure path; the
    # detail message rides error_content even though last_error is not touched.
    ctx = _stub_callback_context(
        user_text="not json at all",
        initial_state={},
    )
    result = asyncio.run(prepare_step(ctx, step="fetch_dosei"))
    assert result is not None
    # No last_error mutation (key absent, not None).
    assert "last_error" not in ctx.state


def test_prepare_step_missing_required_artifact_records_last_error() -> None:
    # Contrast: required-artifact-missing IS a step-level failure and must
    # still populate last_error so the CLI can act on it.
    ctx = _stub_callback_context(
        user_text='{"target_date": "20260101"}',
        initial_state={},
        initial_artifacts=(),
    )
    result = asyncio.run(
        prepare_step(
            ctx,
            step="enrich_news",
            required_artifacts=("dosei.md",),
        )
    )
    assert result is not None
    last_error = ctx.state["last_error"]
    assert last_error["step"] == "enrich_news"
    assert last_error["missing_artifacts"] == ["dosei.md"]


# --------------------------------------------------------------------------- #
# save_step_output — F5: pin wiring-vs-LLM failure-message split
# --------------------------------------------------------------------------- #


def _save_step_stub(
    initial_state: dict[str, Any] | None = None,
) -> _types.SimpleNamespace:
    async def save_artifact(filename, part, custom_metadata=None):
        return 1

    return _types.SimpleNamespace(
        state=_StubState(initial_state or {}),
        save_artifact=save_artifact,
    )


def test_save_step_output_missing_state_entry_reports_wiring_failure() -> None:
    # F5: output_key absent from state == LlmAgent wiring bug, not an LLM
    # failure. The recorded message + Content payload must be the dedicated
    # wiring-error wording (distinguishable from the "value is None" branch
    # and the generic "agent reported failure" sentinel).
    ctx = _save_step_stub(initial_state={})
    content = asyncio.run(
        save_step_output(
            ctx,
            step="fetch_dosei",
            output_key="temp:fetch_dosei_output",
            artifact_name="dosei.md",
        )
    )
    last_error = ctx.state["last_error"]
    assert last_error["step"] == "fetch_dosei"
    assert last_error["message"].startswith("output_key missing from state")
    assert last_error["output_key"] == "temp:fetch_dosei_output"

    # F2: error_content extras must mirror last_error so AgentTool callers can
    # see the same output_key discriminator on the returned Content.
    payload = json.loads(content.parts[0].text)
    assert payload["status"] == "error"
    assert payload["step"] == "fetch_dosei"
    assert payload["message"].startswith("output_key missing from state")
    assert payload["output_key"] == "temp:fetch_dosei_output"


def test_save_step_output_value_is_none_reports_llm_failure() -> None:
    # F5 parallel: output_key is present but value is None == LLM responded
    # with no parseable body. Different root cause from the wiring branch,
    # so the recorded message must differ verbatim.
    ctx = _save_step_stub(initial_state={"temp:fetch_dosei_output": None})
    asyncio.run(
        save_step_output(
            ctx,
            step="fetch_dosei",
            output_key="temp:fetch_dosei_output",
            artifact_name="dosei.md",
        )
    )
    last_error = ctx.state["last_error"]
    assert last_error["step"] == "fetch_dosei"
    assert last_error["message"] == "output_key present but value is None"
