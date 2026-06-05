"""State-mutation tests for `manga_dosei.tools.resize_assets` (refactor unit U8).

Locks the `state["last_error"]` shape and reset behavior after the migration
to `record_last_error` / `clear_last_error`. The success-state keys
(`target_date`, `status`) must keep being written exactly as before.
"""

from __future__ import annotations

import asyncio
import importlib

from google.genai import types

# `manga_dosei.tools.__init__` re-exports `resize_assets` (function), which
# shadows the submodule attribute on the package; load the submodule directly
# so `monkeypatch.setattr(resize_assets_module.Image, ...)` works.
resize_assets_module = importlib.import_module("manga_dosei.tools.resize_assets")
resize_assets = resize_assets_module.resize_assets

# Valid 1x1 PNG; reused across partial-success / catch-all tests.
_ONE_PX_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
    "0000000c49444154789c63f8cfc0000003010100c9fe92ef0000000049454e44ae426082"
)


def test_resize_assets_records_last_error_when_no_assets(stub_tool_context) -> None:
    # No assets/* artifacts → typed LastError payload + status=error result.
    ctx = stub_tool_context()
    result = asyncio.run(resize_assets("20260101", ctx))

    assert result["status"] == "error"
    assert result["step"] == "resize_assets"
    assert "no assets/* artifacts" in result["message"]

    last_error = ctx.state["last_error"]
    assert last_error == {
        "step": "resize_assets",
        "message": "no assets/* artifacts found; run collect_assets first",
    }


def test_resize_assets_clears_last_error_on_success(stub_tool_context) -> None:
    # Pre-seed last_error to verify clear_last_error nulls it on the success path,
    # even when every asset is below the resize threshold (skipped, no save).
    ctx = stub_tool_context(
        initial_state={"last_error": {"step": "prev", "message": "stale"}},
        initial_artifacts={
            "assets/sample.png": types.Part(
                inline_data=types.Blob(data=_ONE_PX_PNG, mime_type="image/png")
            ),
        },
    )
    result = asyncio.run(resize_assets("20260101", ctx))

    assert result["status"] == "success"
    assert result["step"] == "resize_assets"
    assert ctx.state["last_error"] is None
    assert ctx.state["target_date"] == "20260101"
    assert ctx.state["status"] == "assets_resized"


def test_resize_assets_invalid_target_date_records_last_error(
    stub_tool_context,
) -> None:
    # Invalid target_date records last_error so the CLI retry/abort path can
    # observe it (matches prepare_step's convention used by every LlmAgent
    # sibling step — see _common.py L199-203).
    ctx = stub_tool_context()
    result = asyncio.run(resize_assets("bad-date", ctx))

    assert result["status"] == "error"
    assert result["step"] == "resize_assets"
    last_error = ctx.state["last_error"]
    assert last_error["step"] == "resize_assets"
    assert last_error["message"] == result["message"]


def test_resize_assets_partial_failure_keeps_success_and_clears_last_error(
    stub_tool_context,
) -> None:
    # F2 regression: with 2 good assets + 1 corrupt, the step must report
    # status='success', clear last_error, and surface the bad one in `failed`.
    # Pre-seed last_error to verify it gets cleared on the partial-success path.
    ctx = stub_tool_context(
        initial_state={"last_error": {"step": "prev", "message": "stale"}},
        initial_artifacts={
            "assets/good_a.png": types.Part(
                inline_data=types.Blob(data=_ONE_PX_PNG, mime_type="image/png")
            ),
            "assets/good_b.png": types.Part(
                inline_data=types.Blob(data=_ONE_PX_PNG, mime_type="image/png")
            ),
            "assets/bad.png": types.Part(
                inline_data=types.Blob(data=b"not a png", mime_type="image/png")
            ),
        },
    )

    result = asyncio.run(resize_assets("20260101", ctx))

    assert result["status"] == "success"
    assert ctx.state["last_error"] is None
    failed_names = [item["artifact"] for item in result["failed"]]
    assert failed_names == ["assets/bad.png"]
    # The two valid assets land in `skipped` (under 1024 px) or `resized`.
    survivors = [item["artifact"] for item in (result["resized"] + result["skipped"])]
    assert set(survivors) == {"assets/good_a.png", "assets/good_b.png"}


def test_resize_assets_all_failures_records_last_error(
    stub_tool_context,
) -> None:
    # CR3 regression: when every asset fails, the step is a true failure —
    # record last_error and return the clean error shape (no empty
    # `resized` / `skipped` arrays), only `failed` rides the payload.
    ctx = stub_tool_context(
        initial_artifacts={
            "assets/bad_a.png": types.Part(
                inline_data=types.Blob(data=b"not a png", mime_type="image/png")
            ),
            "assets/bad_b.png": types.Part(
                inline_data=types.Blob(data=b"also broken", mime_type="image/png")
            ),
        },
    )

    result = asyncio.run(resize_assets("20260101", ctx))

    assert result["status"] == "error"
    assert ctx.state["last_error"] is not None
    assert ctx.state["last_error"]["step"] == "resize_assets"
    # Clean error shape: no `resized` / `skipped` keys; `failed` present.
    assert "resized" not in result
    assert "skipped" not in result
    failed_names = [item["artifact"] for item in result["failed"]]
    assert set(failed_names) == {"assets/bad_a.png", "assets/bad_b.png"}


def test_resize_assets_all_failures_does_not_advance_status(
    stub_tool_context,
) -> None:
    # F1 regression: on the all-failure path, state must NOT advance to
    # status='assets_resized' — leaving it set would contradict the returned
    # status='error' and confuse inspect_artifacts / the CLI retry path.
    ctx = stub_tool_context(
        initial_artifacts={
            "assets/bad_a.png": types.Part(
                inline_data=types.Blob(data=b"not a png", mime_type="image/png")
            ),
        },
    )

    result = asyncio.run(resize_assets("20260101", ctx))

    assert result["status"] == "error"
    assert ctx.state.get("status") != "assets_resized"


def test_resize_assets_all_none_outcomes_records_last_error(
    stub_tool_context,
) -> None:
    # CR12 regression: when every _resize_one returns None (silent skip —
    # e.g. empty inline_data on every asset), the step must surface an error
    # rather than reporting success with zero artifacts touched. Empty Blobs
    # are exactly the production trigger (`not part.inline_data.data` branch
    # in `_resize_one`), so no monkeypatch is needed — drive the real code.
    ctx = stub_tool_context(
        initial_artifacts={
            "assets/a.png": types.Part(
                inline_data=types.Blob(data=b"", mime_type="image/png")
            ),
            "assets/b.png": types.Part(
                inline_data=types.Blob(data=b"", mime_type="image/png")
            ),
        },
    )

    result = asyncio.run(resize_assets("20260101", ctx))

    assert result["status"] == "error"
    assert result["step"] == "resize_assets"
    assert "produced no usable bytes" in result["message"]
    assert ctx.state["last_error"] is not None
    assert ctx.state["last_error"]["step"] == "resize_assets"


def test_resize_assets_unexpected_exception_is_caught_per_asset(
    stub_tool_context, monkeypatch
) -> None:
    # F5 regression: a non-listed exception type from Pillow (e.g. RuntimeError)
    # must not crash the whole step — it lands in `failed` with the diagnostic.
    ctx = stub_tool_context(
        initial_artifacts={
            "assets/good.png": types.Part(
                inline_data=types.Blob(data=_ONE_PX_PNG, mime_type="image/png")
            ),
            "assets/explodes.png": types.Part(
                inline_data=types.Blob(data=_ONE_PX_PNG, mime_type="image/png")
            ),
        },
    )

    real_open = resize_assets_module.Image.open

    def fake_open(buf, *args, **kwargs):
        # Inspect the bytes to decide which artifact this open() is for.
        # Both inputs are identical PNGs, so route by call count instead.
        fake_open.calls += 1  # type: ignore[attr-defined]
        if fake_open.calls == 2:  # type: ignore[attr-defined]
            raise RuntimeError("synthetic non-Pillow failure")
        return real_open(buf, *args, **kwargs)

    fake_open.calls = 0  # type: ignore[attr-defined]
    monkeypatch.setattr(resize_assets_module.Image, "open", fake_open)

    result = asyncio.run(resize_assets("20260101", ctx))

    assert result["status"] == "success"
    failed = result["failed"]
    assert len(failed) == 1
    assert failed[0]["error_type"] == "RuntimeError"
    assert "synthetic non-Pillow failure" in failed[0]["error_message"]
