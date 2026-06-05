"""Unit tests for `manga_dosei.tools._state` builders (refactor unit U3).

Locks the wire format of `state["last_error"]` and direct-tool return values
so the migration from hand-built dicts to these builders stays byte-compatible
with the existing consumers (see callers in `manga_dosei/tools/*.py`).
"""

from __future__ import annotations

from manga_dosei.tools._state import (
    error_result,
    ok_result,
)


def test_ok_result_version_zero_is_preserved() -> None:
    # Guard: version=0 (first artifact version) must not be dropped by truthiness checks.
    payload = ok_result("step", "msg", artifact="x.md", version=0)
    assert payload["version"] == 0


def test_error_result_minimal_fields() -> None:
    payload = error_result("collect_assets", "no images were collected")
    assert payload == {
        "status": "error",
        "step": "collect_assets",
        "message": "no images were collected",
    }
