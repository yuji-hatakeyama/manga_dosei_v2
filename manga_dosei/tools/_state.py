"""Typed payloads for `state["last_error"]` and direct-tool return values.

Centralises the `last_error` and direct-tool result shape so the existing
hand-built dicts can migrate to one source of truth without changing wire
format.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class LastError(TypedDict):
    """`state["last_error"]` payload. When present, `missing_artifacts` is non-empty.

    The diagnostic fields below (`reason` / `output_key` / `page_number` /
    `failed`) are populated by `record_last_error` callers that pass `extra=`
    with the matching key. They are declared here so the TypedDict actually
    describes what gets persisted to `state["last_error"]` — readers can rely
    on the shape, and adding a new diagnostic field requires editing this
    declaration (no silent widening).
    """

    step: str
    message: str
    missing_artifacts: NotRequired[list[str]]
    reason: NotRequired[str]
    output_key: NotRequired[str]
    page_number: NotRequired[int]
    failed: NotRequired[list[dict[str, Any]]]


def ok_result(
    step: str,
    message: str,
    *,
    artifact: str | None = None,
    version: int | None = None,
) -> dict[str, Any]:
    # NOTE: declared dict[str, Any] (not a TypedDict) so callers can append
    # extras without an intermediate widening cast (F6).
    payload: dict[str, Any] = {"status": "success", "step": step, "message": message}
    if artifact is not None:
        payload["artifact"] = artifact
    if version is not None:
        payload["version"] = version
    return payload


def error_result(step: str, message: str) -> dict[str, Any]:
    return {"status": "error", "step": step, "message": message}
