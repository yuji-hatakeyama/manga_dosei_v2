"""Shared helpers for the image-generation tools.

Used by generate_page_gemini and generate_page_gpt to collapse the duplicated
pattern_id regex, asset fan-out loop, and repo-relative asset directory
derivation that both backends previously redeclared.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, NamedTuple

from google.adk.tools import ToolContext
from google.genai import types

from manga_dosei import paths
from manga_dosei.names import ArtifactName, StateKey
from manga_dosei.tools._common import build_error_result, clear_last_error
from manga_dosei.tools._state import error_result, ok_result
from manga_dosei.validation import validate_target_date

_PATTERN_ID_RE = re.compile(r"^-\s*pattern_id:\s*([A-Za-z0-9_-]+)\s*$", re.MULTILINE)


async def load_brief_and_layout_sample(
    tool_context: ToolContext,
    *,
    step: str,
    target_date: str,
    page_number: int,
) -> tuple[str, Path] | dict[str, Any]:
    """Pre-flight common to both image-gen backends.

    Validates `target_date`, checks the character reference image and the
    `image_brief.md` artifact, parses `pattern_id`, and resolves the matching
    layout sample path. On any failure returns a `build_error_result(...)` dict
    so callers can `return` it directly. On success returns
    `(brief_text, sample_page_path)` — `pattern_id` is consumed internally to
    build the sample path and is not surfaced to callers.
    """
    try:
        validate_target_date(target_date)
    except ValueError as error:
        # NOTE: malformed target_date is wiring, not a step-level failure — do
        # NOT call build_error_result here (it would write last_error and
        # clobber the upstream step's real last_error). Mirrors prepare_step
        # in _common.py L199-203.
        payload = error_result(step, str(error))
        payload["page_number"] = page_number
        return payload

    if not paths.CHARACTER_REF_PATH.exists():
        return build_error_result(
            tool_context,
            step,
            f"character reference image missing: {paths.CHARACTER_REF_PATH}",
            page_number=page_number,
        )

    brief_part = await tool_context.load_artifact(ArtifactName.IMAGE_BRIEF)
    if brief_part is None or brief_part.text is None:
        return build_error_result(
            tool_context,
            step,
            "image_brief.md is missing or unreadable (run compose_image_brief first)",
            page_number=page_number,
        )
    brief_text = brief_part.text

    try:
        pattern_id = parse_pattern_id(brief_text)
    except ValueError as error:
        return build_error_result(tool_context, step, str(error), page_number=page_number)

    sample_page_path = paths.LAYOUTS_DIR / pattern_id / "sample.jpg"
    if not sample_page_path.exists():
        return build_error_result(
            tool_context,
            step,
            f"layout sample image missing for pattern_id={pattern_id}: "
            f"{sample_page_path}",
            page_number=page_number,
        )
    return brief_text, sample_page_path


def parse_pattern_id(image_brief_text: str) -> str:
    match = _PATTERN_ID_RE.search(image_brief_text)
    if not match:
        raise ValueError(
            "pattern_id not found in image_brief.md "
            "(expected `- pattern_id: <id>` near the top)"
        )
    return match.group(1)


class AssetPart(NamedTuple):
    """assets/* artifact 1 件分。image-gen バックエンドが使う識別子をまとめる。

    `basename` は `key` から `assets/` 接頭辞を剥がした生のファイル名 (拡張子付き)。
    `asset_name` は同じく拡張子も除去した表示名。caller がそれぞれ別箇所で再導出
    していた "key.removeprefix" を 1 か所に集約する。
    `data` / `mime` は呼び出し側が `part.inline_data` を辿らずに直接読めるよう
    `load_asset_parts` 内で確定済みの値 (`inline_data.data` と
    `inline_data.mime_type or "image/jpeg"`) をコピーしたもの。
    """

    key: str
    asset_name: str
    basename: str
    part: types.Part
    data: bytes
    mime: str


async def load_asset_parts(tool_context: ToolContext) -> list[AssetPart]:
    """Load assets/* artifacts as `AssetPart` tuples in sorted key order.

    Skips entries whose inline_data is missing/empty — image-gen backends would
    otherwise blow up on zero-byte attachments.
    """
    asset_keys = sorted(
        key for key in await tool_context.list_artifacts() if key.startswith("assets/")
    )
    parts: list[AssetPart] = []
    for key in asset_keys:
        asset_part = await tool_context.load_artifact(key)
        if (
            asset_part is None
            or asset_part.inline_data is None
            or not asset_part.inline_data.data
        ):
            continue
        basename = key.removeprefix("assets/") or "asset"
        asset_name = basename.rsplit(".", 1)[0] if "." in basename else basename
        data = asset_part.inline_data.data
        mime = asset_part.inline_data.mime_type or "image/jpeg"
        parts.append(AssetPart(key, asset_name, basename, asset_part, data, mime))
    return parts


async def save_page_artifact(
    tool_context: ToolContext,
    *,
    step: str,
    model_label: str,
    page_number: int,
    target_date: str,
    image_bytes: bytes,
    mime: str,
    extension: str,
) -> dict[str, Any]:
    """Persist generated page bytes, update session state, return success payload.

    Both backends share an identical tail (artifact-name assembly, save_artifact,
    state.update, success dict); only model_label / extension / mime vary.
    """
    artifact_name = f"pages/{model_label}_{page_number}{extension}"
    version = await tool_context.save_artifact(
        artifact_name,
        types.Part(inline_data=types.Blob(data=image_bytes, mime_type=mime)),
    )

    tool_context.state[StateKey.TARGET_DATE] = target_date
    clear_last_error(tool_context)

    payload = ok_result(step, "page generated", artifact=artifact_name, version=version)
    payload["page_number"] = page_number
    payload["bytes"] = len(image_bytes)
    payload["mime_type"] = mime
    return payload


__all__: list[str] = [
    "AssetPart",
    "load_asset_parts",
    "load_brief_and_layout_sample",
    "parse_pattern_id",
    "save_page_artifact",
]
