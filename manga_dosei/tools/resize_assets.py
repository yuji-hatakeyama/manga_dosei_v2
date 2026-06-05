"""resize_assets: assets/* 配下の画像を長辺 1024px までリサイズする。"""

import io
from typing import Any, Literal

from google.adk.tools import ToolContext
from google.genai import types
from PIL import Image

from manga_dosei.names import StateKey
from manga_dosei.tools._common import (
    build_error_result,
    clear_last_error,
    record_last_error,
)
from manga_dosei.tools._state import error_result, ok_result
from manga_dosei.validation import validate_target_date

_STEP = "resize_assets"
_MAX_LONG_SIDE = 1024
_RESAMPLING = Image.Resampling.LANCZOS

_PIL_FORMAT_BY_MIME = {
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",
    "image/png": "PNG",
    "image/gif": "GIF",
    "image/webp": "WEBP",
}


async def resize_assets(target_date: str, tool_context: ToolContext) -> dict[str, Any]:
    """assets/* 配下の参照画像を長辺 1024 px までリサイズして新バージョンとして保存する。

    前提:
        assets/* artifact が 1 つ以上存在すること（collect_assets 完了想定）。

    挙動:
        - 既存の assets/<name>.<ext> 形式の artifact を全て対象にする。
        - Pillow で読み込み、長辺が 1024 px を超える場合のみ LANCZOS で縮小し、
          同じ artifact 名の新バージョンとして保存する（旧版は残す）。
        - 既に 1024 px 以下のものはスキップ（再保存しない）。
        - フォーマットは元画像の format を維持する。

    引数:
        target_date: YYYYMMDD 形式の対象日。

    返り値（成功時）:
        {
            "status": "success",
            "step": "resize_assets",
            "message": str,
            "resized": [
                {"artifact": "assets/<name>.<ext>",
                 "before_size": [w, h],
                 "after_size": [w, h],
                 "version": int},
                ...
            ],
            "skipped": [
                {"artifact": "...", "size": [w, h]},
                ...
            ],
            "failed": [
                {"artifact": "...",
                 "error_type": str,
                 "error_message": str},
                ...
            ]
        }

        失敗エントリが 1 件以上ある場合、`message` にもファイル名が含まれる
        (status="success" のまま無音で握り潰さない契約)。

    返り値（失敗時）:
        {"status": "error", "step": "resize_assets", "message": str}
    """
    try:
        validate_target_date(target_date)
    except ValueError as error:
        # Mirror prepare_step's all-error-paths-record convention so the CLI
        # retry/abort path can observe pre-flight failures via last_error,
        # matching every LlmAgent sibling step.
        return build_error_result(tool_context, _STEP, str(error))

    artifact_names = await tool_context.list_artifacts()
    asset_names = sorted(n for n in artifact_names if n.startswith("assets/"))

    if not asset_names:
        return build_error_result(
            tool_context, _STEP, "no assets/* artifacts found; run collect_assets first"
        )

    resized: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for name in asset_names:
        outcome = await _resize_one(tool_context, name)
        if outcome is None:
            continue
        kind, payload = outcome
        if kind == "resized":
            resized.append(payload)
        elif kind == "failed":
            failed.append(payload)
        else:
            skipped.append(payload)

    # NOTE: partial-success contract — as long as at least one asset succeeded
    # (resized or skipped), the step is success and last_error is cleared, so
    # the CLI does not retry/abort over an isolated bad asset. Only when every
    # asset failed do we record last_error (full failure deserves retry).
    # F1: advance state ONLY on success branches — leaving status='assets_resized'
    # on the failure branches would contradict the returned status='error'.
    succeeded = bool(resized or skipped)
    if failed and not succeeded:
        # All-failure path: clean error shape per the docstring contract — only
        # `failed` rides the payload (no empty `resized` / `skipped` arrays).
        failed_names = ", ".join(item["artifact"] for item in failed)
        message = f"all {len(failed)} assets failed to resize: {failed_names}"
        record_last_error(tool_context, _STEP, message, {"failed": failed})
        payload = error_result(_STEP, message)
        payload["failed"] = failed
        return payload
    if not resized and not skipped and not failed:
        # Every _resize_one returned None (silent skip on every asset, e.g. stale
        # artifact entries / empty inline_data). Without this branch the step
        # would report success with zero artifacts produced — surface as error.
        message = f"all {len(asset_names)} assets produced no usable bytes"
        record_last_error(tool_context, _STEP, message)
        return error_result(_STEP, message)
    tool_context.state.update(
        {
            StateKey.TARGET_DATE: target_date,
            StateKey.STATUS: "assets_resized",
        }
    )
    clear_last_error(tool_context)
    if failed:
        failed_names = ", ".join(item["artifact"] for item in failed)
        message = f"assets resized; {len(failed)} failed: {failed_names}"
    else:
        message = "assets resized"
    payload = ok_result(_STEP, message)
    payload["resized"] = resized
    payload["skipped"] = skipped
    payload["failed"] = failed
    return payload


_ResizeKind = Literal["resized", "skipped", "failed"]


async def _resize_one(
    tool_context: ToolContext,
    name: str,
) -> tuple[_ResizeKind, dict[str, Any]] | None:
    part = await tool_context.load_artifact(name)
    if part is None or part.inline_data is None or not part.inline_data.data:
        return None
    data = part.inline_data.data
    mime_type = part.inline_data.mime_type or "image/jpeg"

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            before_size = img.size
            long_side = max(before_size)
            if long_side <= _MAX_LONG_SIDE:
                return "skipped", {"artifact": name, "size": list(before_size)}

            ratio = _MAX_LONG_SIDE / long_side
            new_size = (int(img.width * ratio), int(img.height * ratio))
            save_format = img.format or _PIL_FORMAT_BY_MIME.get(mime_type, "JPEG")
            resized_img = img.resize(new_size, _RESAMPLING)
            buf = io.BytesIO()
            resized_img.save(buf, format=save_format)
            new_bytes = buf.getvalue()
    # NOTE: catch-all so a single bad asset cannot crash the whole step.
    # Direct tools are dispatched outside the LLM-runner try/except in
    # _run_step_with_retry, so an uncaught exception here reaches asyncio.run
    # and aborts the pipeline. The per-asset diagnostic (type+message) is
    # preserved in the failed list for visibility.
    except Exception as error:
        return "failed", {
            "artifact": name,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

    version = await tool_context.save_artifact(
        name,
        types.Part(inline_data=types.Blob(data=new_bytes, mime_type=mime_type)),
    )
    return "resized", {
        "artifact": name,
        "before_size": list(before_size),
        "after_size": list(new_size),
        "version": version,
    }
