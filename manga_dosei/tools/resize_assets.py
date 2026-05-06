"""resize_assets: assets/* 配下の画像を長辺 1024px までリサイズする。"""

import io
from typing import Any

from google.adk.tools import ToolContext
from google.genai import types
from PIL import Image

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
            ]
        }

    返り値（失敗時）:
        {"status": "error", "step": "resize_assets", "message": str}
    """
    try:
        validate_target_date(target_date)
    except ValueError as error:
        return _error(str(error))

    artifact_names = await tool_context.list_artifacts()
    asset_names = sorted(n for n in artifact_names if n.startswith("assets/"))

    if not asset_names:
        message = "no assets/* artifacts found; run collect_assets first"
        tool_context.state["last_error"] = {"step": _STEP, "message": message}
        return _error(message)

    resized: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for name in asset_names:
        outcome = await _resize_one(tool_context, name)
        if outcome is None:
            continue
        if outcome.get("resized"):
            resized.append(outcome["resized"])
        else:
            skipped.append(outcome["skipped"])

    tool_context.state.update(
        {
            "target_date": target_date,
            "status": "assets_resized",
            "last_error": None,
        }
    )

    return {
        "status": "success",
        "step": _STEP,
        "resized": resized,
        "skipped": skipped,
    }


async def _resize_one(
    tool_context: ToolContext,
    name: str,
) -> dict[str, Any] | None:
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
                return {"skipped": {"artifact": name, "size": list(before_size)}}

            ratio = _MAX_LONG_SIDE / long_side
            new_size = (int(img.width * ratio), int(img.height * ratio))
            save_format = img.format or _PIL_FORMAT_BY_MIME.get(mime_type, "JPEG")
            resized_img = img.resize(new_size, _RESAMPLING)
            buf = io.BytesIO()
            resized_img.save(buf, format=save_format)
            new_bytes = buf.getvalue()
    except Exception:
        return None

    version = await tool_context.save_artifact(
        name,
        types.Part(inline_data=types.Blob(data=new_bytes, mime_type=mime_type)),
    )
    return {
        "resized": {
            "artifact": name,
            "before_size": list(before_size),
            "after_size": list(new_size),
            "version": version,
        }
    }


def _error(message: str) -> dict[str, Any]:
    return {"status": "error", "step": _STEP, "message": message}
