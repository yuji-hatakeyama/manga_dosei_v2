"""collect_assets: 台本に登場する人物・場所などの参照画像を Wikipedia から集める。"""

import json
from typing import Any, Optional

import httpx
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

from manga_dosei import DEFAULT_TEXT_MODEL
from manga_dosei.tools._common import (
    StepInput,
    error_content,
    prepare_step,
    record_last_error,
    save_text_artifact,
    status_content,
)
from manga_dosei.tools._wikipedia import (
    wiki_image_info,
    wiki_image_search,
    wikimedia_user_agent,
)


_STEP = "collect_assets"
_MANIFEST_ARTIFACT = "manifests/assets.json"
_REQUIRED = ("scenario.md",)
_COLLECTED_KEY = "temp:collected_assets"


_DESCRIPTION = """\
台本に登場する人物・場所などの参照画像を Wikipedia から収集して
assets/<name>.<ext> 形式で保存し、manifests/assets.json も出力するツール。

前提: scenario.md が存在すること。
引数: target_date は YYYYMMDD 形式の対象日。

完了時は収集枚数を含む構造化レスポンスを返す。
前提 artifact が無い場合や失敗時はエラー詳細を含む。
"""


# 対応する MIME type → 拡張子マップ。比較時は lower-case 正規化する前提。
# SVG など Pillow で開けないラスタ以外の形式はここに含めず、download_image で拒否する。
_EXTENSION_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


async def download_image(
    name: str,
    source_url: str,
    description_url: str,
    license: str,
    mime_type: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """画像 URL をダウンロードして assets/<name>.<拡張子> として artifact 保存する。

    引数:
        name: 表示用フルネーム（例: "田中太郎", "国会議事堂"）。ファイル名にも使う。
        source_url: 画像の直接 URL。
        description_url: Wikipedia の description ページ URL。
        license: ライセンス情報。
        mime_type: 画像の MIME type。対応するのは image/jpeg, image/jpg, image/png,
            image/gif, image/webp（大文字小文字は不問）。SVG など Pillow で
            開けない形式は status="error" で拒否する。
    """
    normalized_mime = (mime_type or "").lower()
    if normalized_mime not in _EXTENSION_BY_MIME:
        return {
            "status": "error",
            "name": name,
            "message": (
                f"unsupported mime_type: {mime_type!r}; "
                f"supported: {sorted(_EXTENSION_BY_MIME.keys())}"
            ),
        }
    extension = _EXTENSION_BY_MIME[normalized_mime]
    artifact_name = f"assets/{name}{extension}"

    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": wikimedia_user_agent()},
        ) as client:
            response = await client.get(source_url)
            response.raise_for_status()
            image_bytes = response.content
    except httpx.HTTPError as error:
        return {
            "status": "error",
            "name": name,
            "message": f"download failed: {error}",
        }

    if not image_bytes:
        return {
            "status": "error",
            "name": name,
            "message": "downloaded image is empty",
        }

    version = await tool_context.save_artifact(
        artifact_name,
        types.Part(inline_data=types.Blob(data=image_bytes, mime_type=normalized_mime)),
    )

    collected = list(tool_context.state.get(_COLLECTED_KEY, []) or [])
    collected.append(
        {
            "name": name,
            "artifact": artifact_name,
            "source_url": source_url,
            "description_url": description_url,
            "license": license,
            "mime_type": normalized_mime,
            "bytes": len(image_bytes),
            "version": version,
        }
    )
    tool_context.state[_COLLECTED_KEY] = collected

    return {
        "status": "success",
        "artifact": artifact_name,
        "bytes": len(image_bytes),
        "version": version,
    }


def _build_prompt(scenario_text: str) -> str:
    return f"""
{scenario_text}

上記が「漫画にする首相動静」の台本です。

この台本の登場人物一覧を含めた人物・場所などマンガの資料として様々なものに対して、Wikipedia のツールを利用して、画像を取得してください。
ただし国旗など明らかに画像生成AI が知っているものは不要です。また「高市早苗」もすでにあるので不要。
取得する画像の上限は7枚で人物優先です。優先順位をつけて取得する画像を決定してください。

対応する画像形式は JPEG (image/jpeg, image/jpg)、PNG (image/png)、GIF (image/gif)、WEBP (image/webp) のみです（大文字小文字は不問）。SVG など他の形式は別の候補を選んでください。

保存は assets/田中太郎.png や assets/国会議事堂.jpg のようにフルネームで配置してください。

## ツールの使い方

1. wiki_image_search で候補画像を検索する
2. 候補から相応しいものを wiki_image_info で詳細（ライセンス・サイズ・description URL）を取得する
3. download_image を呼び、画像を artifact として保存する
4. download_image の引数には name（フルネーム）, source_url, description_url, license, mime_type を必ず含める（manifest 生成のため）

すべて完了したら何枚を集めたかを簡潔に報告してください。
""".strip()


def _build_instruction(context: ReadonlyContext) -> str:
    scenario_text = context.state.get("temp:scenario_text", "")
    return _build_prompt(scenario_text)


async def _before(callback_context: CallbackContext) -> Optional[types.Content]:
    err = await prepare_step(
        callback_context,
        step=_STEP,
        required_artifacts=_REQUIRED,
        load_prior={"temp:scenario_text": "scenario.md"},
    )
    if err is not None:
        return err
    callback_context.state[_COLLECTED_KEY] = []
    return None


async def _after(callback_context: CallbackContext) -> Optional[types.Content]:
    target_date = callback_context.state.get("temp:target_date", "")
    collected = callback_context.state.get(_COLLECTED_KEY, []) or []

    if not collected:
        record_last_error(callback_context, _STEP, "no images were collected")
        return error_content(_STEP, "no images were collected")

    manifest = {
        "target_date": target_date,
        "assets": [
            {
                "name": item["name"],
                "artifact": item["artifact"],
                "source_url": item["source_url"],
                "description_url": item["description_url"],
                "mime_type": item["mime_type"],
                "license": item["license"],
            }
            for item in collected
        ],
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    version = await save_text_artifact(
        callback_context,
        _MANIFEST_ARTIFACT,
        manifest_text,
        mime_type="application/json",
    )

    callback_context.state.update(
        {
            "target_date": target_date,
            "asset_manifest_artifact": _MANIFEST_ARTIFACT,
            "asset_count": len(collected),
            "status": "assets_completed",
            "last_error": None,
        }
    )

    return status_content(
        {
            "status": "success",
            "step": _STEP,
            "asset_count": len(collected),
            "manifest_artifact": _MANIFEST_ARTIFACT,
            "manifest_version": version,
        }
    )


_agent = LlmAgent(
    name=_STEP,
    model=DEFAULT_TEXT_MODEL,
    description=_DESCRIPTION,
    instruction=_build_instruction,
    input_schema=StepInput,
    tools=[wiki_image_search, wiki_image_info, download_image],
    before_agent_callback=_before,
    after_agent_callback=_after,
)


collect_assets_tool = AgentTool(agent=_agent)
