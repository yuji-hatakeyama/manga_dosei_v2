"""Wikipedia Commons API の薄いラッパー。collect_assets の LlmAgent から FunctionTool として呼ばれる。

MCP server (wikipedia-mcp-image-crawler) の代替実装。Pure Python で
User-Agent を本プロジェクト固有のものにし、Wikipedia の rate-limit ポリシーに
適切に応じる。
"""

import os
from typing import Any

import httpx


_API_ENDPOINT = "https://commons.wikimedia.org/w/api.php"
_TIMEOUT_SECONDS = 30.0


def wikimedia_user_agent() -> str:
    """Wikipedia rate-limit policy に準拠した User-Agent を返す。

    `WIKIMEDIA_CONTACT_EMAIL` 環境変数で連絡先メールアドレスを指定する。
    未設定の場合は連絡先なしで返すが、Wikipedia 側で rate-limit を
    厳しく扱われる可能性があるため、運用時は設定すること。
    """
    contact = os.environ.get("WIKIMEDIA_CONTACT_EMAIL", "").strip()
    contact_part = f"; {contact}" if contact else ""
    return (
        f"manga_dosei/1.0 "
        f"(https://github.com/yuji-hatakeyama/manga_dosei_v2{contact_part})"
    )


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=_TIMEOUT_SECONDS,
        headers={"User-Agent": wikimedia_user_agent()},
    )


async def wiki_image_search(query: str, limit: int = 10) -> dict[str, Any]:
    """Wikipedia Commons で画像を検索する。

    引数:
        query: 検索キーワード（例: "高市早苗", "国会議事堂"）。`File:` プレフィックスは
            内部で付加されるため指定不要。
        limit: 結果数の上限 (1〜50)。範囲外は clamp される。

    返り値（成功時）:
        {"results": [
            {"title": "File:...", "url": "https://...",
             "mime_type": "image/jpeg", "size": int,
             "dimensions": {"width": int, "height": int}},
            ...
        ]}
    返り値（失敗時）:
        {"status": "error", "message": str}
    """
    bounded_limit = max(1, min(50, int(limit)))
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"File:{query}",
        "gsrlimit": bounded_limit,
        "prop": "imageinfo",
        "iiprop": "url|size|mime",
        "format": "json",
        "origin": "*",
    }
    try:
        async with _client() as client:
            response = await client.get(_API_ENDPOINT, params=params)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as error:
        return {"status": "error", "message": f"wikipedia search failed: {error}"}

    pages = (data.get("query") or {}).get("pages") or {}
    results: list[dict[str, Any]] = []
    for page in pages.values():
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        results.append(
            {
                "title": page.get("title"),
                "url": info.get("url"),
                "mime_type": info.get("mime"),
                "size": info.get("size"),
                "dimensions": {
                    "width": info.get("width"),
                    "height": info.get("height"),
                },
            }
        )
    return {"results": results}


async def wiki_image_info(title: str) -> dict[str, Any]:
    """Wikipedia Commons の特定画像の詳細メタデータを取得する。

    引数:
        title: 画像タイトル (例: "File:Yoshino_Tomoko.jpg")。`File:` プレフィックス必須。
            wiki_image_search の `title` フィールドをそのまま渡せばよい。

    返り値（成功時）:
        {"title": ..., "url": ..., "description_url": ...,
         "mime_type": ..., "size": int,
         "dimensions": {"width": int, "height": int},
         "license": str | None, "author": str | None}
    返り値（失敗時）:
        {"status": "error", "message": str}
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "format": "json",
    }
    try:
        async with _client() as client:
            response = await client.get(_API_ENDPOINT, params=params)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as error:
        return {"status": "error", "message": f"wikipedia info failed: {error}"}

    pages = (data.get("query") or {}).get("pages") or {}
    page = next(iter(pages.values()), None)
    if page is None:
        return {"status": "error", "message": f"no page returned for title: {title}"}

    infos = page.get("imageinfo") or []
    if not infos:
        return {"status": "error", "message": f"no imageinfo for title: {title}"}
    info = infos[0]
    extmeta = info.get("extmetadata") or {}

    return {
        "title": page.get("title"),
        "url": info.get("url"),
        "description_url": info.get("descriptionurl"),
        "mime_type": info.get("mime"),
        "size": info.get("size"),
        "dimensions": {
            "width": info.get("width"),
            "height": info.get("height"),
        },
        "license": (extmeta.get("LicenseShortName") or {}).get("value"),
        "author": (extmeta.get("Artist") or {}).get("value"),
    }
