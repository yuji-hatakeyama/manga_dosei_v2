"""URL を取得して本文テキストを返すシンプルな fetcher。

Tavily extract は markdown 変換時に `<time datetime>` 等の semantic HTML を
落とすため、jiji.com など報道記事の配信日時が取れない。このツールは
httpx で直接 HTML を取得し、boilerplate を除いた visible text をそのまま返す。
配信日時・タイトル等の判別は呼び出し側 LLM に任せる。
"""

import re
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup


# jiji.com など報道サイトは bot 判定で 403 を返すことがあるため、
# 一般的なブラウザ UA を前段に置きつつ、末尾に project 識別子と連絡先 (GitHub URL) を
# 付けて自己申告する。連絡先を付けるのは responsible scraping のための慣行。
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36 "
    "manga_dosei/1.0 (+https://github.com/yuji-hatakeyama/manga_dosei_v2)"
)
_TIMEOUT_SECONDS = 30.0
_MAX_BYTES = 5_000_000  # 5MB 上限。それ以上は切り捨て (壊れた巨大ページ防御)
_MAX_CONTENT_CHARS = 12_000


async def fetch_url(url: str) -> dict[str, Any]:
    """指定 URL の HTML を取得し、boilerplate を除いた visible text を返す。

    引数:
        url: 取得する絶対 URL (https://...)。

    返り値（成功時）:
        {
            "status": "success",
            "url": str,
            "content": str,          # 本文 (boilerplate 除去後、最大 12000 chars)
            "content_truncated": bool,
        }

    返り値（失敗時）:
        {"status": "error", "url": str, "message": str}
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return {
            "status": "error",
            "url": url,
            "message": f"unsupported scheme: {parsed.scheme!r}",
        }

    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT_SECONDS,
            follow_redirects=True,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": (
                    "text/html,application/xhtml+xml,"
                    "application/xml;q=0.9,*/*;q=0.8"
                ),
                "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
            },
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as error:
        return {
            "status": "error",
            "url": url,
            "message": f"fetch failed: {error}",
        }

    html_bytes = response.content[:_MAX_BYTES]
    encoding = response.encoding or "utf-8"
    try:
        html_text = html_bytes.decode(encoding, errors="replace")
    except LookupError:
        html_text = html_bytes.decode("utf-8", errors="replace")

    soup = BeautifulSoup(html_text, "html.parser")
    for selector in ("script", "style", "nav", "header", "footer", "aside", "form"):
        for el in soup.find_all(selector):
            el.decompose()

    container = soup.find("article") or soup.find("main") or soup.body or soup
    text = container.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)

    truncated = len(text) > _MAX_CONTENT_CHARS
    if truncated:
        text = text[:_MAX_CONTENT_CHARS]

    return {
        "status": "success",
        "url": str(response.url),
        "content": text,
        "content_truncated": truncated,
    }
