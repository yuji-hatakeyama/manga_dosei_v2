"""Tavily Search ベースの検索ツール (REST API 直叩き)。

コード側でパラメータ (`topic`, `search_depth`, `include_domains`, 日付範囲 等)
を固定した `FunctionTool` を `make_tavily_search_tool` で量産する。
LLM に露出する引数は基本的に `query` のみで、それ以外は生成時に渡した値で
常に上書きされる。
"""

from collections.abc import Callable
from datetime import date, timedelta
from typing import Any, Literal

import httpx
from google.adk.tools import FunctionTool, ToolContext

from manga_dosei.config import get_settings
from manga_dosei.names import TEMP_TARGET_DATE
from manga_dosei.validation import validate_target_date

_TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_TIMEOUT_SECONDS = 30.0

TavilyTopic = Literal["general", "news", "finance"]
TavilySearchDepth = Literal["basic", "advanced"]

# 日付パラメータは固定文字列 (ISO `YYYY-MM-DD`)、
# もしくは ToolContext を受けて文字列を返す callable を許容する。
# target_date 等を state から取って動的に決めたい場合に callable を使う。
DateResolver = str | Callable[[ToolContext], str] | None


def resolve_date_range(
    target_date: str,
    start_offset: int | None,
    end_offset: int | None,
) -> tuple[str | None, str | None]:
    """`target_date` (YYYYMMDD) から ISO `yyyy-mm-dd` の (start, end) を返す純関数。

    `start_offset` は target からの「日数前」(正値で過去側)、
    `end_offset` は target からの「日数後」(正値で未来側) を表す。
    `None` のオフセットは `None` のまま返す。
    """
    validate_target_date(target_date)
    base = date(int(target_date[:4]), int(target_date[4:6]), int(target_date[6:8]))
    start = (
        (base - timedelta(days=start_offset)).isoformat()
        if start_offset is not None
        else None
    )
    end = (
        (base + timedelta(days=end_offset)).isoformat()
        if end_offset is not None
        else None
    )
    return start, end


async def _call_tavily_search(
    *,
    api_key: str,
    query: str,
    topic: str,
    search_depth: str,
    max_results: int,
    include_domains: list[str] | None,
    exclude_domains: list[str] | None,
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "topic": topic,
        "search_depth": search_depth,
        "max_results": max_results,
    }
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains
    if start_date:
        payload["start_date"] = start_date
    if end_date:
        payload["end_date"] = end_date

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
        resp = await client.post(_TAVILY_SEARCH_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


def _resolve_date(resolver: DateResolver, tool_context: ToolContext) -> str | None:
    if resolver is None:
        return None
    if callable(resolver):
        return resolver(tool_context) or None
    return resolver


def _require_tavily_api_key() -> str:
    # NOTE: read at call time (not at factory time) so tests/import-time code
    # that has not yet set TAVILY_API_KEY do not crash with a bare KeyError.
    secret = get_settings().tavily_api_key
    if secret is None:
        raise RuntimeError("TAVILY_API_KEY is not set")
    return secret.get_secret_value()


def make_tavily_search_tool(
    *,
    name: str,
    description: str,
    topic: TavilyTopic = "general",
    search_depth: TavilySearchDepth = "basic",
    max_results: int = 10,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    start_date: DateResolver = None,
    end_date: DateResolver = None,
) -> FunctionTool:
    """Tavily Search をパラメータ固定で呼ぶ `FunctionTool` を生成する。

    LLM に露出するのは `query` のみ。`topic` / `search_depth` /
    `max_results` / `include_domains` / `exclude_domains` / 日付範囲は
    生成時に渡した値で常に上書きされ、LLM 側で揺らがない。

    日付パラメータは固定文字列 (`"2026-04-01"` 等) のほか、
    `ToolContext` を受けて文字列を返す callable も許容する。
    対象日に応じて動的に決めたい場合は
    `start_date_offset_from_target` / `end_date_offset_from_target` を使う。

    返り値は context window 節約のため `query` と
    `results` (`title` / `url` / `content` / `published_date` のみ) に絞る。
    """
    fixed_include = list(include_domains) if include_domains else None
    fixed_exclude = list(exclude_domains) if exclude_domains else None

    async def _impl(query: str, tool_context: ToolContext) -> dict[str, Any]:
        api_key = _require_tavily_api_key()
        data = await _call_tavily_search(
            api_key=api_key,
            query=query,
            topic=topic,
            search_depth=search_depth,
            max_results=max_results,
            include_domains=fixed_include,
            exclude_domains=fixed_exclude,
            start_date=_resolve_date(start_date, tool_context),
            end_date=_resolve_date(end_date, tool_context),
        )
        return {
            "query": data.get("query", query),
            "results": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "published_date": r.get("published_date", ""),
                }
                for r in data.get("results", [])
            ],
        }

    _impl.__name__ = name
    _impl.__doc__ = description
    return FunctionTool(func=_impl)


def _state_target_date(tool_context: ToolContext) -> str | None:
    target_date = tool_context.state.get(TEMP_TARGET_DATE, "") or ""
    return target_date or None


def start_date_offset_from_target(*, days_before: int) -> Callable[[ToolContext], str]:
    """`state["temp:target_date"]` から N 日前の ISO 日付を返す resolver。

    `days_before=2` なら対象日の 2 日前 (Tavily の index ラグ対策)。
    state に target_date が無い、または YYYYMMDD フォーマットを満たさない場合
    は空文字を返し、API には日付フィルタを渡さない (no-filter フォールバック)。
    adk web 等で state を書き換える経路で型崩れが起きても Tavily 呼び出しが
    例外で落ちないようにするための耐性。
    """

    def _resolver(ctx: ToolContext) -> str:
        target_date = _state_target_date(ctx)
        if target_date is None:
            return ""
        try:
            start, _ = resolve_date_range(target_date, days_before, None)
        except ValueError:
            return ""
        return start or ""

    return _resolver


def end_date_offset_from_target(*, days_after: int) -> Callable[[ToolContext], str]:
    """`state["temp:target_date"]` から N 日後の ISO 日付を返す resolver。

    `days_after=1` なら対象日の翌日まで (後日報道を index 段階で除外)。
    state に target_date が無い、または YYYYMMDD フォーマットを満たさない場合
    は空文字を返し、API には日付フィルタを渡さない (no-filter フォールバック)。
    """

    def _resolver(ctx: ToolContext) -> str:
        target_date = _state_target_date(ctx)
        if target_date is None:
            return ""
        try:
            _, end = resolve_date_range(target_date, None, days_after)
        except ValueError:
            return ""
        return end or ""

    return _resolver
