"""Tavily MCP toolset.

Gemini 内蔵 grounding (`google_search` / `url_context`) は jiji.com の URL を
安定して surface しないため、Tavily の Web 検索 + コンテンツ抽出 MCP server を
代替として使う。Tavily の remote (Streamable HTTP) MCP に接続し、
`tavily_search` / `tavily_extract` 等のツールを LlmAgent に提供する。
"""

import os
from typing import Optional
from urllib.parse import quote

from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams


def build_tavily_toolset(
    tool_filter: Optional[list[str]] = None,
) -> McpToolset:
    """Tavily MCP に接続する toolset を返す。

    `TAVILY_API_KEY` 環境変数が未設定の場合は KeyError。`.env` 経由でロードされる
    想定 (`manga_dosei/__init__.py` で `load_dotenv` を呼んでいる)。

    引数:
        tool_filter: 有効化するツール名のホワイトリスト（例: ["tavily_search"]）。
            None ならすべて有効。tavily_extract は本文をそのまま返すため、
            context window の保護が必要な調査エージェントでは
            ["tavily_search"] を渡すこと。
    """
    api_key = os.environ["TAVILY_API_KEY"]
    return McpToolset(
        connection_params=StreamableHTTPConnectionParams(
            url=f"https://mcp.tavily.com/mcp/?tavilyApiKey={quote(api_key, safe='')}",
            timeout=30.0,
        ),
        tool_filter=tool_filter,
    )
