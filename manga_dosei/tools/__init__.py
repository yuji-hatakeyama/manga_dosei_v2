"""ADK tools for the manga dosei workflow."""

from manga_dosei.tools.collect_assets import collect_assets_tool
from manga_dosei.tools.enrich_news import enrich_news_tool
from manga_dosei.tools.fetch_dosei import fetch_dosei_tool
from manga_dosei.tools.generate_page_gemini import generate_page_gemini
from manga_dosei.tools.generate_page_gpt import generate_page_gpt
from manga_dosei.tools.generate_scenario import generate_scenario_tool
from manga_dosei.tools.inspect_artifacts import inspect_artifacts
from manga_dosei.tools.resize_assets import resize_assets

__all__ = [
    "collect_assets_tool",
    "enrich_news_tool",
    "fetch_dosei_tool",
    "generate_page_gemini",
    "generate_page_gpt",
    "generate_scenario_tool",
    "inspect_artifacts",
    "resize_assets",
]
