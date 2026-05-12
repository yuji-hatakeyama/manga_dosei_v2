# manga_dosei

A daily pipeline that turns the Japanese Prime Minister's published schedule (`首相動静`, *shushou dōsei*) into a one-page manga.

[日本語版 README](README.ja.md)

![Sample output](assets/samples/sample.jpg)

> Generated pages are AI-created fiction based on publicly reported schedules. They are not endorsed by, affiliated with, or representative of any of the people or organizations depicted, and no responsibility is taken for their content.

## Overview

`manga_dosei` is an ADK workflow that summarizes the Prime Minister's agenda for a given date as a one-page manga. The pipeline:

1. Fetches the day's `首相動静` article from JIJI.COM (www.jiji.com).
2. Enriches it with background research (Tavily web search).
3. Drafts a manga script.
4. Collects character / location reference images from Wikimedia Commons.
5. Resizes the references and generates the final manga page with multiple backends (Gemini Image and OpenAI GPT Image), producing several variants to pick from.

## Tech stack

- **Language**: Python 3.12
- **Tooling**: uv, httpx, BeautifulSoup, Pillow, ...
- **Framework**: [Google ADK](https://github.com/google/adk-python)
- **LLM (text)**: Gemini (`gemini-3.1-pro-preview` by default)
- **LLM (image)**: Gemini Image (`gemini-3-pro-image-preview` by default) and OpenAI GPT Image (`gpt-image-2` by default)
- **Web search (MCP)**: [Tavily](https://docs.tavily.com/)
- **Data sources**: JIJI.COM (www.jiji.com), Wikimedia Commons

## Requirements

API keys for Gemini, OpenAI, and Tavily (see `.env.example`).

## Setup

```bash
uv sync
cp .env.example .env   # then fill in GEMINI_API_KEY, OPENAI_API_KEY, TAVILY_API_KEY, WIKIMEDIA_CONTACT_EMAIL
```

## Usage

Run the full pipeline for a given date (`YYYYMMDD`):

```bash
uv run manga_dosei 20260410
```

Outputs (sessions and artifacts) are written under `.adk/`.

To inspect or resume a session interactively via the ADK Web UI, pointing it at the same storage:

```bash
uv run adk web \
  --session_service_uri='sqlite:///./.adk/sessions.db' \
  --artifact_service_uri='file://./.adk/artifacts' \
  .
```
