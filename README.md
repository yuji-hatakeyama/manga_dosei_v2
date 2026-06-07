# manga_dosei

A daily pipeline that turns the Japanese Prime Minister's published schedule (`首相動静`, *shushou dōsei*) into a one-page manga.

[日本語版 README](README.ja.md)

![Sample output](assets/samples/sample.jpg)

> Generated pages are AI-created fiction based on publicly reported schedules. They are not endorsed by, affiliated with, or representative of any of the people or organizations depicted, and no responsibility is taken for their content.

## Overview

`manga_dosei` is an ADK workflow that summarizes the Prime Minister's agenda for a given date as a one-page manga. Every run is keyed by `target_date` (`YYYYMMDD`); each step writes an artifact under that session, and the next step consumes those artifacts. The intermediate files are kept on disk (not in-memory) so any step can be re-run independently from `adk web`.

## Pipeline

```mermaid
flowchart TD
    CLI([uv run manga_dosei YYYYMMDD])

    FD[fetch_dosei<br/>LlmAgent]

    subgraph EN["enrich_news · SequentialAgent"]
        direction TB
        ENR[section_researcher<br/>LlmAgent]
        subgraph ENL["LoopAgent · max_iterations=2"]
            direction TB
            ENV[research_evaluator<br/>LlmAgent] --> ENG{grade?}
            ENG -- fail --> ENX[enhanced_search_executor<br/>LlmAgent]
        end
        ENO[news_composer<br/>LlmAgent]
        ENR --> ENL
        ENG -- pass --> ENO
    end

    SU[summarize_url<br/>AgentTool · LlmAgent]
    ENR -. uses .-> SU
    ENX -. uses .-> SU

    GS[generate_scenario<br/>LlmAgent]
    CA[collect_assets<br/>LlmAgent]
    DL[define_layout<br/>LlmAgent]
    CB[compose_image_brief<br/>LlmAgent]
    GP[generate_page_gemini × 5<br/>Gemini Image]

    CLI ==> FD
    FD == dosei.md ==> EN
    EN == news.md ==> GS
    GS == scenario.md ==> CA
    CA == "assets/* + manifests/assets.json (resized ≤1024px)" ==> DL
    DL == layout.md ==> CB
    CB == image_brief.md ==> GP
    GP ==> OUT[/"pages/gemini_1..5"/]
```

Each step is one ADK tool, and each produces a stable artifact name that downstream steps depend on. The narrative content, the page structure, and the final image-gen brief are intentionally separate files so you can edit one without rebuilding the rest. The non-LLM helpers omitted from the diagram for readability (Tavily / httpx / Wikimedia API, etc.) are listed in the "Tools" column.

| # | Step | Produces | Tools | What it does |
|---|---|---|---|---|
| 1 | `fetch_dosei` | `dosei.md` | `search_jiji_for_dosei` (Tavily / jiji.com only), `fetch_url` (httpx) | Finds the article URL, fetches the page HTML, and transcribes the day's `首相動静` verbatim (including the trailing 配信日時 line). No summarization. |
| 2 | `enrich_news` | `news.md` | `search_news_jiji` / `search_news_yahoo` (Tavily), `summarize_url` | `SequentialAgent` orchestrating 3 stages: **section_researcher** (gather initial findings) → **LoopAgent** (max 2 iterations: **research_evaluator** grades pass/fail, and on fail **enhanced_search_executor** runs the follow-ups) → **news_composer** (merges findings + `dosei.md` into the final markdown). Output covers manga story candidates, per-person profiles, meeting context, and surrounding political background — all with sources. `summarize_url` summarizes page bodies inside a child `LlmAgent` so raw HTML never reaches the parent context. |
| 3 | `generate_scenario` | `scenario.md` | — | Drafts the actual manga script from `news.md`: page title, panel titles, per-panel 状況 / イラスト / dialogue (verbatim, numbered), 登場人物一覧, and X-post text. **Sole authority for narrative content** — downstream steps only restructure or render what's here. |
| 4 | `collect_assets` | `assets/<name>.<ext>` + `manifests/assets.json` | `wiki_image_search` / `wiki_image_info` (Wikimedia), `download_image` (httpx) | Reads `scenario.md`'s cast list and collects ≤7 references (people first). The manifest records source URL, license, and MIME type per image. |
| 5 | `resize_assets` | `assets/<name>.<ext>` (new versions) | Pillow (LANCZOS) | Downsizes only reference images whose long side exceeds 1024 px, writing them back as new versions. Original versions are kept on the artifact store. |
| 6 | `define_layout` | `layout.md` | — | Picks a `pattern_id` from the layout catalog (`assets/layouts/{3a,4a,4b,4c,4d,5a}/`) based on panel count and pacing, then transcribes the ASCII figure and per-row layout verbatim and adds per-panel character placement (画面左 / 画面右) derived from speaker order. Layout-only — no titles, dialogue, or visuals. |
| 7 | `compose_image_brief` | `image_brief.md` | — | Merges `scenario.md` + `layout.md` + `news.md` + `manifests/assets.json` into a single self-contained brief for the image generator: page header (verbatim), 登場人物プロフィール (with 参照画像あり/なし flag), the layout's ASCII figure, and per-panel specs (position + character placement + situation/illustration hints + verbatim balloons). Fields tagged `(verbatim)` are the only things meant to render as on-page text. |
| 8 | `generate_page_gemini` (×5) | `pages/gemini_<N>.<jpg\|png>` | Gemini Image | Feeds `image_brief.md` plus three kinds of reference (the layout sample for the chosen `pattern_id`, the Takaichi Sanae character reference `assets/samples/sanae.jpg`, and the resized `assets/*` images). No internal retry — called `PAGE_VARIANT_COUNT` times; the best variant is picked manually. |

The daily CLI currently uses only the Gemini Image backend. `generate_page_gpt` (OpenAI GPT Image, `PAGE_VARIANT_COUNT=2`) is still registered on the ADK agent and reachable from `adk web`, but is not invoked by `uv run manga_dosei` — Gemini gives more reliable composition and on-page text rendering.

## Tech stack

- **Language**: Python 3.12
- **Tooling**: uv, httpx, BeautifulSoup, Pillow, …
- **Framework**: [Google ADK](https://github.com/google/adk-python)
- **LLM (text)**: Gemini (`gemini-3.5-flash` by default)
- **LLM (image)**: Gemini Image (`gemini-3-pro-image-preview` by default; OpenAI GPT Image `gpt-image-2` is available through the ADK agent but is not invoked by the daily CLI)
- **Web search**: [Tavily REST API](https://docs.tavily.com/) (parameters fixed in code via `make_tavily_search_tool`; the LLM only chooses `query`)
- **Data sources**: jiji.com, Yahoo! News (news.yahoo.co.jp), Wikimedia Commons

## Requirements

Gemini and Tavily API keys plus a Wikimedia contact email are required for the daily CLI. `OPENAI_API_KEY` is only needed if you also invoke `generate_page_gpt` via `adk web` (see `.env.example`).

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

Outputs (sessions and artifacts) are written to `.adk/` anchored to the parent of the `manga_dosei` package directory, not your CWD, so running `manga_dosei` from a subdirectory still writes to the same place. With the default editable `uv sync` install, that resolves to `<repo>/.adk/`.

### Re-running with a distinct session id

By default the session id equals `target_date`, so re-running the same date overwrites / mixes with the previous run's session and artifacts under `.adk/`. Pass `--session-id` to keep a second run isolated:

```bash
uv run manga_dosei 20260410 --session-id 20260410_retry1
```

The value must match `^\d{8}(_[A-Za-z0-9_-]+)?$` and its first 8 chars must equal `target_date` (the session id is also used as the artifact storage subdirectory name, so the suffix is restricted to filesystem-safe characters).

### Dumping artifacts for archiving

Pass `--publish-dir PATH` to also write the latest version of every artifact under `PATH`, preserving the artifact-name slash hierarchy (e.g. `pages/gemini_1.jpg` → `PATH/pages/gemini_1.jpg`):

```bash
uv run manga_dosei 20260410 --publish-dir /tmp/manga-out
```

### Pushing the dump to a GitHub archive repo

`manga_dosei-publish` is a separate CLI that uploads a directory to a GitHub repo as one fast-forward commit, used for example to archive each daily run into a private repo:

```bash
GITHUB_OUTPUT_TOKEN=<fine-grained PAT> \
uv run manga_dosei-publish \
  --source /tmp/manga-out \
  --repo owner/manga_dosei_v2_artifacts \
  --dest 2026/04/20260410 \
  --message "publish: 20260410"
```

- Files and directories under `--source` whose name starts with `.` are skipped, and symlinks are not followed (so `.env`, `.git/`, `.adk/`, editor swap files, etc. are excluded). Logs intended to ride the same commit (e.g. redirected stdout/stderr) must therefore not start with `.`.
- `--dest` must be a clean relative POSIX path (see `manga_dosei-publish --help` for the exact rules); an empty / `/` value pushes to the repo root.

The token is read only from `GITHUB_OUTPUT_TOKEN` in the process env; it is never accepted as a CLI flag, and `manga_dosei-publish` deliberately ignores any `.env` on disk so a stray `.env` under `--source` / the CI working directory cannot inject the token. The pipeline (`uv run manga_dosei`) does load `.env` from CWD as usual — only the publisher opts out.

### Interactive session

To inspect or resume a session interactively via the ADK Web UI, pointing it at the same storage:

```bash
uv run adk web \
  --session_service_uri="sqlite:///$(pwd)/.adk/sessions.db" \
  --artifact_service_uri="file://$(pwd)/.adk/artifacts" \
  .
```

Invoke `adk web` from the same directory the CLI writes its `.adk/` to (as noted above, usually `<repo-root>/.adk/`) so that `$(pwd)/.adk/...` resolves to the same path. Absolute paths are required: `file://./...` is parsed with `.` as a host name and rejected ("file:// artifact URIs must reference the local filesystem.").

## Tests

```bash
uv run pytest
```

Pure unit-level checks under `tests/` — no LLM, network, or `.adk/` storage.
