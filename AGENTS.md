# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. `CLAUDE.md` is a symlink to this file.

## Reference materials (consult actively)

This project is built on **Google ADK (Agent Development Kit)**. When designing or modifying agent/tool code, prefer reading these references over guessing:

- ADK samples: https://github.com/google/adk-samples — concrete patterns (e.g. `python/agents/deep-search` is the basis for `enrich_news`'s `SequentialAgent → LoopAgent → composer` shape).
- ADK full documentation (single-file dump for LLM consumption): https://adk.dev/llms-full.txt — fetch it with `WebFetch` whenever ADK behavior (`LlmAgent`, `AgentTool`, `Runner`, `*ArtifactService`, `*SessionService`, callbacks, schemas, MCP toolsets, etc.) is unclear.

ADK APIs evolve quickly; do not rely on prior knowledge alone.

## Commands

Dependencies are managed by `uv` and the package is installed as a console script.

```bash
# install / sync deps (creates .venv, installs the manga_dosei script)
uv sync

# run the full daily pipeline (sequence assembled in run_daily.STEPS;
# expands page-generation steps from each tool's PAGE_VARIANT_COUNT)
uv run manga_dosei YYYYMMDD            # e.g. 20260101

# interactive ADK web UI against root_agent (manga_dosei/agent.py)
# IMPORTANT: pass the same service URIs the CLI uses, otherwise adk web
# resolves .adk/ relative to AGENTS_DIR (and may fall back to in-memory),
# so sessions/artifacts written by `manga_dosei YYYYMMDD` will not appear.
# Absolute paths are required: file://./... is parsed with `.` as a host and
# rejected ("file:// artifact URIs must reference the local filesystem.").
uv run adk web \
  --session_service_uri="sqlite:///$(pwd)/.adk/sessions.db" \
  --artifact_service_uri="file://$(pwd)/.adk/artifacts" \
  .                                     # AGENTS_DIR = repo root (parent of manga_dosei/)
```

Notes on the URIs:
- `run_daily.py` uses `sqlite+aiosqlite:///./.adk/sessions.db` (async driver). `adk web` accepts the plain `sqlite://` SQLAlchemy URI; both point to the same SQLite file as long as `adk web` is started from the repo root.
- `--use_local_storage` is the default but resolves `.adk/` relative to `AGENTS_DIR`, which is **not necessarily the project root**. Passing the URIs explicitly with absolute paths (`$(pwd)/...`) is the only way to guarantee the web UI sees what the CLI wrote.

Linting / formatting (ruff):

```bash
uv run ruff check .                  # lint
uv run ruff check --fix .            # lint + auto-fix
uv run ruff format .                 # format
```

Ruff config lives in `pyproject.toml` (`[tool.ruff]` / `[tool.ruff.lint]`). There is no test suite at this time; do not introduce one without being asked.

## Required environment

`.env` (loaded automatically by `manga_dosei/__init__.py` via `python-dotenv`, then again at CLI entrypoint):

- `GEMINI_API_KEY` — Gemini text + image generation
- `GEMINI_TEXT_MODEL` (default `gemini-3.1-pro-preview`) — overrides `DEFAULT_TEXT_MODEL`
- `GEMINI_IMAGE_MODEL` (default `gemini-3-pro-image-preview`) — used by `generate_page_gemini` (`response_modalities=["IMAGE","TEXT"]`, `image_size="2K"`)
- `OPENAI_API_KEY` — OpenAI **Images Edit** API (`client.images.edit`), used by `generate_page_gpt`. Edit endpoint (not Generate) because it takes multiple reference images (layout sample + character refs) as inline input.
- `OPENAI_IMAGE_MODEL` (default `gpt-image-2`) — used by `generate_page_gpt` at `1024x1536` / `quality=high` / `output_format=png`
- `TAVILY_API_KEY` — required at import time of any tool that uses `build_tavily_toolset` (`fetch_dosei`, `enrich_news`)
- `WIKIMEDIA_CONTACT_EMAIL` — appended to the User-Agent for Wikimedia API calls (rate-limit etiquette)

`.env` is loaded with `override=False`, so values exported in the shell win.

## Architecture

### What this app does

Generates a one-page A4 manga summarizing the Japanese Prime Minister's daily schedule (`首相動静`, "shushou dōsei"). Each daily run produces, under the session for that `target_date`:

1. `dosei.md` — raw schedule article from JIJI.COM (www.jiji.com)
2. `news.md` — schedule + enriched background research (canonical source for character profiles)
3. `scenario.md` — manga script: page title, panel titles, 状況 + イラスト (visual instructions), dialogue verbatim, 登場人物一覧, X-post text. **Sole authority for narrative content.**
4. `assets/<name>.<ext>` + `manifests/assets.json` — Wikipedia reference images for characters/places
5. `assets/<name>.<ext>` resized to ≤1024px long side (new artifact versions, originals kept)
6. `layout.md` — **pure layout structure** derived from `scenario.md`: **`pattern_id` (chosen from the `assets/layouts/` catalog)**, ASCII page diagram (transcribed verbatim from the catalog), 段ごとの配置 (verbatim from catalog), per-panel 位置 (段 N 右側/左側/全幅) + キャラ配置 (画面左=A、画面右=B), and 描画前チェックリスト. **Does not contain panel titles, dialogue, or 視覚要素** — those stay in `scenario.md`. The `pattern_id` decides which canonical layout (and matching reference image) the downstream image-gen step uses.
7. `image_brief.md` — **single, self-contained input for image generation**, produced by `compose_image_brief` from `scenario.md` + `layout.md` + `news.md` + `manifests/assets.json`. Contains: `pattern_id` (transcribed from layout.md, drives image-gen sample selection), page header spec, 登場人物プロフィール (with 参照画像あり/なし flag from manifest), page layout (ASCII + 段配置 copied from layout.md), per-panel spec (位置 / キャラ配置 / 視覚要素 = scenario の 状況 + イラスト verbatim / 吹き出し verbatim), 描画前チェックリスト, page-footer disclaimer. Fields tagged `(verbatim)` are renderable text; everything else is a drawing hint and must not be rendered as on-page text.
8. `pages/<model>_<N>.<ext>` (e.g. `pages/gemini_1.jpg`, `pages/gpt_2.png`) — **independent generation attempts of the same one-page manga**, one set per image-generation backend. Each `generate_page_*` tool exposes a `PAGE_VARIANT_COUNT` constant and is called with `page_number=1..PAGE_VARIANT_COUNT` (currently gemini=5, gpt=2; counts vary because per-call quality/spread differs by model). The output is *not* a multi-page comic — best variant is picked manually. **The daily CLI currently invokes only `generate_page_gemini`** (see `run_daily.STEPS`); `generate_page_gpt` remains registered on `root_agent` so it is still callable via `adk web` / the interactive agent. Both image-gen tools read **only `image_brief.md`** for textual input; they do not load `scenario.md` or `layout.md` directly. They parse `pattern_id` out of `image_brief.md` and attach `assets/layouts/<pattern_id>/sample.jpg` as the 【ページ例】 reference (whose ASCII figure matches the brief's layout exactly).

### Two ways to drive the workflow

Both use the same `LlmAgent`s and the same `.adk/` storage; choose by use case:

- **`manga_dosei/run_daily.py` (CLI)** — deterministic. The CLI ignores the `root_agent`'s instruction-driven flow and instead loops over a hard-coded `STEPS` list, building one prompt per step that says "call tool `X` exactly once with these args". The CLI sequence is `fetch_dosei → enrich_news → generate_scenario → collect_assets → resize_assets → define_layout → compose_image_brief → generate_page_gemini × PAGE_VARIANT_COUNT`. `generate_page_gpt` is intentionally **not** in `STEPS` (Gemini gives more reliable composition and text rendering); to re-enable it, follow the comment block above `STEPS` in `run_daily.py`. Each step retries once on error (`RETRY_EXEMPT` lets specific tools opt out of CLI retry when they have internal retry; currently empty). State is persisted to `sqlite+aiosqlite:///./.adk/sessions.db`; artifacts to `.adk/artifacts/`.
- **`manga_dosei/agent.py` (`root_agent`)** — interactive. `instruction` tells the LLM to call `inspect_artifacts` first, then call exactly one upstream-most missing step per turn. The instruction's canonical order is `fetch_dosei → enrich_news → generate_scenario → collect_assets → resize_assets → define_layout → compose_image_brief → generate_page_gemini と generate_page_gpt をそれぞれ PAGE_VARIANT_COUNT 回`. Useful for `adk web`. **Do not change the workflow's canonical step order or the "one step per turn" rule** without project-owner approval — the same instruction also forbids changes to content-generating prompts (see `agent.py` "コンテンツ挙動の保全" section).

### How a tool/step is structured

Every workflow step in `manga_dosei/tools/` follows the same shape — internalize this before adding or modifying tools:

1. **Wrap an `LlmAgent` in an `AgentTool`** so the parent `root_agent` can invoke it as a tool. The agent uses `input_schema=StepInput` (just `target_date`) and, when its job is to produce text artifacts, `output_schema=StepOutput` (`body` for success, `error` for failure) plus `output_key="temp:<step>_output"`.
2. **`before_agent_callback` calls `prepare_step(...)` from `tools/_common.py`**: validates `target_date`, checks `required_artifacts`, optionally loads prior artifact text into `state["temp:..."]` keys that the `instruction` interpolates.
3. **`after_agent_callback` calls `save_step_output(...)` from `tools/_common.py`**: reads the structured output_key, persists `body` to the artifact filename, clears `last_error` on success, sets `last_error` on failure (so the CLI retry/abort path can see it).
4. **`instruction` is a callable** `(ReadonlyContext) -> str` that pulls `temp:target_date` and any prior-artifact text from state. This indirection is what lets the same `LlmAgent` be reused across `target_date`s.
5. **Failure contract**: on failure, return an "error" `Content` via `error_content` / `missing_content`, *and* call `record_last_error` so `state["last_error"]` is populated. The CLI checks `state["last_error"]` after every step to decide retry/abort. Do not rely on exceptions reaching the CLI — only exceptions raised outside the agent loop are caught by `_record_error` in `run_daily.py`.

Two exceptions that intentionally do not follow this template:

- **`inspect_artifacts`** and **`resize_assets`** are plain `async` functions taking `ToolContext` (registered as raw `FunctionTool`s on `root_agent`). They do not produce text artifacts — they read state or rewrite binary artifacts in place — so they don't need the `LlmAgent` + `StepOutput` indirection.
- **`enrich_news`** wraps a `SequentialAgent(researcher → LoopAgent(evaluator → escalation_checker → enhanced_searcher) → composer)` to do deep research before composing `news.md`. `LoopAgent.max_iterations=2`; `EscalationChecker` is a plain `BaseAgent` that reads `state[_EVAL_KEY]` and only emits `actions.escalate=True` when `grade == "pass"` (LLM-driven exit_loop was prone to early exits). The same `prepare_step`/`save_step_output` callbacks attach at the outer `SequentialAgent` level. Mirror this pattern (rather than inventing a new one) if another step needs iterative research.

### Context-window discipline

Several deliberate choices exist purely to prevent agent context blowup — do not undo them without thinking:

- **Tavily Search is a code-parameterized `FunctionTool` factory, not the MCP server.** `manga_dosei/tools/_tavily.py::make_tavily_search_tool` produces `FunctionTool`s whose `topic` / `search_depth` / `max_results` / `include_domains` / date range are fixed at construction time; the LLM only ever supplies `query`. `start_date_offset_from_target` / `end_date_offset_from_target` resolve target-date-relative bounds from `state["temp:target_date"]` at call time. Call sites: `fetch_dosei` (`search_jiji_for_dosei`, `start_date = target − 2d`), `enrich_news` (`search_news_jiji` and `search_news_yahoo`, both `end_date = target + 1d`). Tavily's `extract` endpoint is **not** wrapped — only `search` — because raw page text is what blew up context in the MCP-era flow.
- **Page-body retrieval goes through `summarize_url` (an AgentTool wrapping `fetch_url`), never raw extract.** `summarize_url` runs in a child `InMemorySessionService` (AgentTool's default), so only the focused summary returns to the parent researcher — the raw HTML/text stays in the child. `fetch_url` itself exists because Tavily's markdown conversion strips `<time datetime>` and similar semantic tags that JIJI.COM (www.jiji.com) needs for the `配信日時` line. `fetch_url` also defends downstream context by truncating the HTML download at 5MB and the cleaned-text body at 12,000 chars (BeautifulSoup-stripped, `article` → `main` → `body` preferred container).
- **`generate_page_gemini` / `generate_page_gpt` have no internal retry** and are called once per `page_number` by the CLI (count comes from each tool's `PAGE_VARIANT_COUNT`). Adding internal retry there would mean the CLI's outer retry runs them twice on each failure — preserve "internal retry XOR CLI retry" via `RETRY_EXEMPT` in `run_daily.py`.

### Storage layout

- `.adk/sessions.db` — `DatabaseSessionService` SQLite store. One session per `target_date` (`session_id == target_date`, `user_id == "daily"`, `app_name == "manga_dosei"`).
- `.adk/artifacts/` — `FileArtifactService` root. Files within are versioned (each `save_artifact` produces a new version; old ones are kept).
- `.adk/` is gitignored except for `.gitkeep`.
- `assets/samples/sanae.jpg` (in repo) is the **Takaichi Sanae character reference** baked into both `generate_page_gemini` and `generate_page_gpt`. Renaming or replacing silently changes generated output. `compose_image_brief` also has a hard-coded rule that injects 「高市早苗」 into the 登場人物プロフィール block (with `参照画像: あり (assets/samples/sanae.jpg)`) **even when she does not appear in `news.md`'s 主要人物プロフィール**, so the character reference is always wired up. (`assets/samples/sample.jpg` is no longer used by the pipeline — kept only because the READMEs embed it as the output sample.)
- `assets/layouts/<pattern_id>/` is the **layout pattern catalog** consumed by `define_layout` and `generate_page_*`. Each pattern directory contains `meta.json` (id / panels / name / when_to_use / rows), `ascii.txt` (canonical ASCII figure), and `sample.jpg` (per-pattern reference image with the matching layout). Current patterns: `3a` (1-1-1), `4a` (2x2), `4b` (1-1-1-1), `4c` (1-1-2), `4d` (1-2-1), `5a` (2-1-2). Adding a pattern is "drop a new `<id>/` directory with these three files"; no code changes needed.

### Conventions specific to this repo

- All tool docstrings are in 日本語 and are part of the contract — the parent agent reads them via the ADK `Tool` abstraction. When changing tool behavior, update the docstring's 前提 / 引数 / 返り値 sections too.
- `manga_dosei/validation.py::validate_target_date` only checks the 8-digit format (`\d{8}`); it does not verify calendar validity. Callers that need a real date (`_tavily.py::_parse_target_date`) build a `date(...)` from the slices and rely on that for calendar errors.
- `state["temp:..."]` keys are scratch (not persisted across invocations); non-`temp:` keys persist in the session DB. Use `temp:` for anything you only need within a single agent run.
- Artifact names are stable contracts between steps (`dosei.md`, `news.md`, `scenario.md`, `manifests/assets.json`, `assets/<name>.<ext>`, `layout.md`, `image_brief.md`, `pages/<model>_<N>.<ext>`). Renaming one requires updating every downstream `required_artifacts` tuple and `load_prior` map. `generate_page_*` tools load `image_brief.md` directly via `tool_context.load_artifact` (not through `load_prior`) and paste it verbatim into the image-gen prompt — they do not load `scenario.md` or `layout.md` themselves.
