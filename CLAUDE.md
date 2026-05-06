# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

# run the full daily pipeline (sequential 10 steps)
uv run manga_dosei YYYYMMDD            # e.g. 20260101

# interactive ADK web UI against root_agent (manga_dosei/agent.py)
# IMPORTANT: pass the same service URIs the CLI uses, otherwise adk web
# resolves .adk/ relative to AGENTS_DIR (and may fall back to in-memory),
# so sessions/artifacts written by `manga_dosei YYYYMMDD` will not appear.
uv run adk web \
  --session_service_uri='sqlite:///./.adk/sessions.db' \
  --artifact_service_uri='file://./.adk/artifacts' \
  .                                     # AGENTS_DIR = repo root (parent of manga_dosei/)
```

Notes on the URIs:
- `run_daily.py` uses `sqlite+aiosqlite:///./.adk/sessions.db` (async driver). `adk web` accepts the plain `sqlite://` SQLAlchemy URI; both point to the same SQLite file.
- `--use_local_storage` is the default but resolves `.adk/` relative to `AGENTS_DIR`, which is **not necessarily the project root**. Passing the URIs explicitly (with `./` to anchor at CWD) is the only way to guarantee the web UI sees what the CLI wrote.

There is no test suite, linter config, or formatter config in the repo at this time. Do not introduce one without being asked.

## Required environment

`.env` (loaded automatically by `manga_dosei/__init__.py` via `python-dotenv`, then again at CLI entrypoint):

- `GEMINI_API_KEY` — Gemini text + image generation
- `GEMINI_TEXT_MODEL` (default `gemini-3.1-pro-preview`) — overrides `DEFAULT_TEXT_MODEL`
- `GEMINI_IMAGE_MODEL` (default `gemini-3-pro-image-preview`) — used by `generate_page_gemini`
- `TAVILY_API_KEY` — required at import time of any tool that uses `build_tavily_toolset` (`fetch_dosei`, `enrich_news`)
- `WIKIMEDIA_CONTACT_EMAIL` — appended to the User-Agent for Wikimedia API calls (rate-limit etiquette)

`.env` is loaded with `override=False`, so values exported in the shell win.

## Architecture

### What this app does

Generates a one-page A4 manga summarizing the Japanese Prime Minister's daily schedule (`首相動静`, "shushou dōsei"). Each daily run produces, under the session for that `target_date`:

1. `dosei.md` — raw schedule article from jiji.com
2. `news.md` — schedule + enriched background research
3. `scenario.md` — manga script (panel-by-panel, 4–5 panels per page)
4. `assets/<name>.<ext>` + `manifests/assets.json` — Wikipedia reference images for characters/places
5. `assets/<name>.<ext>` resized to ≤1024px long side (new artifact versions, originals kept)
6. `pages/page_1.png` … `pages/page_5.png` — **five independent generation attempts of the same one-page manga**. The CLI calls `generate_page_gemini` with `page_number=1..5` to produce variants for quality (best is picked manually); the output is *not* a 5-page comic.

### Two ways to drive the workflow

Both use the same `LlmAgent`s and the same `.adk/` storage; choose by use case:

- **`manga_dosei/run_daily.py` (CLI)** — deterministic. The CLI ignores the `root_agent`'s instruction-driven flow and instead loops over a hard-coded `STEPS` list, building one prompt per step that says "call tool `X` exactly once with these args". Each step retries once on error (`RETRY_EXEMPT` lets specific tools opt out of CLI retry when they have internal retry). State is persisted to `sqlite+aiosqlite:///./.adk/sessions.db`; artifacts to `.adk/artifacts/`.
- **`manga_dosei/agent.py` (`root_agent`)** — interactive. `instruction` tells the LLM to call `inspect_artifacts` first, then call exactly one upstream-most missing step per turn. Useful for `adk web`. **Do not change the workflow's canonical step order or the "one step per turn" rule** without project-owner approval — the same instruction also forbids changes to content-generating prompts (see `agent.py` "コンテンツ挙動の保全" section).

### How a tool/step is structured

Every workflow step in `manga_dosei/tools/` follows the same shape — internalize this before adding or modifying tools:

1. **Wrap an `LlmAgent` in an `AgentTool`** so the parent `root_agent` can invoke it as a tool. The agent uses `input_schema=StepInput` (just `target_date`) and, when its job is to produce text artifacts, `output_schema=StepOutput` (`body` for success, `error` for failure) plus `output_key="temp:<step>_output"`.
2. **`before_agent_callback` calls `prepare_step(...)` from `tools/_common.py`**: validates `target_date`, checks `required_artifacts`, optionally loads prior artifact text into `state["temp:..."]` keys that the `instruction` interpolates.
3. **`after_agent_callback` calls `save_step_output(...)` from `tools/_common.py`**: reads the structured output_key, persists `body` to the artifact filename, clears `last_error` on success, sets `last_error` on failure (so the CLI retry/abort path can see it).
4. **`instruction` is a callable** `(ReadonlyContext) -> str` that pulls `temp:target_date` and any prior-artifact text from state. This indirection is what lets the same `LlmAgent` be reused across `target_date`s.
5. **Failure contract**: on failure, return an "error" `Content` via `error_content` / `missing_content`, *and* call `record_last_error` so `state["last_error"]` is populated. The CLI checks `state["last_error"]` after every step to decide retry/abort. Do not rely on exceptions reaching the CLI — only exceptions raised outside the agent loop are caught by `_record_error` in `run_daily.py`.

Two exceptions that intentionally do not follow this template:

- **`inspect_artifacts`** and **`resize_assets`** are plain `async` functions taking `ToolContext` (registered as raw `FunctionTool`s on `root_agent`). They do not produce text artifacts — they read state or rewrite binary artifacts in place — so they don't need the `LlmAgent` + `StepOutput` indirection.
- **`enrich_news`** wraps a `SequentialAgent(researcher → LoopAgent(evaluator → escalation_checker → enhanced_searcher) → composer)` to do deep research before composing `news.md`. The same `prepare_step`/`save_step_output` callbacks attach at the outer `SequentialAgent` level. Mirror this pattern (rather than inventing a new one) if another step needs iterative research.

### Context-window discipline

Several deliberate choices exist purely to prevent agent context blowup — do not undo them without thinking:

- **`tavily_extract` is wrapped in a `summarize_url` AgentTool** (see `enrich_news.py`). Calling `tavily_extract` directly leaks `raw_content` (full page text) into the parent agent's history and exceeded the 1M-token context. The `AgentTool` runs the extract in a child session (InMemorySessionService), so only the summary returns to the parent.
- **`build_tavily_toolset(tool_filter=["tavily_search"])`** in `fetch_dosei` deliberately omits `tavily_extract` for the same reason. `fetch_url` (HTML → boilerplate-stripped visible text, capped to 12k chars) is used instead when the agent actually needs a page body — this is also why `fetch_url` exists at all: Tavily's markdown conversion strips `<time datetime>` and similar semantic tags that jiji.com requires for the `配信日時` line.
- **`generate_page_gemini` has no internal retry** and is called once per `page_number` (1..5) by the CLI. Adding internal retry there would mean the CLI's outer retry runs it twice on each failure — preserve "internal retry XOR CLI retry" via `RETRY_EXEMPT` in `run_daily.py`.

### Storage layout

- `.adk/sessions.db` — `DatabaseSessionService` SQLite store. One session per `target_date` (`session_id == target_date`, `user_id == "daily"`, `app_name == "manga_dosei"`).
- `.adk/artifacts/` — `FileArtifactService` root. Files within are versioned (each `save_artifact` produces a new version; old ones are kept).
- `.adk/` is gitignored except for `.gitkeep`.
- `assets/samples/` (in repo, not under `.adk/`) holds the two **reference images** baked into `generate_page_gemini`'s prompt: `sample.jpg` (page-layout exemplar) and `sanae.jpg` (Takaichi Sanae character ref). Replacing or renaming these silently changes generated output.

### Conventions specific to this repo

- All tool docstrings are in 日本語 and are part of the contract — the parent agent reads them via the ADK `Tool` abstraction. When changing tool behavior, update the docstring's 前提 / 引数 / 返り値 sections too.
- `state["temp:..."]` keys are scratch (not persisted across invocations); non-`temp:` keys persist in the session DB. Use `temp:` for anything you only need within a single agent run.
- Artifact names are stable contracts between steps (`dosei.md`, `news.md`, `scenario.md`, `manifests/assets.json`, `assets/<name>.<ext>`, `pages/page_NN.png`). Renaming one requires updating every downstream `required_artifacts` tuple and `load_prior` map.
