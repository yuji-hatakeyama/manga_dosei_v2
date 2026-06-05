import argparse
import asyncio
import json
import re
import sys
import traceback
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, NamedTuple

from dotenv import load_dotenv
from google.adk.agents.invocation_context import InvocationContext
from google.adk.artifacts import FileArtifactService
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService, Session
from google.adk.tools import ToolContext
from google.genai import types

from manga_dosei import APP_NAME, DEFAULT_USER_ID
from manga_dosei.agent import root_agent
from manga_dosei.config import get_settings
from manga_dosei.names import StateKey
from manga_dosei.tools import inspect_artifacts, resize_assets
from manga_dosei.tools._state import LastError
from manga_dosei.tools.generate_page_gemini import (
    PAGE_VARIANT_COUNT as GEMINI_PAGE_VARIANT_COUNT,
)
from manga_dosei.validation import validate_target_date

# Direct tools widen the return dict beyond the base status/step/message
# shape, so type as dict[str, Any].
DirectFn = Callable[[str, ToolContext], Awaitable[dict[str, Any]]]

# LLM をスキップして CLI から直接呼ぶ決定的 tool。adk web / interactive agent
# 経由ルートは agent.py の root_agent.tools 経由で引き続き利用可能。
_DIRECT_TOOLS: dict[str, DirectFn] = {
    "inspect_artifacts": inspect_artifacts,
    "resize_assets": resize_assets,
}


class StepSpec(NamedTuple):
    tool_name: str
    extra_args: dict[str, object]
    retry_exempt: bool = False


# `.adk/` を CWD 相対にすると、サブディレクトリから CLI を起動した瞬間に
# 別 store (`<subdir>/.adk/...`) ができてしまい、翌日の run / `adk web` /
# `--publish-dir` がサイレントに別 store を参照する。`Path(".adk").mkdir()`
# はどこでも成功するので失敗が観測できない。これを避けるため、SQLite URL
# と artifact root をリポジトリルート (= パッケージの親) にアンカーする。
# 同じアンカーを `adk web` 起動コマンドにも使う運用 (AGENTS.md 参照)。
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ADK_DIR = _REPO_ROOT / ".adk"
SESSION_URI = f"sqlite+aiosqlite:///{_ADK_DIR / 'sessions.db'}"
ARTIFACT_DIR = _ADK_DIR / "artifacts"

# session_id は SQLite の行 id だけでなく FileArtifactService のディレクトリ
# 名にも使われるので、`/` / `..` / 空白などをサフィックスから排除して
# ファイルシステム安全な文字種に限定する。
_SESSION_ID_REGEX = r"\d{8}(_[A-Za-z0-9_-]+)?"
_SESSION_ID_PATTERN = re.compile(_SESSION_ID_REGEX)


# NOTE: generate_page_gpt は ADK agent / web UI 経由では引き続き利用可能だが、
# 日次 CLI では呼び出さない (配置・文字品質ともに Gemini の方が安定するため)。
# 再有効化する場合: 上の import で
# `from manga_dosei.tools.generate_page_gpt import PAGE_VARIANT_COUNT
#     as GPT_PAGE_VARIANT_COUNT` を追加し、STEPS に
# `*[StepSpec("generate_page_gpt", {"page_number": n})
#     for n in range(1, GPT_PAGE_VARIANT_COUNT + 1)]` を戻す。
STEPS: list[StepSpec] = [
    StepSpec("fetch_dosei", {}),
    StepSpec("enrich_news", {}),
    StepSpec("generate_scenario", {}),
    StepSpec("collect_assets", {}),
    StepSpec("resize_assets", {}),
    StepSpec("define_layout", {}),
    StepSpec("compose_image_brief", {}),
    *[
        StepSpec("generate_page_gemini", {"page_number": n})
        for n in range(1, GEMINI_PAGE_VARIANT_COUNT + 1)
    ],
]


def main() -> None:
    args = _parse_args()
    target_date = args.target_date
    session_id = args.session_id or target_date
    validate_target_date(target_date)
    _validate_session_id(session_id, target_date)
    # NOTE: `manga_dosei/__init__.py` already loads `<repo>/.env`, but a per-run
    # CWD `.env` (CI rotation, debug overrides) needs a second pass. Both calls
    # use override=False so the priority stays shell > CWD `.env` > repo `.env`.
    # Invalidate get_settings's lru_cache so values land in Settings.
    load_dotenv(override=False)
    get_settings.cache_clear()
    # 起動時に env を解決させ、`.env` 不備を最初のツール呼び出しまで遅延させない。
    get_settings()
    publish_dir = Path(args.publish_dir) if args.publish_dir else None
    asyncio.run(_run(target_date, session_id, publish_dir=publish_dir))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="manga_dosei",
        description="Run the manga dosei daily content workflow.",
    )
    parser.add_argument("target_date", help="YYYYMMDD")
    parser.add_argument(
        "--session-id",
        default=None,
        help=(
            "ADK session id. Defaults to target_date. "
            f"Must match ^{_SESSION_ID_REGEX}$ and start with target_date. "
            "Suffix allows only [A-Za-z0-9_-] so the value is safe to use as "
            "a filesystem path component (e.g. 20260315_retry)."
        ),
    )
    parser.add_argument(
        "--publish-dir",
        default=None,
        help=(
            "If set, after the pipeline completes, write the latest version "
            "of every artifact in the session under this directory. "
            "Artifact names retain their slash hierarchy "
            "(e.g. pages/gemini_1.jpg lands at <publish-dir>/pages/gemini_1.jpg). "
            "The directory is created if missing."
        ),
    )
    return parser.parse_args()


def _validate_session_id(session_id: str, target_date: str) -> None:
    if _SESSION_ID_PATTERN.fullmatch(session_id) is None:
        raise SystemExit(
            f"invalid --session-id={session_id!r}: must match ^{_SESSION_ID_REGEX}$"
        )
    if session_id[:8] != target_date:
        raise SystemExit(
            f"invalid --session-id={session_id!r}: "
            f"first 8 digits must equal target_date={target_date}"
        )


async def _run(
    target_date: str,
    session_id: str,
    *,
    publish_dir: Path | None,
) -> None:
    _ADK_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    session_service = DatabaseSessionService(db_url=SESSION_URI)
    artifact_service = FileArtifactService(root_dir=str(ARTIFACT_DIR))
    await _ensure_session(session_service, target_date, session_id)

    runner = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        artifact_service=artifact_service,
        session_service=session_service,
    )

    for step in STEPS:
        retry = not step.retry_exempt
        success = await _run_step_with_retry(
            runner,
            artifact_service,
            session_service,
            target_date,
            session_id,
            step.tool_name,
            extra_args=step.extra_args,
            retry=retry,
        )
        if not success:
            err = await _last_error(session_service, session_id)
            label = _step_label(step.tool_name, step.extra_args)
            print(
                f"[{label}] failed; aborting. last_error={err}",
                file=sys.stderr,
            )
            sys.exit(1)

    if publish_dir is not None:
        await _dump_artifacts_to_dir(artifact_service, session_id, publish_dir)


async def _dump_artifacts_to_dir(
    artifact_service: FileArtifactService,
    session_id: str,
    publish_dir: Path,
) -> None:
    """セッション内の全 artifact (最新 version のみ) を `publish_dir` 配下に書き出す。

    artifact 名のスラッシュ階層はそのまま保持し、衝突回避用の prefix も付けない。
    例: `pages/gemini_1.jpg` は `<publish_dir>/pages/gemini_1.jpg` に書く。
    バイナリ artifact (inline_data) は bytes をそのまま、テキスト artifact (text) は
    UTF-8 でエンコードして書き出す。
    """
    publish_dir.mkdir(parents=True, exist_ok=True)
    keys = await artifact_service.list_artifact_keys(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
        session_id=session_id,
    )
    written = 0
    for name in sorted(keys):
        part = await artifact_service.load_artifact(
            app_name=APP_NAME,
            user_id=DEFAULT_USER_ID,
            session_id=session_id,
            filename=name,
        )
        data = _extract_part_bytes(part, name=name)
        if data is None:
            # NOTE: 失敗したまま空 Part を残した artifact が `wrote N` カウント
            # だけで隠れないよう、欠落を必ず stderr に出す。
            print(
                f"[publish_dir] skipping {name}: no data to write",
                file=sys.stderr,
            )
            continue
        dest = publish_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        written += 1
    print(f"[publish_dir] wrote {written} artifact(s) under {publish_dir}")


def _extract_part_bytes(part: types.Part | None, *, name: str) -> bytes | None:
    if part is None:
        return None
    if part.inline_data is not None and part.inline_data.data:
        return part.inline_data.data
    # NOTE: ADK can round-trip text artifacts through google-genai Content with
    # an empty inline_data Blob and the real text in `part.text`; fall through
    # to the text branch when inline_data is present-but-empty.
    if part.text:
        return part.text.encode("utf-8")
    if part.inline_data is not None:
        print(
            f"[publish_dir] {name}: inline_data present but empty",
            file=sys.stderr,
        )
    return None


async def _get_session(
    session_service: DatabaseSessionService,
    session_id: str,
) -> Session | None:
    # NOTE: APP_NAME / DEFAULT_USER_ID are CLI-wide constants; this helper
    # exists so a future multi-tenant switch is one edit.
    return await session_service.get_session(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
        session_id=session_id,
    )


async def _ensure_session(
    session_service: DatabaseSessionService,
    target_date: str,
    session_id: str,
) -> None:
    session = await _get_session(session_service, session_id)
    if session is not None:
        return
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
        session_id=session_id,
        state={
            StateKey.TARGET_DATE: target_date,
            StateKey.STATUS: "initialized",
            StateKey.LAST_ERROR: None,
        },
    )


async def _run_step_with_retry(
    runner: Runner,
    artifact_service: FileArtifactService,
    session_service: DatabaseSessionService,
    target_date: str,
    session_id: str,
    tool_name: str,
    *,
    extra_args: dict[str, object],
    retry: bool,
) -> bool:
    attempts = 2 if retry else 1
    label = _step_label(tool_name, extra_args)
    direct = tool_name in _DIRECT_TOOLS
    if direct and extra_args:
        # `_DIRECT_TOOLS` の正規シグネチャは `(target_date, tool_context)` のみ。
        # extra_args を渡せる窓口は無いので、サイレントに LLM 経路へ落とさず
        # 登録ミスとして即時失敗させる。
        raise RuntimeError(
            f"direct tool {tool_name} cannot take extra_args; got {extra_args!r}"
        )
    for attempt in range(1, attempts + 1):
        try:
            if direct:
                await _run_step_direct(
                    session_service,
                    artifact_service,
                    target_date,
                    session_id,
                    tool_name,
                    attempt=attempt,
                )
            else:
                await _run_step(
                    runner,
                    target_date,
                    session_id,
                    tool_name,
                    extra_args=extra_args,
                    attempt=attempt,
                )
        except Exception as error:
            print(
                f"[{label}] (attempt {attempt}) unhandled error: {error!r}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            await _record_error(session_service, session_id, tool_name, repr(error))
        if not await _last_error(session_service, session_id):
            return True
        if attempt < attempts:
            print(
                f"[{label}] error on attempt {attempt}; retrying once",
                file=sys.stderr,
            )
    return False


async def _run_step_direct(
    session_service: DatabaseSessionService,
    artifact_service: FileArtifactService,
    target_date: str,
    session_id: str,
    tool_name: str,
    *,
    attempt: int,
) -> None:
    """LLM をスキップして決定的 tool を直接呼び、結果を session.state に永続化する。

    各 tool は ToolContext.state 経由で `last_error` / `status` 等を書き換える。
    これは `event_actions.state_delta` に溜まるので、最後に append_event で
    session に commit する。こうしないと既存の retry 判定 (`_last_error`) が
    変更を観測できない。
    """
    direct_fn = _DIRECT_TOOLS[tool_name]
    session = await _get_session(session_service, session_id)
    if session is None:
        raise RuntimeError(f"session {session_id} not found")

    invocation_id = f"cli-direct-{session_id}-{tool_name}-{attempt}"
    invocation_context = InvocationContext(
        session_service=session_service,
        artifact_service=artifact_service,
        invocation_id=invocation_id,
        agent=root_agent,
        session=session,
    )
    event_actions = EventActions()
    tool_context = ToolContext(invocation_context, event_actions=event_actions)

    result = await direct_fn(target_date, tool_context)

    await session_service.append_event(
        session,
        Event(
            invocation_id=invocation_id,
            author="run_daily",
            actions=event_actions,
        ),
    )

    label = _step_label(tool_name, {})
    suffix = f" (attempt {attempt})" if attempt > 1 else ""
    print(f"[{label}]{suffix} {json.dumps(result, ensure_ascii=False)}")


async def _run_step(
    runner: Runner,
    target_date: str,
    session_id: str,
    tool_name: str,
    *,
    extra_args: dict[str, object],
    attempt: int,
) -> None:
    label = _step_label(tool_name, extra_args)
    prompt = _build_step_prompt(tool_name, target_date, extra_args)
    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    final_text = None
    async for event in runner.run_async(
        user_id=DEFAULT_USER_ID,
        session_id=session_id,
        new_message=message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(part.text or "" for part in event.content.parts)
    suffix = f" (attempt {attempt})" if attempt > 1 else ""
    print(f"[{label}]{suffix} {final_text or ''}")


def _build_step_prompt(
    tool_name: str,
    target_date: str,
    extra_args: dict[str, object],
) -> str:
    """root_agent に対する 1 ステップ実行指示を組み立てる。"""
    if not extra_args:
        return (
            f"Call the {tool_name} tool exactly once for target_date={target_date}. "
            "Report the tool result."
        )
    args = {"target_date": target_date, **extra_args}
    args_json = json.dumps(args, ensure_ascii=False)
    return (
        f"Call the {tool_name} tool exactly once with the following arguments "
        f"as JSON: {args_json}. Report the tool result."
    )


def _step_label(tool_name: str, extra_args: dict[str, object]) -> str:
    if not extra_args:
        return tool_name
    extra = ", ".join(f"{k}={v}" for k, v in extra_args.items())
    return f"{tool_name}({extra})"


async def _last_error(
    session_service: DatabaseSessionService,
    session_id: str,
) -> LastError | None:
    session = await _get_session(session_service, session_id)
    if session is None:
        return None
    return session.state.get(StateKey.LAST_ERROR)


async def _record_error(
    session_service: DatabaseSessionService,
    session_id: str,
    step: str,
    message: str,
) -> None:
    """ツール実行が例外で死んだ場合に、retry / 終了判定が機能するよう
    state に記録する。"""
    session = await _get_session(session_service, session_id)
    if session is None:
        return
    payload: LastError = {"step": step, "message": message}
    await session_service.append_event(
        session,
        Event(
            invocation_id=f"cli-error-{session_id}",
            author="run_daily",
            actions=EventActions(state_delta={StateKey.LAST_ERROR: payload}),
        ),
    )


if __name__ == "__main__":
    main()
