import argparse
import asyncio
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.adk.agents.invocation_context import InvocationContext
from google.adk.artifacts import FileArtifactService
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.tools import ToolContext
from google.genai import types

from manga_dosei import APP_NAME, DEFAULT_USER_ID
from manga_dosei.agent import root_agent
from manga_dosei.tools import inspect_artifacts, resize_assets
from manga_dosei.tools.generate_page_gemini import (
    PAGE_VARIANT_COUNT as GEMINI_PAGE_VARIANT_COUNT,
)
from manga_dosei.validation import validate_target_date

# LLM をスキップして CLI から直接呼ぶ決定的 tool。
# LLM 経由ルート（adk web / interactive agent）は agent.py の
# root_agent.tools 経由で従来通り動作する。
_DIRECT_TOOLS: dict[str, tuple[Any, bool]] = {
    "inspect_artifacts": (inspect_artifacts, True),  # bool: target_date を引数で取るか
    "resize_assets": (resize_assets, True),
}

# NOTE: generate_page_gpt は ADK agent / web UI 経由では引き続き利用可能だが、
# 日次 CLI では呼び出さない方針（配置・文字品質ともに Gemini の方が安定するため）。
# 再有効化する場合: 上の import で
# `from manga_dosei.tools.generate_page_gpt import PAGE_VARIANT_COUNT
#     as GPT_PAGE_VARIANT_COUNT` を追加し、STEPS に
# `*_page_steps("generate_page_gpt", GPT_PAGE_VARIANT_COUNT)` を戻す。

SESSION_URI = "sqlite+aiosqlite:///./.adk/sessions.db"
ARTIFACT_DIR = Path(".adk/artifacts")

_SESSION_ID_PATTERN = re.compile(r"\d{8}(_.+)?")

_Step = tuple[str, dict[str, object]]


def _page_steps(tool_name: str, count: int) -> list[_Step]:
    return [(tool_name, {"page_number": n}) for n in range(1, count + 1)]


STEPS: list[_Step] = [
    ("fetch_dosei", {}),
    ("enrich_news", {}),
    ("generate_scenario", {}),
    ("collect_assets", {}),
    ("resize_assets", {}),
    ("define_layout", {}),
    ("compose_image_brief", {}),
    *_page_steps("generate_page_gemini", GEMINI_PAGE_VARIANT_COUNT),
]

# CLI レベルで retry しないツール。ツール内部で retry を持っているものを
# ここに入れる（多重 retry 防止）。現状は該当なし。
RETRY_EXEMPT: set[str] = set()


def main() -> None:
    load_dotenv(dotenv_path=Path(".env"), override=False)
    args = _parse_args()
    target_date = args.target_date
    session_id = args.session_id or target_date
    validate_target_date(target_date)
    _validate_session_id(session_id, target_date)
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
            "Must match ^\\d{8}(_.+)?$ and start with target_date "
            "(e.g. 20260315_retry)."
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
            f"invalid --session-id={session_id!r}: must match ^\\d{{8}}(_.+)?$"
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
    Path(".adk").mkdir(parents=True, exist_ok=True)
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

    for tool_name, extra_args in STEPS:
        retry = tool_name not in RETRY_EXEMPT
        success = await _run_step_with_retry(
            runner,
            artifact_service,
            session_service,
            target_date,
            session_id,
            tool_name,
            extra_args=extra_args,
            retry=retry,
        )
        if not success:
            err = await _last_error(session_service, session_id)
            label = _step_label(tool_name, extra_args)
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
        data = _extract_part_bytes(part)
        if data is None:
            continue
        dest = publish_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        written += 1
    print(f"[publish_dir] wrote {written} artifact(s) under {publish_dir}")


def _extract_part_bytes(part: types.Part | None) -> bytes | None:
    if part is None:
        return None
    if part.inline_data is not None and part.inline_data.data:
        return part.inline_data.data
    if part.text is not None:
        return part.text.encode("utf-8")
    return None


async def _ensure_session(
    session_service: DatabaseSessionService,
    target_date: str,
    session_id: str,
) -> None:
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
        session_id=session_id,
    )
    if session is not None:
        return
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
        session_id=session_id,
        state={
            "target_date": target_date,
            "status": "initialized",
            "last_error": None,
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
    direct = tool_name in _DIRECT_TOOLS and not extra_args
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
    fn, needs_target_date = _DIRECT_TOOLS[tool_name]
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
        session_id=session_id,
    )
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

    if needs_target_date:
        result = await fn(target_date, tool_context)
    else:
        result = await fn(tool_context)

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
) -> Any:
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
        session_id=session_id,
    )
    if session is None:
        return None
    return session.state.get("last_error")


async def _record_error(
    session_service: DatabaseSessionService,
    session_id: str,
    step: str,
    message: str,
) -> None:
    """ツール実行が例外で死んだ場合に、retry / 終了判定が機能するよう
    state に記録する。"""
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
        session_id=session_id,
    )
    if session is None:
        return
    await session_service.append_event(
        session,
        Event(
            invocation_id=f"cli-error-{session_id}",
            author="run_daily",
            actions=EventActions(
                state_delta={"last_error": {"step": step, "message": message}}
            ),
        ),
    )


if __name__ == "__main__":
    main()
