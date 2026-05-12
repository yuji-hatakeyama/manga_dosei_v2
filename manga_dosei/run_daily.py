import asyncio
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.adk.artifacts import FileArtifactService
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

from manga_dosei import APP_NAME, DEFAULT_USER_ID
from manga_dosei.agent import root_agent
from manga_dosei.tools.generate_page_gemini import (
    PAGE_VARIANT_COUNT as GEMINI_PAGE_VARIANT_COUNT,
)
from manga_dosei.tools.generate_page_gpt import (
    PAGE_VARIANT_COUNT as GPT_PAGE_VARIANT_COUNT,
)
from manga_dosei.validation import validate_target_date

SESSION_URI = "sqlite+aiosqlite:///./.adk/sessions.db"
ARTIFACT_DIR = Path(".adk/artifacts")

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
    *_page_steps("generate_page_gemini", GEMINI_PAGE_VARIANT_COUNT),
    *_page_steps("generate_page_gpt", GPT_PAGE_VARIANT_COUNT),
]

# CLI レベルで retry しないツール。ツール内部で retry を持っているものを
# ここに入れる（多重 retry 防止）。現状は該当なし。
RETRY_EXEMPT: set[str] = set()


def main() -> None:
    load_dotenv(dotenv_path=Path(".env"), override=False)
    if len(sys.argv) != 2:
        raise SystemExit("usage: manga_dosei YYYYMMDD")
    target_date = sys.argv[1]
    validate_target_date(target_date)
    asyncio.run(_run(target_date))


async def _run(target_date: str) -> None:
    Path(".adk").mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    session_service = DatabaseSessionService(db_url=SESSION_URI)
    artifact_service = FileArtifactService(root_dir=str(ARTIFACT_DIR))
    await _ensure_session(session_service, target_date)

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
            session_service,
            target_date,
            tool_name,
            extra_args=extra_args,
            retry=retry,
        )
        if not success:
            err = await _last_error(session_service, target_date)
            label = _step_label(tool_name, extra_args)
            print(
                f"[{label}] failed; aborting. last_error={err}",
                file=sys.stderr,
            )
            sys.exit(1)


async def _ensure_session(
    session_service: DatabaseSessionService,
    target_date: str,
) -> None:
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
        session_id=target_date,
    )
    if session is not None:
        return
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
        session_id=target_date,
        state={
            "target_date": target_date,
            "status": "initialized",
            "last_error": None,
        },
    )


async def _run_step_with_retry(
    runner: Runner,
    session_service: DatabaseSessionService,
    target_date: str,
    tool_name: str,
    *,
    extra_args: dict[str, object],
    retry: bool,
) -> bool:
    attempts = 2 if retry else 1
    label = _step_label(tool_name, extra_args)
    for attempt in range(1, attempts + 1):
        try:
            await _run_step(
                runner,
                target_date,
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
            await _record_error(session_service, target_date, tool_name, repr(error))
        if not await _last_error(session_service, target_date):
            return True
        if attempt < attempts:
            print(
                f"[{label}] error on attempt {attempt}; retrying once",
                file=sys.stderr,
            )
    return False


async def _run_step(
    runner: Runner,
    target_date: str,
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
        session_id=target_date,
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
    target_date: str,
) -> Any:
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
        session_id=target_date,
    )
    if session is None:
        return None
    return session.state.get("last_error")


async def _record_error(
    session_service: DatabaseSessionService,
    target_date: str,
    step: str,
    message: str,
) -> None:
    """ツール実行が例外で死んだ場合に、retry / 終了判定が機能するよう
    state に記録する。"""
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
        session_id=target_date,
    )
    if session is None:
        return
    await session_service.append_event(
        session,
        Event(
            invocation_id=f"cli-error-{target_date}",
            author="run_daily",
            actions=EventActions(
                state_delta={"last_error": {"step": step, "message": message}}
            ),
        ),
    )


if __name__ == "__main__":
    main()
