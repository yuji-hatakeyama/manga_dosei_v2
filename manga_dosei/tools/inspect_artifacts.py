"""inspect_artifacts: 現在のセッションの artifacts と state を一覧する。"""

from typing import Any

from google.adk.tools import ToolContext

from manga_dosei.names import persistent_state
from manga_dosei.tools._state import error_result, ok_result
from manga_dosei.validation import validate_target_date

_STEP = "inspect_artifacts"


async def inspect_artifacts(
    target_date: str, tool_context: ToolContext
) -> dict[str, Any]:
    """現在のセッションの artifacts と state を一覧する。

    前提:
        なし。

    用途:
        ワークフローを進める前の状況確認。どの artifact が既に存在するかを見て、
        次に呼ぶべきステップを判断するために使う。

    返り値:
        {
            "status": "success",
            "step": "inspect_artifacts",
            "message": str,
            "target_date": str,
            "artifacts": list[str],   # 既存 artifact ファイル名（昇順）
            "state": dict,            # 現在のセッション state
        }
    """
    # NOTE: read-only inspection — never touch state['last_error'] on any path,
    # so the previous step's failure context survives for the CLI retry/abort
    # decision (AGENTS.md inspect_artifacts contract).
    try:
        validate_target_date(target_date)
    except ValueError as error:
        return error_result(_STEP, str(error))

    artifact_names = await tool_context.list_artifacts()
    sorted_names = sorted(artifact_names)
    payload = ok_result(_STEP, f"{len(sorted_names)} artifacts")
    payload["target_date"] = target_date
    payload["artifacts"] = sorted_names
    payload["state"] = persistent_state(tool_context.state.to_dict())
    return payload
