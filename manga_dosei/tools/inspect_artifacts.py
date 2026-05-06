"""inspect_artifacts: 現在のセッションの artifacts と state を一覧する。"""

from typing import Any

from google.adk.tools import ToolContext

from manga_dosei.validation import validate_target_date


async def inspect_artifacts(target_date: str, tool_context: ToolContext) -> dict[str, Any]:
    """現在のセッションの artifacts と state を一覧する。

    前提:
        なし。

    用途:
        ワークフローを進める前の状況確認。どの artifact が既に存在するかを見て、
        次に呼ぶべきステップを判断するために使う。

    返り値:
        {
            "status": "success",
            "target_date": str,
            "artifacts": list[str],   # 既存 artifact ファイル名（昇順）
            "state": dict,            # 現在のセッション state
        }
    """
    try:
        validate_target_date(target_date)
    except ValueError as error:
        return {
            "status": "error",
            "step": "inspect_artifacts",
            "message": str(error),
        }

    artifact_names = await tool_context.list_artifacts()
    return {
        "status": "success",
        "target_date": target_date,
        "artifacts": sorted(artifact_names),
        "state": tool_context.state.to_dict(),
    }
