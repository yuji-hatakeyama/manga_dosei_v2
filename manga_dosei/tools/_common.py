"""Shared helpers for ADK workflow tools."""

import json
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.genai import types
from pydantic import BaseModel, ValidationError

from manga_dosei.validation import validate_target_date


class StepInput(BaseModel):
    """全 workflow step 共通の AgentTool 入力スキーマ。"""

    target_date: str


class StepOutput(BaseModel):
    """全 workflow step 共通の LLM 出力スキーマ (LlmAgent.output_schema 用)。

    LLM はこのどちらかで応答する:
    - 成功時: body=生成 markdown 本文 (error は空)
    - 失敗時: error=失敗理由 (body は空、artifact は保存されない)
    """

    body: str = ""
    error: str = ""


def parse_target_date_input(callback_context: CallbackContext) -> str | None:
    """AgentTool が user_content の text に JSON で詰めた引数から target_date を取り出す。

    input_schema=StepInput により形式は保証されているので、JSON parse と
    Pydantic validation のみ行い、失敗したら None を返す。
    """
    user_content = callback_context.user_content
    if not user_content or not user_content.parts:
        return None
    text = (user_content.parts[0].text or "").strip()
    if not text:
        return None
    try:
        return StepInput.model_validate_json(text).target_date
    except ValidationError:
        return None


def status_content(payload: dict[str, Any]) -> types.Content:
    """Status dict を JSON 文字列として持つ model Content を作る。

    AgentTool は内部 agent の最終応答テキストを呼び出し元に返す。
    呼び出し側 LLM はこの JSON 文字列を読み取って構造化情報として扱える。
    """
    return types.Content(
        role="model",
        parts=[types.Part(text=json.dumps(payload, ensure_ascii=False))],
    )


def error_content(step: str, message: str) -> types.Content:
    return status_content({"status": "error", "step": step, "message": message})


def missing_content(step: str, missing: list[str]) -> types.Content:
    return status_content(
        {
            "status": "error",
            "step": step,
            "message": "required artifacts are missing",
            "missing_artifacts": missing,
        }
    )


async def save_text_artifact(
    callback_context: CallbackContext,
    filename: str,
    text: str,
    *,
    mime_type: str = "text/markdown",
) -> int:
    return await callback_context.save_artifact(
        filename,
        types.Part.from_text(text=text.rstrip() + "\n"),
        custom_metadata={"mime_type": mime_type},
    )


async def load_text_artifact(
    callback_context: CallbackContext,
    filename: str,
) -> str | None:
    artifact = await callback_context.load_artifact(filename)
    if artifact is None or artifact.text is None:
        return None
    return artifact.text


async def list_artifact_keys(callback_context: CallbackContext) -> set[str]:
    keys = await callback_context.list_artifacts()
    return set(keys)


def record_last_error(
    callback_context: CallbackContext,
    step: str,
    message: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {"step": step, "message": message}
    if extra:
        payload.update(extra)
    callback_context.state["last_error"] = payload


async def prepare_step(
    callback_context: CallbackContext,
    *,
    step: str,
    required_artifacts: tuple[str, ...] = (),
    load_prior: dict[str, str] | None = None,
) -> types.Content | None:
    """各 step の before_agent_callback 共通実装。

    target_date の検証 → 前段 artifact の存在確認 →
    必要に応じて本文を temp state にロード。
    エラー時は適切な error/missing Content を返し、何も問題なければ None を返す。

    引数:
        step: 対象 step 名（例: "fetch_dosei"）。
        required_artifacts: 前提として必要な artifact 名（例: ("dosei.md",)）。
        load_prior: state_key -> artifact_name のマップ。本文を state にロードする。
            instruction で `{state_key}` を参照する場合に使う。
    """
    target_date = parse_target_date_input(callback_context)
    if target_date is None:
        return error_content(step, "missing or invalid target_date")
    try:
        validate_target_date(target_date)
    except ValueError as error:
        record_last_error(callback_context, step, str(error))
        return error_content(step, str(error))

    if required_artifacts:
        existing = await list_artifact_keys(callback_context)
        missing = [name for name in required_artifacts if name not in existing]
        if missing:
            record_last_error(
                callback_context,
                step,
                "required artifacts are missing",
                {"missing_artifacts": missing},
            )
            return missing_content(step, missing)

    callback_context.state["temp:target_date"] = target_date
    for state_key, artifact_name in (load_prior or {}).items():
        text = await load_text_artifact(callback_context, artifact_name)
        if text is None:
            message = f"{artifact_name} is unreadable"
            record_last_error(callback_context, step, message)
            return error_content(step, message)
        callback_context.state[state_key] = text
    return None


async def save_step_output(
    callback_context: CallbackContext,
    *,
    step: str,
    output_key: str,
    artifact_name: str,
) -> types.Content:
    """各 step の after_agent_callback 共通実装。

    LlmAgent が output_schema=StepOutput で書き込んだ構造化応答を読み、
    body が空でない場合のみ artifact として保存する。
    body が空 (= 失敗) なら error メッセージで error_content を返し、artifact は書かない。
    """
    output = callback_context.state.get(output_key) or {}
    body = (output.get("body") or "").strip()
    if not body:
        message = output.get("error") or "agent reported failure"
        record_last_error(callback_context, step, message)
        return error_content(step, message)
    version = await save_text_artifact(callback_context, artifact_name, body)
    callback_context.state["last_error"] = None
    return status_content(
        {
            "status": "success",
            "step": step,
            "artifact": artifact_name,
            "version": version,
            "bytes": len(body.encode("utf-8")),
        }
    )
