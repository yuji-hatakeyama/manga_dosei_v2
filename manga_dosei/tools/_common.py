"""Shared helpers for ADK workflow tools."""

import json
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.genai import types
from pydantic import BaseModel, ValidationError

from manga_dosei.names import TEMP_TARGET_DATE, StateKey
from manga_dosei.tools._state import LastError, error_result
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


def parse_step_output(output: Any) -> tuple[str | None, str | None]:
    """LlmAgent の StepOutput 出力を (body, error) に正規化する純粋関数。

    `save_step_output` の callback から呼ばれる業務ロジックの pure core。
    CallbackContext を触らないので単体テスト可能。

    挙動:
      - dict / StepOutput / str を受け付ける (LLM 応答経路で実際に観測される形)。
      - error が非空なら ambiguity を潰すため body より優先する (失敗を見逃さない)。
      - body が空白のみなら未設定扱い (rstrip 後の空文字を error 化はしない)。
      - 入力が None や未対応型なら (None, "<理由>") を返す。
    """
    if output is None:
        return None, "output_key present but value is None"
    if isinstance(output, BaseModel):
        output = output.model_dump()
    if isinstance(output, str):
        body = output.strip()
        return (body or None), None
    if not isinstance(output, dict):
        return None, f"unexpected output type: {type(output).__name__}"
    error = (output.get("error") or "").strip()
    if error:
        return None, error
    body = (output.get("body") or "").strip()
    if not body:
        return None, "agent reported failure"
    return body, None


def decide_missing(
    required: tuple[str, ...] | list[str],
    available: set[str] | frozenset[str] | list[str] | tuple[str, ...],
) -> list[str]:
    """`required` のうち `available` に無いものを順序保ったまま返す純粋関数。

    `prepare_step` の required_artifacts チェックの pure core。
    """
    available_set = set(available)
    return [name for name in required if name not in available_set]


def parse_target_date_input(
    callback_context: CallbackContext,
) -> tuple[str | None, str | None]:
    """AgentTool が user_content の text に JSON で詰めた引数から target_date を取り出す。

    input_schema=StepInput により形式は保証されているので、JSON parse と
    Pydantic validation のみ行い、失敗時は (None, reason) を返す。
    reason は呼び出し元が last_error に積むための discriminator
    (`no_user_content` / `empty_text` / `validation_error: <detail>`)。
    """
    user_content = callback_context.user_content
    if not user_content or not user_content.parts:
        return None, "no_user_content"
    text = (user_content.parts[0].text or "").strip()
    if not text:
        return None, "empty_text"
    try:
        return StepInput.model_validate_json(text).target_date, None
    except ValidationError as error:
        # NOTE: pydantic detail を含めないと LLM が自己修正できない
        return None, f"validation_error: {error}"


def status_content(payload: dict[str, Any]) -> types.Content:
    """Status dict を JSON 文字列として持つ model Content を作る。

    AgentTool は内部 agent の最終応答テキストを呼び出し元に返す。
    呼び出し側 LLM はこの JSON 文字列を読み取って構造化情報として扱える。
    """
    return types.Content(
        role="model",
        parts=[types.Part(text=json.dumps(payload, ensure_ascii=False))],
    )


def error_content(step: str, message: str, **extras: Any) -> types.Content:
    payload: dict[str, Any] = {"status": "error", "step": step, "message": message}
    payload.update(extras)
    return status_content(payload)


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
    callback_context: Any,
    step: str,
    message: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: LastError = {"step": step, "message": message, **(extra or {})}  # type: ignore[typeddict-item]
    callback_context.state[StateKey.LAST_ERROR] = payload


def clear_last_error(callback_context: Any) -> None:
    callback_context.state[StateKey.LAST_ERROR] = None


def build_error_result(
    tool_context: Any,
    step: str,
    message: str,
    **extras: Any,
) -> dict[str, Any]:
    """Direct-tool error helper: record `last_error` and return an error dict.

    Used by `FunctionTool`-style tools (no prepare_step/save_step_output
    pipeline) that need to both write `state["last_error"]` for the CLI
    retry/abort path and return a payload to the caller. `extras` ride the
    returned dict (e.g. `page_number=`) and any key that is also declared
    on `LastError` rides `state["last_error"]` too.
    """
    payload = error_result(step, message)
    payload.update(extras)
    record_last_error(tool_context, step, message, extras or None)
    return payload


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
    target_date, reason = parse_target_date_input(callback_context)
    if target_date is None:
        # NOTE: AgentTool input plumbing problem (wiring) — not a step-level
        # failure. Do NOT touch last_error so an upstream step's failure
        # context survives the CLI retry/abort decision.
        message = f"missing or invalid target_date ({reason})"
        return error_content(step, message, reason=reason)
    try:
        validate_target_date(target_date)
    except ValueError as error:
        # Same rationale as above: malformed target_date is wiring, not a step
        # failure.
        return error_content(step, str(error))

    if required_artifacts:
        existing = await list_artifact_keys(callback_context)
        missing = decide_missing(required_artifacts, existing)
        if missing:
            record_last_error(
                callback_context,
                step,
                "required artifacts are missing",
                {"missing_artifacts": missing},
            )
            return error_content(
                step,
                "required artifacts are missing",
                missing_artifacts=missing,
            )

    callback_context.state[TEMP_TARGET_DATE] = target_date
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
    # NOTE: state に output_key 自体が無い場合は LlmAgent の配線不良
    # (output_schema 不一致 / callback 順序バグ等) で、LLM 応答の失敗とは別物。
    # 同じ "agent reported failure" に潰すと根本原因が見えなくなるので分岐する。
    if output_key not in callback_context.state:
        message = "output_key missing from state (no state entry written by agent)"
        record_last_error(callback_context, step, message, {"output_key": output_key})
        # F2: keep Content extras aligned with last_error so the AgentTool-facing
        # payload carries the same output_key discriminator.
        return error_content(step, message, output_key=output_key)
    body, error = parse_step_output(callback_context.state.get(output_key))
    if body is None:
        message = error or "agent reported failure"
        record_last_error(callback_context, step, message)
        return error_content(step, message)
    version = await save_text_artifact(callback_context, artifact_name, body)
    clear_last_error(callback_context)
    return status_content(
        {
            "status": "success",
            "step": step,
            "artifact": artifact_name,
            "version": version,
            "bytes": len(body.encode("utf-8")),
        }
    )
