#!/usr/bin/env bash
# PostToolUse hook: run `ruff format` + `ruff check --fix` on the Python file
# touched by Edit/Write. Input arrives on stdin as JSON; we extract
# `tool_input.file_path` and only act on `.py` files inside this repo.
#
# On environment problems (missing uv/jq) we surface a JSON warning so the
# user sees a `systemMessage` AND Claude sees an `additionalContext` nudge
# to fix the environment.

set -uo pipefail

emit_warning() {
  # msg は本スクリプト内の固定文字列のみ。" \ 改行 を含めない前提で
  # printf による素朴な JSON 生成を使う (jq に依存しない)。
  local msg="$1"
  local ctx="ruff-fix hook skipped: ${msg}. Ask the user to fix the environment so Python edits get auto-formatted."
  printf '{"systemMessage":"%s","hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":"%s"}}\n' "$msg" "$ctx"
  exit 0
}

INPUT=$(cat)

if ! command -v jq >/dev/null 2>&1; then
  emit_warning "jq not in PATH; install jq to enable the hook"
fi

FILE=$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path // empty')
[[ -z "${FILE:-}" || "$FILE" != *.py || ! -f "$FILE" ]] && exit 0

if ! command -v uv >/dev/null 2>&1; then
  emit_warning "uv not in PATH; install uv (https://docs.astral.sh/uv/)"
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-}"
[[ -z "$PROJECT_DIR" ]] && exit 0

# project 外のファイルはこの hook の管轄外 (Claude が他リポを編集している等)
[[ "$FILE" != "$PROJECT_DIR"/* ]] && exit 0

# project 内なのに pyproject.toml が無いのは project 構成の異常
if [[ ! -f "$PROJECT_DIR/pyproject.toml" ]]; then
  emit_warning "pyproject.toml not found at \$CLAUDE_PROJECT_DIR; ruff cannot run"
fi

# 事前チェック: ruff が uv 環境に入っていなければ env 異常として通知。
# 入っていない場合に format/check を走らせると "command not found" が
# stderr 経由で Claude に届き「lint feedback」と誤認される。
if ! (cd "$PROJECT_DIR" && uv run --quiet ruff --version >/dev/null 2>&1); then
  emit_warning "ruff not installed in the uv env; run \`uv sync\` to install"
fi

# ruff の出力は exit 2 + stderr 経由で Claude にフィードバックする。
# 成功時の "1 file reformatted" 等は capture して捨てる (stdout 漏れ防止)。

fmt_output=$(cd "$PROJECT_DIR" && uv run --quiet ruff format "$FILE" 2>&1)
fmt_rc=$?
if [[ $fmt_rc -ne 0 ]]; then
  printf '%s\n' "$fmt_output" >&2
  exit 2
fi

chk_output=$(cd "$PROJECT_DIR" && uv run --quiet ruff check --fix "$FILE" 2>&1)
chk_rc=$?
if [[ $chk_rc -ne 0 ]]; then
  printf '%s\n' "$chk_output" >&2
  exit 2
fi

exit 0
