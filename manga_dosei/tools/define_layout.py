"""define_layout: scenario.md からレイアウトパターンを選び layout.md を生成する。

レイアウトカタログ (`assets/layouts/<id>/`) には id ごとに ASCII 図・段ごとの
配置・参考画像 (sample.jpg) が用意されている。本ツールはコマ数と台本の展開を
見て、カタログから 1 つの pattern_id を選び、その正準 ASCII + 段配置を
layout.md に転記したうえで、コマ別のキャラ配置と描画前チェックリストを
組み立てる。

layout.md にはコマタイトル・セリフ・視覚要素は書かない (compose_image_brief が
scenario.md から直接転記する責務)。
"""

import json
from functools import lru_cache

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.agent_tool import AgentTool

from manga_dosei import paths
from manga_dosei.config import get_settings
from manga_dosei.names import (
    TEMP_TARGET_DATE,
    ArtifactName,
    temp_key,
)
from manga_dosei.tools._common import (
    StepInput,
    StepOutput,
    error_content,
    prepare_step,
    record_last_error,
    save_step_output,
)

_STEP = "define_layout"
_ARTIFACT = ArtifactName.LAYOUT
_OUTPUT_KEY = temp_key(f"{_STEP}_output")
_REQUIRED = (ArtifactName.SCENARIO,)
_CATALOG_STATE_KEY = temp_key("layout_catalog_block")
_SCENARIO_TEXT_KEY = temp_key("scenario_text")


def _load_catalog() -> dict[str, dict]:
    """assets/layouts/<id>/{meta.json, ascii.txt} を読み込んでカタログ辞書を返す。"""
    # NOTE: attribute access on `paths` so tests monkeypatching `paths.LAYOUTS_DIR`
    # win — a `from paths import LAYOUTS_DIR` snapshot would freeze the real path.
    layouts_dir = paths.LAYOUTS_DIR
    catalog: dict[str, dict] = {}
    if not layouts_dir.is_dir():
        raise FileNotFoundError(f"layout catalog directory not found: {layouts_dir}")
    for entry in sorted(layouts_dir.iterdir()):
        if not entry.is_dir():
            continue
        meta_path = entry / "meta.json"
        ascii_path = entry / "ascii.txt"
        # NOTE: 不完全なパターンディレクトリは silent skip しない。
        # 「ファイル名 typo でカタログに載らず、image-gen で pattern_id 不明
        # エラーになる」という debug 困難な状況を防ぐため、明示的に失敗させる。
        if not meta_path.exists() or not ascii_path.exists():
            raise FileNotFoundError(
                f"layout pattern {entry.name} missing meta.json or ascii.txt"
            )
        # NOTE: 裸の KeyError("'rows'") / JSONDecodeError 等では「どのパターンが
        # 原因か」が last_error から読み取れない。entry.name を前置して再送する。
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            ascii_text = ascii_path.read_text(encoding="utf-8").rstrip("\n")
            catalog[meta["id"]] = {
                "id": meta["id"],
                "panels": meta["panels"],
                "name": meta["name"],
                "when_to_use": meta["when_to_use"],
                "rows": meta["rows"],
                "ascii": ascii_text,
            }
        except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"layout pattern {entry.name}: {exc}") from exc
    if not catalog:
        raise FileNotFoundError(f"no valid layout patterns found under {layouts_dir}")
    return catalog


# NOTE: lazy + memoised so import 時に filesystem を叩かず、初回呼び出しでのみ disk へ。
# カタログ破損が import error として上がり無関係な adk web まで巻き添えで落ちる
# のを防ぐため (AGENTS.md の last_error / retry 契約)。
@lru_cache(maxsize=1)
def get_layout_catalog() -> dict[str, dict]:
    return _load_catalog()


def _format_catalog_for_prompt() -> str:
    blocks: list[str] = []
    for pattern_id, entry in get_layout_catalog().items():
        rows_text = "\n".join(f"- {row}" for row in entry["rows"])
        blocks.append(
            f"### {pattern_id} — {entry['name']} (panels={entry['panels']})\n"
            f"向き: {entry['when_to_use']}\n\n"
            f"ASCII:\n```\n{entry['ascii']}\n```\n\n"
            f"段ごとの配置:\n{rows_text}"
        )
    return "\n\n".join(blocks)


_DESCRIPTION = """\
scenario.md を読み、レイアウトカタログから pattern_id を 1 つ選び、その
正準 ASCII + 段配置 + 台本に応じたキャラ配置を含む layout.md を artifact
として保存するツール。

前提: scenario.md が存在すること。assets/layouts/ にパターンカタログが
存在すること。
引数: target_date は YYYYMMDD 形式の対象日。

layout.md は **版下構造のみ** を含む (コマタイトル・セリフ・視覚要素は
書かない)。先頭に `pattern_id: <id>` を必ず記載し、後段の compose_image_brief
は同 ID を image_brief.md に伝搬する。
"""


def _build_prompt(scenario_text: str, target_date: str, catalog_block: str) -> str:
    return f"""\
あなたは漫画版下デザイナーです。下記「対象の台本」のコマ数と展開を見て、
**レイアウトカタログから pattern_id を 1 つ選び**、その正準 ASCII と段配置を
転記したうえで、コマ別のキャラ配置と描画前チェックリストを書いてください。

コマタイトル・セリフ・視覚要素は layout.md に書きません (後段の
compose_image_brief が scenario.md から直接拾います)。

対象日: {target_date}

# レイアウトパターン カタログ

下記から、台本のコマ数 (panels) と展開に最適な **pattern_id を 1 つ** 選択。
台本のコマ数と一致する panels のパターンしか選んではいけない。

{catalog_block}

# 出力フォーマット (厳守)

`body` フィールドに、下記構造の markdown 全体を入れてください。
コードブロックでラップせず、markdown 本文として直接記載。

```
- pattern_id: <選んだ ID>

## ページレイアウト

<選んだパターンの ASCII を一字一句そのまま転記 (```text のコードブロックで囲む)>

## 段ごとの配置

<選んだパターンの「段ごとの配置」を一字一句そのまま転記>

## コマ別配置

### コマ ① (例: 1 人のコマ)
- 位置: <段 N 右側 / 段 N 左側 / 段 N 全幅 のいずれか、選んだパターンの段配置と一致させる>
- キャラ配置: A のみ（中央配置）

### コマ ③ (例: 3 人のコマ、全幅)
- 位置: 段 2 全幅
- キャラ配置: **画面左=A、中央=B、画面右=C**

### コマ ④ (例: 2 人のコマ)
- 位置: 段 3 右側
- キャラ配置: **画面左=A、画面右=B**

## 🔴 描画前チェックリスト

2 人以上のコマで、画面の **物理的な左右位置** を再確認するための最終ブロック。
画像生成 AI はここを最後に必ず読み、上の「キャラ配置」と矛盾なく描画すること。

- コマ N: 画面左 = **(発言順 #最後)** ／ (中央 = ... ／)
  画面右 = (発言順 #1)
```

# 版下構造のルール

1. **pattern_id 選択**:
   - 台本のコマ数と一致する panels のパターンのみ選択可能
   - 複数候補がある場合は `when_to_use` の説明と台本の展開を見て最適な 1 つを選ぶ
   - 選んだパターンの ASCII と「段ごとの配置」は **一字一句そのまま転記** すること
     (カタログが正典)。創作・編集禁止

2. **コマの読み順** (日本式漫画):
   - 右→左、上→下
   - コマ① は **必ずページ右上**、最後のコマは **必ずページ左下**
   - 段 (row) の中では右が先、左が後
   - カタログの ASCII はこの読み順に従って設計済み

3. **コマ別配置の「位置」フィールド**:
   - 各コマ仕様の先頭に必ず置く
   - 選んだパターンの「段ごとの配置」と完全に一致する文字列を使う
     (例: `段 1 右側` / `段 2 全幅` 等)

4. **キャラ配置の表記（最重要）**:
   - 「キャラ配置」フィールドは **必ず下記いずれかの形式** で出力する
     （矢印 `→` や `、` の連結ではなく、**画面位置=名前** の明示形式）:
     - 1 人なら: `<話者名> のみ（中央配置）`
     - 2 人なら: `画面左=<話者名>、画面右=<話者名>`
     - 3 人なら: `画面左=<話者名>、中央=<話者名>、画面右=<話者名>`
     - 4 人以上なら: `画面左=<話者>、左中=<話者>、右中=<話者>、画面右=<話者>`
   - 並び順は **台本のセリフ番号順** で、画面右に #1、画面左に #N
     （日本式漫画の読み順「右→左」と一致させる）
   - 並びの根拠は人物属性ではなく **セリフ順** とする

5. **🔴 描画前チェックリスト (末尾必須)**:
   - markdown 末尾に `## 🔴 描画前チェックリスト` セクションを置く
   - 2 人以上のコマだけを以下の形式で列挙:
     `- コマ N: 画面左 = **<話者名>** ／ (中央 = <話者名> ／) 画面右 = <話者名>`
   - 1 人だけのコマは含めない
   - 上の「キャラ配置」と完全に一致する内容で書く（再確認用ブロック）

# CRITICAL RULES

- 台本のコマ番号 (①②③...) をそのまま使うこと
- 台本に書かれていない発言者・コマを追加しないこと
- コマタイトル / セリフ / 視覚要素 は **書かない** (compose_image_brief が
  scenario.md から直接転記する責務)
- 画面位置 (左右/中央) の根拠は台本のセリフ番号順
- pattern_id は冒頭に `- pattern_id: <id>` の形で必ず記載する

# 応答フォーマット (output_schema=StepOutput)

- 正常に layout を作成できた場合:
  `body` に上記 markdown 全体、`error` は空文字
- 台本のコマ数に一致する pattern が無い・台本が読めない場合:
  `body` は空文字、`error` に理由

---

## 対象の台本

{scenario_text}
""".strip()


def _build_instruction(context: ReadonlyContext) -> str:
    scenario_text = context.state.get(_SCENARIO_TEXT_KEY, "")
    target_date = context.state.get(TEMP_TARGET_DATE, "")
    catalog_block = context.state.get(_CATALOG_STATE_KEY, "")
    return _build_prompt(scenario_text, target_date, catalog_block)


async def _before(callback_context: CallbackContext):
    prepared = await prepare_step(
        callback_context,
        step=_STEP,
        required_artifacts=_REQUIRED,
        load_prior={_SCENARIO_TEXT_KEY: ArtifactName.SCENARIO},
    )
    if prepared is not None:
        return prepared
    # NOTE: カタログ読み込みを import 時ではなくここで初回評価することで、
    # 破損時に import error ではなく step の last_error / retry 契約に乗せる。
    try:
        catalog_block = _format_catalog_for_prompt()
    except (FileNotFoundError, ValueError) as exc:
        message = f"layout catalog load failed: {exc}"
        record_last_error(callback_context, _STEP, message)
        return error_content(_STEP, message)
    callback_context.state[_CATALOG_STATE_KEY] = catalog_block
    return None


async def _after(callback_context: CallbackContext):
    return await save_step_output(
        callback_context,
        step=_STEP,
        output_key=_OUTPUT_KEY,
        artifact_name=_ARTIFACT,
    )


_agent = LlmAgent(
    name=_STEP,
    model=get_settings().gemini_text_model,
    description=_DESCRIPTION,
    instruction=_build_instruction,
    input_schema=StepInput,
    output_schema=StepOutput,
    output_key=_OUTPUT_KEY,
    before_agent_callback=_before,
    after_agent_callback=_after,
)


define_layout_tool = AgentTool(agent=_agent)
