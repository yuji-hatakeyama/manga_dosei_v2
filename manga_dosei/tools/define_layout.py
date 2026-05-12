"""define_layout: scenario.md から漫画ページの layout.md を生成する。

画像生成 (generate_page_gemini / generate_page_gpt) の前段で実行され、
シナリオを構造化したレイアウト指示書 (`layout.md`) を出力する。
画像生成側はこの layout.md をそのままプロンプトに埋め込んで使うことで、
シナリオ書式変動への耐性を確保する。
"""

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.agent_tool import AgentTool

from manga_dosei import DEFAULT_TEXT_MODEL
from manga_dosei.tools._common import (
    StepInput,
    StepOutput,
    prepare_step,
    save_step_output,
)

_STEP = "define_layout"
_ARTIFACT = "layout.md"
_OUTPUT_KEY = "temp:define_layout_output"
_REQUIRED = ("scenario.md",)


_DESCRIPTION = """\
scenario.md を読み、漫画ページの版下設計書 layout.md を artifact として
保存するツール。

前提: scenario.md が存在すること。
引数: target_date は YYYYMMDD 形式の対象日。

layout.md は後段の画像生成 (generate_page_gemini / generate_page_gpt) が
そのままプロンプトに貼り付けて使う設計指示書。読み順・コマ別の発言者配置・
吹き出し順・視覚要素を構造化 markdown で記述する。
"""


def _build_prompt(scenario_text: str, target_date: str) -> str:
    return f"""\
あなたは漫画版下デザイナーです。下記「対象の台本」から、
画像生成 AI に渡す版下設計書 (`layout.md`) を作成してください。

対象日: {target_date}

# 出力フォーマット (厳守)

`body` フィールドに、下記構造の markdown 全体を入れてください。
コードブロックでラップせず、markdown 本文として直接記載。

```
## ページレイアウト

(ASCII 図でページ全体の配置を視覚化。日本の漫画は右→左、上→下に読むので、
コマ①が右上、最後のコマが左下に来るよう配置すること)

```
+--------+--------+
|   ②   |   ①   |   ← 段 1 (右に①, 左に②)
+--------+--------+
|       ③       |   ← 段 2 (③ 全幅)
+--------+--------+
|   ⑤   |   ④   |   ← 段 3 (右に④, 左に⑤)
+--------+--------+
```

## 段ごとの配置

- 段 1: 右側=①、左側=②
- 段 2: ③ を全幅で配置
- 段 3: 右側=④、左側=⑤

## コマ別仕様

### コマ ① (例: 1 人のコマ)
- タイトル: (台本のコマタイトルを正確に転記)
- キャラ配置: A のみ（中央配置）
- 吹き出し（右→左の読み順、台本のセリフ番号順）:
  1. A「(セリフ本文を一字一句正確に転記)」
- 視覚要素: (場所・小道具・背景の手がかり。台本のイラスト欄から抽出)

### コマ ③ (例: 3 人のコマ)
- タイトル: ...
- キャラ配置: **画面左=A、中央=B、画面右=C**
- 吹き出し（右→左の読み順、台本のセリフ番号順）:
  1. C「...」
  2. B「...」
  3. A「...」
- 視覚要素: ...

### コマ ④ (例: 2 人のコマ)
- タイトル: ...
- キャラ配置: **画面左=A、画面右=B**
- 吹き出し（右→左の読み順、台本のセリフ番号順）:
  1. B「...」
  2. A「...」
- 視覚要素: ...

## 🔴 描画前チェックリスト

2 人以上のコマで、画面の **物理的な左右位置** を再確認するための最終ブロック。
画像生成 AI はここを最後に必ず読み、上の「キャラ配置」と矛盾なく描画すること。

- コマ N: 画面左 = **(発言順 #最後)** ／ (中央 = ... ／)
  画面右 = (発言順 #1)
```

# レイアウト作成ルール

1. **コマの読み順** (日本式漫画):
   - 右→左、上→下
   - コマ① は **必ずページ右上**、最後のコマは **必ずページ左下**
   - 段 (row) の中では右が先、左が後

2. **段の区切り方**:
   - 台本のコマ数が 4 なら 2 段（段1: ①② / 段2: ③④）
   - 5 なら 3 段（段1: ①② / 段2: ③ 全幅 / 段3: ④⑤）。
     全幅コマは台本の中で **情報量が多い・複数登場人物のいるコマ** を選ぶ
   - 6 なら 3 段（段1: ①② / 段2: ③④ / 段3: ⑤⑥）
   - 3 以下なら縦に並べる
   - 全幅にするコマは 1 つに絞る (複数の全幅コマは禁止)

3. **キャラ配置の表記（最重要）**:
   - 「キャラ配置」フィールドは **必ず下記いずれかの形式** で出力する
     （矢印 `→` や `、` の連結ではなく、**画面位置=名前** の明示形式）:
     - 1 人なら: `<話者名> のみ（中央配置）`
     - 2 人なら: `画面左=<話者名>、画面右=<話者名>`
     - 3 人なら: `画面左=<話者名>、中央=<話者名>、画面右=<話者名>`
     - 4 人以上なら: `画面左=<話者>、左中=<話者>、右中=<話者>、画面右=<話者>`
   - 並び順は **台本のセリフ番号順** で、画面右に #1、画面左に #N
     （日本式漫画の読み順「右→左」と一致させる）
   - 並びの根拠は人物属性ではなく **セリフ順** とする

4. **吹き出しの読み順**:
   - 「吹き出し（右→左の読み順）」は **台本のセリフ番号順** で記載
   - 番号は台本そのまま (1, 2, 3...)

5. **視覚要素**:
   - 台本の「イラスト:」欄から場所・小道具・背景手がかりを 1〜2 行に要約
   - 過剰に細かい指示は避ける（画像 AI の創造性に委ねる）

6. **🔴 描画前チェックリスト (末尾必須)**:
   - markdown 末尾に `## 🔴 描画前チェックリスト` セクションを置く。
   - 2 人以上のコマだけを以下の形式で列挙:
     `- コマ N: 画面左 = **<話者名>** ／ (中央 = <話者名> ／) 画面右 = <話者名>`
   - 1 人だけのコマは含めない。
   - 上の「キャラ配置」と完全に一致する内容で書く（再確認用ブロック）。

# CRITICAL RULES

- 台本のセリフは **一字一句正確に転記** すること。要約・改変禁止
- 台本のコマ番号 (①②③...) をそのまま使うこと
- 台本に書かれていない発言者・セリフ・コマを追加しないこと
- セリフ番号と配置の整合（右端=#1、左端=#最後）を必ず保つこと

# 応答フォーマット (output_schema=StepOutput)

- 正常に layout を作成できた場合:
  `body` に上記 markdown 全体、`error` は空文字
- 台本が読めない・コマが識別できない場合: `body` は空文字、`error` に理由

---

## 対象の台本

{scenario_text}
""".strip()


def _build_instruction(context: ReadonlyContext) -> str:
    scenario_text = context.state.get("temp:scenario_text", "")
    target_date = context.state.get("temp:target_date", "")
    return _build_prompt(scenario_text, target_date)


async def _before(callback_context: CallbackContext):
    return await prepare_step(
        callback_context,
        step=_STEP,
        required_artifacts=_REQUIRED,
        load_prior={"temp:scenario_text": "scenario.md"},
    )


async def _after(callback_context: CallbackContext):
    return await save_step_output(
        callback_context,
        step=_STEP,
        output_key=_OUTPUT_KEY,
        artifact_name=_ARTIFACT,
    )


_agent = LlmAgent(
    name=_STEP,
    model=DEFAULT_TEXT_MODEL,
    description=_DESCRIPTION,
    instruction=_build_instruction,
    input_schema=StepInput,
    output_schema=StepOutput,
    output_key=_OUTPUT_KEY,
    before_agent_callback=_before,
    after_agent_callback=_after,
)


define_layout_tool = AgentTool(agent=_agent)
