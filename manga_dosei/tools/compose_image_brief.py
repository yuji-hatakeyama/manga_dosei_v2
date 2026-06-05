"""compose_image_brief: scenario.md + layout.md + news.md + manifests/assets.json
を統合した画像生成専用ブリーフ image_brief.md を生成する。

後段の generate_page_gemini はこの 1 ファイルだけを読めば良く、
scenario.md / layout.md / news.md の中身が変わってもブリーフ層で吸収できる。
"""

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.agent_tool import AgentTool

from manga_dosei.config import get_settings
from manga_dosei.names import (
    TEMP_TARGET_DATE,
    ArtifactName,
    temp_key,
)
from manga_dosei.tools._common import (
    StepInput,
    StepOutput,
    prepare_step,
    save_step_output,
)

_STEP = "compose_image_brief"
_ARTIFACT = ArtifactName.IMAGE_BRIEF
_OUTPUT_KEY = temp_key(f"{_STEP}_output")
_REQUIRED = (
    ArtifactName.SCENARIO,
    ArtifactName.LAYOUT,
    ArtifactName.NEWS,
    ArtifactName.ASSETS_MANIFEST,
)
_SCENARIO_TEXT_KEY = temp_key("scenario_text")
_LAYOUT_TEXT_KEY = temp_key("layout_text")
_NEWS_TEXT_KEY = temp_key("news_text")
_ASSETS_MANIFEST_KEY = temp_key("assets_manifest")


_DESCRIPTION = """\
scenario.md / layout.md / news.md / manifests/assets.json を統合して、
画像生成専用のブリーフ image_brief.md を artifact として保存するツール。

前提: scenario.md, layout.md, news.md, manifests/assets.json が存在すること。
引数: target_date は YYYYMMDD 形式の対象日。

image_brief.md は generate_page_gemini / generate_page_gpt が
そのまま読み込む単一の入力で、ページタイトル・登場人物プロフィール・
ページレイアウト・コマ別仕様 (verbatim セリフ/視覚要素含む)・
描画前チェックリスト・免責文言までを含む。
"""


def _build_prompt(
    target_date: str,
    scenario_text: str,
    layout_text: str,
    news_text: str,
    assets_manifest: str,
) -> str:
    return f"""\
あなたは漫画ページ画像生成のためのブリーフライターです。
下記 4 つのソースを統合して、画像生成 AI が単独で読める
ブリーフ `image_brief.md` を作成してください。

対象日: {target_date}

# 入力 1: scenario.md (原典: タイトル/コマ別状況/イラスト/セリフ)

{scenario_text}

---

# 入力 2: layout.md (構造: ASCII 図/段配置/キャラ配置/チェックリスト)

{layout_text}

---

# 入力 3: news.md (人物プロフィール等の取材記録)

{news_text}

---

# 入力 4: manifests/assets.json (収集済み参照画像の一覧)

{assets_manifest}

---

# 出力フォーマット (厳守)

`body` フィールドに、下記構造の markdown 全体を入れてください。
コードブロックでラップせず、markdown 本文として直接記載。

```
# 画像生成ブリーフ

- 対象日: {target_date}
- pattern_id: <layout.md 冒頭の `- pattern_id: <id>` 行から取得した値をそのまま転記>
- 日付表記 (verbatim): YYYY年MM月DD日
- 主タイトル本文 (verbatim): <タイトル本文 (scenario の「### タイトル」から
  日付プレフィックスを除いた部分)>

## 登場人物プロフィール

(画像生成時の視覚的手がかり。テキストとしてページに描画しないこと。)

- **<氏名>** (<性別> / <年齢感> / <所属・役職>)
  - 参照画像: あり (<assets/<filename>>) ／ なし
  - 視覚特徴: <髪型・眼鏡の有無・服装等、news から拾えた特徴を 1 行で。
              無理に作らない、書ける情報だけ書く>

## ページレイアウト

(layout.md の ASCII 図と段配置をそのまま転記)

\\```
+--------+--------+
|   ②   |   ①   |
+--------+--------+
...
\\```

- 段 1: 右側=コマ①、左側=コマ②
- 段 2: ...

## コマ別仕様

### コマ ①

- 位置: <段 N の 右側 / 左側 / 全幅 (layout の段配置に合わせる)>
- コマタイトル (verbatim): <scenario の「### ① ...」の見出し本文 verbatim>
- キャラ配置: <layout の「キャラ配置」をそのまま (例: 画面左=高市早苗、画面右=片山さつき)>
- 視覚要素 (描画手がかり、文字として描画しない):
  - 状況: <scenario の「状況:」をそのまま転記>
  - 描写: <scenario の「イラスト:」をそのまま転記>
- 吹き出し (右→左 = 番号順、verbatim):
  1. <話者名>「<セリフ本文 verbatim>」
  2. ...

### コマ ②
...

## 描画前チェックリスト

(layout.md の末尾チェックリストをそのまま転記。2 人以上のコマのみ)

- コマ ①: 画面左 = ... ／ 画面右 = ...
- ...

## ページ下部 (赤字で必ず描画、verbatim)

※報道の首相動静を元にAIで創作したフィクションです。内容について一切の責任を負いません。
```

# 統合ルール

1. **scenario が原典**: コマタイトル / 状況 / イラスト / セリフは scenario の
   文字列を **一字一句そのまま転記** すること。圧縮・要約・改変禁止。
2. **layout が構造の正典**: ASCII 図・段配置・キャラ配置・描画前チェックリストは
   layout からそのまま転記。コマ番号 (①②...) で scenario と JOIN する。
   layout 冒頭の `- pattern_id: <id>` 行から ID を取得し、ブリーフ冒頭の
   `- pattern_id: <id>` にそのまま転記する (後段 generate_page_* がこの ID
   からレイアウト参考画像を選択するため、改変・省略禁止)。
3. **コマの「位置」**: layout の「## 段ごとの配置」から、各コマがどの段の
   どちら (右/左/全幅) かを読み取り、各コマ仕様の先頭に明記する。
4. **登場人物プロフィール**:
   - news の「## 主要人物プロフィール」を主要ソースとする。
   - scenario の「## 登場人物一覧」にあるが news に無い人物は、
     名前と役職のみ書き、視覚特徴は空欄でよい。
   - 「高市早苗」は news に出てこなくても常に **必ず含める** (女性 / 60代前半 /
     内閣総理大臣、参照画像: あり (assets/samples/sanae.jpg))。
   - 各人物の「参照画像: あり / なし」は manifests/assets.json の `assets[].name`
     と人物氏名を照合して決める。`name` がフルネーム一致 (姓 + 名) または
     苗字一致するなら「あり」+ `artifact` を記載、そうでなければ「なし」。
   - 視覚特徴は news の「人間味要素」「経歴の要点」から **見た目に関係しそうな
     断片** (メガネ・髪色・体格・年齢感 等) を抽出する。発言や政策的特徴は書かない。
5. **コマ別仕様の「視覚要素」**:
   - scenario の「状況:」と「イラスト:」を両方そのまま転記。
   - layout の「視覚要素」フィールドは scenario の圧縮版なので、
     **layout からはコピーしない**。
6. **吹き出し**:
   - layout の「吹き出し (右→左の読み順、台本のセリフ番号順)」をそのまま転記する。
   - 番号 (1., 2., ...) を保持し、話者名「セリフ本文」の形式も保持する。
7. **不要な情報は捨てる**: scenario の「## X 投稿用テキスト」、
   news の「## 漫画ネタ候補」「## 各面会・会議の詳細」「## 周辺政治情勢」
   「## 参照した関連ニュース」「首相動静本文」はブリーフに含めない。

# CRITICAL RULES

- `(verbatim)` 付きフィールド (日付表記 / 主タイトル本文 / コマタイトル / 吹き出し /
  ページ下部) は **画像内に文字として描画される対象** なので、ソースから
  **一字一句そのまま** 転記する。誤字・字形変更・改行追加・空白追加は禁止。
- それ以外のフィールド (登場人物プロフィール / 視覚要素 / キャラ配置 /
  描画前チェックリスト / 位置) は **画像生成 AI への手がかり** であり、
  描画対象ではない。ただし内容を改変してはいけない (転記元の文字列を保つ)。
- 創作禁止。入力に無い情報を追加しないこと (人物特徴の捏造、セリフの追加等)。
- コマ番号は scenario と layout で一致しているはず。一致しない場合は
  `error` に矛盾箇所を記載して `body` は空。

# 応答フォーマット (output_schema=StepOutput)

- 正常に統合できた場合: `body` に上記 markdown 全体、`error` は空文字。
- 入力に矛盾があり統合不可能な場合: `body` は空文字、`error` に理由。
""".strip()


def _build_instruction(context: ReadonlyContext) -> str:
    return _build_prompt(
        target_date=context.state.get(TEMP_TARGET_DATE, ""),
        scenario_text=context.state.get(_SCENARIO_TEXT_KEY, ""),
        layout_text=context.state.get(_LAYOUT_TEXT_KEY, ""),
        news_text=context.state.get(_NEWS_TEXT_KEY, ""),
        assets_manifest=context.state.get(_ASSETS_MANIFEST_KEY, ""),
    )


async def _before(callback_context: CallbackContext):
    return await prepare_step(
        callback_context,
        step=_STEP,
        required_artifacts=_REQUIRED,
        load_prior={
            _SCENARIO_TEXT_KEY: ArtifactName.SCENARIO,
            _LAYOUT_TEXT_KEY: ArtifactName.LAYOUT,
            _NEWS_TEXT_KEY: ArtifactName.NEWS,
            _ASSETS_MANIFEST_KEY: ArtifactName.ASSETS_MANIFEST,
        },
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
    model=get_settings().gemini_text_model,
    description=_DESCRIPTION,
    instruction=_build_instruction,
    input_schema=StepInput,
    output_schema=StepOutput,
    output_key=_OUTPUT_KEY,
    before_agent_callback=_before,
    after_agent_callback=_after,
)


compose_image_brief_tool = AgentTool(agent=_agent)
