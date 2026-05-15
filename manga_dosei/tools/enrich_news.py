"""enrich_news: dosei.md に関連ニュースを付加して news.md を保存する。

deep research パターン:
SequentialAgent(researcher
    → LoopAgent(evaluator → escalation_checker → enhanced_searcher)
    → composer)
で、Tavily を使って多角的に深掘り調査してから最終 markdown を組み立てる。
adk-samples/python/agents/deep-search の構造を踏襲。
"""

from collections.abc import AsyncGenerator
from typing import Literal

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events import Event, EventActions
from google.adk.tools import FunctionTool
from google.adk.tools.agent_tool import AgentTool
from pydantic import BaseModel, Field

from manga_dosei import DEFAULT_TEXT_MODEL
from manga_dosei.tools._common import (
    StepInput,
    StepOutput,
    prepare_step,
    save_step_output,
)
from manga_dosei.tools._fetch_url import fetch_url
from manga_dosei.tools._tavily import (
    end_date_offset_from_target,
    make_tavily_search_tool,
)

_fetch_url_tool = FunctionTool(func=fetch_url)

# Tavily Search を 2 種類のドメイン固定 tool としてコード側で量産する。
# どちらも `end_date=対象日の翌日` で後日報道を index 段階で除外する。
# LLM は query (検索語) だけ決めればよく、topic / depth / domain / 日付範囲は
# プロンプトで揺らがない。

_search_news_jiji = make_tavily_search_tool(
    name="search_news_jiji",
    description=(
        "JIJI.COM (www.jiji.com) のニュース記事を Tavily で検索するツール。"
        "ニュース系の取材 ([SCOOP] / [MEETINGS] / [BACKGROUND] / follow-up) に使う。"
        "topic=news / max_results=20 / include_domains=['www.jiji.com'] / "
        "end_date=対象日の翌日 はコード側で固定。引数は query (検索語) のみ。"
    ),
    topic="news",
    max_results=20,
    include_domains=["www.jiji.com"],
    end_date=end_date_offset_from_target(days_after=1),
)

_search_news_yahoo = make_tavily_search_tool(
    name="search_news_yahoo",
    description=(
        "Yahoo!ニュース (news.yahoo.co.jp) を Tavily で検索するツール。"
        "人物背景・経歴・発言 ([PEOPLE]) の取材に使う。"
        "topic=news / max_results=20 / include_domains=['news.yahoo.co.jp'] / "
        "end_date=対象日の翌日 はコード側で固定。引数は query (検索語) のみ。"
    ),
    topic="news",
    max_results=20,
    include_domains=["news.yahoo.co.jp"],
    end_date=end_date_offset_from_target(days_after=1),
)


# --- summarize_url sub-agent (AgentTool) ---
# tavily_extract を直接 section_researcher に渡すと raw_content (URL 本文の全文) が
# 履歴に蓄積して 1M token を超えるため、要約 sub-agent でラップする。
# AgentTool は InMemorySessionService で child を動かすので、raw_content は
# child の history に閉じ込められ、親 (section_researcher) には要約のみが返る。


class SummarizeUrlInput(BaseModel):
    """summarize_url ツールの入力。"""

    url: str = Field(description="要約対象の絶対 URL (https://...)")
    focus: str = Field(
        description=(
            "抽出したい観点を 1 文で。例: 「高市総理の発言と本日の決定事項」"
            "「〇〇 (人物) の経歴・性別・年齢・所属」"
        )
    )


class SummarizeUrlOutput(BaseModel):
    """summarize_url ツールの出力 (コンパクトな要約のみ)。"""

    summary: str = Field(
        default="",
        description="focus 観点に絞った markdown 要約 (概ね 800 文字以内)。失敗時は空。",
    )
    source_title: str = Field(default="", description="取得できた記事タイトル")
    publish_date: str = Field(
        default="", description="配信日 (取得できれば 'YYYY年MM月DD日' 形式)"
    )
    error: str = Field(default="", description="失敗時の理由。成功時は空文字。")


def _summarize_url_instruction(context: ReadonlyContext) -> str:
    return """\
あなたは URL 要約エージェントです。引数 (`url`, `focus`) に従い、
URL 本文から focus 観点に絞ったコンパクトな markdown 要約を返します。

# 手順

1. `fetch_url` ツールで `url` を取得し、`content` フィールド (本文 plain text) を読む
2. 本文から `focus` に関連する情報のみを抽出
3. 概ね **800 文字以内** の markdown で要約 (見出し・箇条書き活用可)

# CRITICAL RULES

- `fetch_url` の戻り値が `status="error"` または `content` が空の場合は
  `error` に理由を書き、`summary` は空文字
- `focus` に関係ない情報は省略 (記事の冒頭定型文・関連リンク・広告等)
- 創作禁止。本文に書かれていない情報は追加しない
- `source_title` は本文冒頭やページタイトルから判断
- `publish_date` は本文中の配信日表記を判別して `YYYY年MM月DD日` 形式で返す。
  本文中の典型例:
  「2026年04月28日22時03分配信」「2026/04/28 21:38」「4/28(火) 7:00配信」など。
  判別不能なら空文字。

# 応答 (output_schema=SummarizeUrlOutput)

- 成功時: `summary` に要約、`source_title` / `publish_date` に判明した値、`error` は空
- 失敗時: `error` に理由、その他は空
"""


_summarizer = LlmAgent(
    name="summarize_url",
    model=DEFAULT_TEXT_MODEL,
    description=(
        "URL を `fetch_url` で取得し、`focus` で指定した観点に絞った"
        "コンパクトな markdown 要約 (概ね 800 文字以内) を返すツール。"
        "raw HTML 由来の boilerplate を除いた本文を内部で読み、"
        "親 agent には要約のみが返るので context-safe に深掘りできる。"
    ),
    instruction=_summarize_url_instruction,
    input_schema=SummarizeUrlInput,
    output_schema=SummarizeUrlOutput,
    output_key="temp:summarize_url_output",
    tools=[_fetch_url_tool],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    include_contents="none",
)

summarize_url_tool = AgentTool(agent=_summarizer)


def _build_research_tools() -> list:
    """section_researcher / enhanced_search_executor で使う共通ツールセット。

    - search_news_jiji: JIJI.COM のニュース検索 (snippet レベルで context-safe)
    - search_news_yahoo: Yahoo!ニュース検索 (人物背景用)
    - summarize_url_tool: 重要 URL を深掘りするときに要約のみ返す
    """
    return [
        _search_news_jiji,
        _search_news_yahoo,
        summarize_url_tool,
    ]


_STEP = "enrich_news"
_ARTIFACT = "news.md"
_OUTPUT_KEY = "temp:enrich_news_output"
_FINDINGS_KEY = "temp:enrich_section_findings"
_EVAL_KEY = "temp:enrich_research_evaluation"
_REQUIRED = ("dosei.md",)
_MAX_REFINEMENT_ITERATIONS = 2


_DESCRIPTION = """\
dosei.md に JIJI.COM (www.jiji.com) 等の関連ニュースと人物背景を付加して
news.md を artifact として保存するツール。

前提: dosei.md が存在すること。
引数: target_date は YYYYMMDD 形式の対象日。

内部では deep research パターン (researcher → 評価ループ → composer) で多角的に調査。
完了時は処理結果の要約を構造化レスポンスとして返す。失敗時はエラー詳細を含む。
"""


class SearchQuery(BaseModel):
    """個別の Tavily 検索クエリ。"""

    search_query: str = Field(description="Tavily で実行する具体的な検索クエリ")


class Feedback(BaseModel):
    """research_evaluator の出力スキーマ。

    grade=pass で EscalationChecker がループを抜ける。
    grade=fail のときは follow_up_queries を埋めて enhanced_searcher が全消化する。
    """

    grade: Literal["pass", "fail"] = Field(
        description=(
            "調査の十分性。pass なら追加調査不要、fail なら follow_up_queries を埋める。"
        )
    )
    comment: str = Field(description="評価の理由・不足箇所の具体的な指摘。")
    follow_up_queries: list[SearchQuery] | None = Field(
        default=None,
        description=(
            "grade=fail のときに不足を埋めるための追加検索クエリ。pass なら null。"
        ),
    )


class EscalationChecker(BaseAgent):
    """research_evaluation.grade=='pass' で escalate=True を立ててループを抜ける。

    LLM に exit_loop を任せると早期に抜けがちなので、Python で grade を見る。
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        evaluation = ctx.session.state.get(_EVAL_KEY)
        if evaluation and evaluation.get("grade") == "pass":
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            yield Event(author=self.name)


def _section_researcher_instruction(context: ReadonlyContext) -> str:
    target_date = context.state.get("temp:target_date", "")
    dosei_text = context.state.get("temp:dosei_text", "")
    return f"""\
あなたは「漫画台本作家のための取材リサーチャー」です。
下記「対象の首相動静」を題材に、
後段の漫画台本生成エージェントが 5 コマ漫画を書けるように、ニュース材料を集めてください。

# 利用可能な検索ツール

検索パラメータ (topic / search_depth / max_results / include_domains / 日付範囲) は
ツール側でコードによって固定されている。`query` (検索語) のみ渡す。

- `search_news_jiji`: JIJI.COM (www.jiji.com) のニュース検索。
  [SCOOP] / [MEETINGS] / [BACKGROUND] / follow-up に使う。
- `search_news_yahoo`: Yahoo!ニュース (news.yahoo.co.jp) の検索。
  [PEOPLE] (人物背景) に使う。
- `summarize_url`: 重要 URL の本文を focus 観点で要約取得。
  生本文ではなく要約しか返らないので何度呼んでも context-safe。

# あなたの役割: 素材を均等に提供する取材役

scenario writer は「学べる・興味を持つ・身近に感じる」を満たす漫画を作ります。
あなたは **その素材を提供する取材役** です。3 軸の素材を **均等に** 揃えてください:

- **学べる素材**: 当日の政治・政策の動き（誰が何を決めたか・何が議題か）
- **興味を持つ素材**: タイムライン上のインパクトある動き・スクープ的な動き
- **身近に感じる素材**: 政治家の発言・人間味エピソード

人物エピソードに偏らないこと。
コマ配分・演出・敬意フィルタ・ハッシュタグ選定は scenario writer の責務なので、
あなたは事実を出典付きで提供することに徹する。

---

対象日: {target_date}

対象の首相動静:

{dosei_text}

---

# 調査スコープ

## [SCOOP] 漫画ネタ候補（最重要・最初に着手）

下記 3 ジャンルから **それぞれ最低 1-2 件** ずつ拾う。
コマ割り・演出指示は scenario writer の仕事なので書かない:

### A. インパクト型 — 「えっ?」と思う動き
異様に短い面会・急な離席・驚きの動線・異例の組合せ・数字が立つ事象。
例: 「衆院本会議に出てから 11 分で退席して官邸戻ってビデオ収録」「160 人超の参拝」

### B. 政策決定型 — 当日のニュースの中身
当日決まったこと・発表されたこと・大きく動いた案件・対外発言の中心テーマ。
例: 「アフリカ4カ国歴訪、レアメタル確保が主題」
「外国人政策で日本語学習課程試行」

### C. 人物エピソード型 — 「人」として身近に感じる切り口
直接発言・人間味・趣味・対立・ほのぼの。
例: 「『僕が作った』とコロン香る市川局長」
「トゥンクトゥンクお気に入りの高市首相」

`search_news_jiji` で当たり、
ピンと来た URL は `summarize_url` で深掘り。
1-2 行の紹介で終わらせず、**直接 quote と背景** をセットで取れ。

## [PEOPLE] 主要人物のキャラクターシート

動静に登場する主要人物 (高市総理を除く、概ね 8-12 名) について、
漫画家が画と性格を起こせるレベルの素材を集める:
- 漢字氏名・フリガナ・性別・年齢・所属（党派・役職）
- 経歴の **要点** (細かい歴任ポストは不要、印象的なものだけ)
- **最近の直接発言** を 1-2 個 verbatim で
- **人間味要素** (口癖・趣味・持病・対立関係・特徴的エピソード) があれば

`search_news_yahoo` + `summarize_url` で関連記事を当たる。

## [MEETINGS] 各面会・会議の詳細

動静の面会について、議題・決定事項・主要な発言があれば事実ベースで簡潔に記録する。
読者がニュースを学ぶ手がかりになるので、形式的だからといって安易に切り捨てない。
本当に詳細不明な短時間ブリーフィングは無理に膨らませない。

## [BACKGROUND] 周辺政治情勢

対象日 ({target_date}) の前日・翌日も含む同時期の動向。
当日のニュースの伏線・背景になるもの:
- `search_news_jiji` を 5-8 回、多角的なキーワードで
- 重要そうな関連報道は `summarize_url` で深掘り

---

# Iterative Deepening — 必ず実行

このフェーズは「checklist 網羅」とは別軸で iterative に動く:

- `summarize_url` の戻り要約は毎回必ず読む。
  下記が見つかったら **追加で search/summarize_url を実行**:
  - **新しい人物名** (動静に無いが文脈で重要)
  - **新しい論点・固有名詞** (制度名・会議体・法案名・事件・企業名)
  - **新しい引用元 URL** → 重要なら再帰的深掘り
- 「もう十分」と早期判断しない。**気になる枝は全て辿ってから** 判断する。
- 同じクエリ繰り返し禁止、必ず新しい角度で。

---

# CRITICAL RULES

- 利用可能なツールは `search_news_jiji` / `search_news_yahoo` / `summarize_url` のみ
- `summarize_url` は **30-50 回まで** 使ってよい。深掘り優先
- **配信日が対象日 {target_date} 当日またはそれ以前** の記事のみ使用
- **配信日が確認できない記事、後日報道は絶対に使用しない**
- 創作禁止。`summarize_url` または信頼できる search snippet で確認した情報のみ記載
- 直接 quote は **改変せず原文ママ** で（「」で囲む）
- 各情報には URL を明記
- **外国企業・外国製品・外国サービス・外国 AI モデル等の固有名詞 (人名以外)**
  は、可能な範囲で **原語表記 (英語/アルファベット等)** も把握する。
  記事中の括弧書き・英文版記事 (jiji english 等) から拾える場合が多い。
  findings 本文への記載は **初出時に `カタカナ（English）` の形式** で併記する。
  例: `アンドリーセン・ホロウィッツ（Andreessen Horowitz）`、
  `クロード・ミュトス（Claude Mythos）`。
  原語が確認できない場合はカタカナのみで構わない (推測で原語を作らない)

---

# 最終出力フォーマット

最後の応答テキスト全体を以下の markdown として出力する:

## 漫画ネタ候補

漫画台本作家への素材提供。
**[A] インパクト型 / [B] 政策決定型 / [C] 人物エピソード型**
から **それぞれ最低 1-2 件**、合計 6-9 件。
各ネタは「キャッチコピー + 何が起きたか（直接 quote 含む）+ 出典」。
コマ割り・演出指示は書かない:

- **[A] 異例の 11 分退席**: 〜〜〜。 [出典: タイトル (配信日) URL]
- **[B] アフリカ4カ国歴訪、レアメタル確保が主題**: 〜〜〜。 [出典: ...]
- **[C] 〇〇さんの意外な発言**: 〜〜〜。 [出典: ...]
- ...

## 主要人物プロフィール

- **氏名 (フリガナ)** (性別・年齢・所属):
  - 経歴の要点
  - 最近の直接発言: 「〜〜」(出典: 配信日, URL)
  - 人間味要素: 〜〜 (取れた場合)
- ...

## 各面会・会議の詳細

動静の面会を時系列に並べる。
**漫画化に効きそうな面会は段落で narrative 重視**、機械的な面会は 1 行で OK。
テンプレートは強制しない。

### 〇時〇分 〇〇との面会
（議題が漫画化に使えるなら段落で詳しく書く。
背景・出席者・決定事項を物語的に。直接 quote があれば必ず収録。）
[出典: タイトル (配信日) URL]

### 〇時〇分 〇〇との面会
1 行サマリで十分。 [出典: URL]

## 周辺政治情勢

日付別の見出しには **対象日に基づく実際の日付（M月D日）を埋める** こと
（"対象日の前日" のような文字列リテラルは使わない）:

### 前日 (M月D日)
- ネタ 1: 〜〜 [出典: URL]

### 当日 (M月D日)
- ネタ 1: 〜〜 [出典: URL]

### 翌日 (M月D日)
- ネタ 1: 〜〜 [出典: URL]

「翌日の動向」は **対象日以前に配信された記事に書かれた予定情報**
（朝刊の予告記事、政府の事前発表等）から拾うこと。
対象日より後に配信された後追い記事は使用禁止。

## 参照ソース

利用した全ソースを 1 行 1 件で列挙。
`summarize_url` の戻り値の `source_title` / `publish_date` をそのまま使う。
タイトル不明・配信日不明のものは載せない:

- [src-1] タイトル | 配信日: YYYY年MM月DD日 | URL: https://...
- [src-2] タイトル | 配信日: YYYY年MM月DD日 | URL: https://...
...
"""


_section_researcher = LlmAgent(
    name="enrich_section_researcher",
    model=DEFAULT_TEXT_MODEL,
    description="Tavily で多角的に深掘り調査し research findings を生成。",
    instruction=_section_researcher_instruction,
    input_schema=StepInput,
    tools=_build_research_tools(),
    output_key=_FINDINGS_KEY,
)


def _research_evaluator_instruction(context: ReadonlyContext) -> str:
    target_date = context.state.get("temp:target_date", "")
    findings = context.state.get(_FINDINGS_KEY, "")
    return f"""\
あなたは漫画編集者です。下記 findings は「漫画台本作家に渡す取材資料」として、
3 軸（学べる・興味を持つ・身近に感じる）の素材バランスが揃っているか評価してください。
コマ配分や敬意フィルタは scenario writer の責務なので評価対象外。
素材として揃っているかだけを見ます。

対象日: {target_date}

findings:

{findings}

---

# 評価軸（学べる・興味を持つ・身近に感じるの 3 軸を満たすか）

以下を厳しくチェック:

1. **漫画ネタ候補のジャンルバランス** — `## 漫画ネタ候補` に
   [A] インパクト型 / [B] 政策決定型 / [C] 人物エピソード型 が
   **それぞれ最低 1 件ずつ** あるか。合計 6+ 件、直接 quote 1+ 個含まれているか。
   人物エピソードばかりが 5 件並ぶようなら fail。
2. **政策・決定事項のカバー率** — 当日の主要な政策動向（発表・決定・議題）が
   news.md から読み取れるか。
3. **主要人物プロフィールの verbatim 発言** — 各人物に直接の最近発言があるか。
   経歴の羅列だけだと身近に感じられない。
4. **時事的背景の網羅性** — `## 周辺政治情勢` で前日・当日・翌日
   それぞれ最低 1-2 件の動向があるか。

# grade の決め方

- **grade="pass"**: 漫画ネタ候補がジャンル A/B/C に分散しており、
  政策動向と人物発言と時事背景が揃っている
- **grade="fail"**: 上記が顕著に欠ける場合。`comment` で具体的に欠陥を指摘し、
  `follow_up_queries` には **欠けているジャンルを補う方向** のクエリを 5-7 個生成する。
  例: 政策決定型が薄ければ「外国人政策 4月28日 決定事項」、
  インパクト型が薄ければ「衆院本会議 高市 退席 ビデオ収録」など。

応答は Feedback schema に validate される **単一の raw JSON object** のみ。
前後に説明文を付けないこと。
"""


_research_evaluator = LlmAgent(
    name="enrich_research_evaluator",
    model=DEFAULT_TEXT_MODEL,
    description="findings を評価して follow-up クエリを生成。",
    instruction=_research_evaluator_instruction,
    output_schema=Feedback,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key=_EVAL_KEY,
    # 直前の section_researcher の tool 呼び出し履歴は不要
    # (findings は state 経由で受け取る)
    include_contents="none",
)


def _enhanced_search_instruction(context: ReadonlyContext) -> str:
    target_date = context.state.get("temp:target_date", "")
    findings = context.state.get(_FINDINGS_KEY, "")
    evaluation = context.state.get(_EVAL_KEY, {}) or {}
    comment = evaluation.get("comment", "")
    follow_up_queries = evaluation.get("follow_up_queries") or []
    queries_text = (
        "\n".join(f"- {q.get('search_query', '')}" for q in follow_up_queries if q)
        or "(なし)"
    )
    return f"""\
あなたは漫画台本のための取材リサーチャーで、
前回 findings は漫画編集者に **'fail'** と評価されました。
「漫画化に効くネタ・直接発言・人間味」を補強してください。

# 利用可能な検索ツール

検索パラメータはツール側でコード固定 (`query` だけ渡す):

- `search_news_jiji`: JIJI.COM (www.jiji.com) のニュース検索。
  ニュース系 ([SCOOP] / [MEETINGS] / [BACKGROUND] / follow-up) はこちら。
- `search_news_yahoo`: Yahoo!ニュース (news.yahoo.co.jp) の検索。
  人物背景 ([PEOPLE]) はこちら。
- `summarize_url`: 重要 URL の本文を focus 観点で要約取得。

対象日: {target_date}

編集者コメント: {comment}

実行すべき follow-up queries (**全て実行必須**):
{queries_text}

---

# 手順

1. 上記 follow_up_queries の **EVERY query** を、
   クエリの内容に応じて `search_news_jiji` または `search_news_yahoo` で実行
   （一つも省略しない）。
   人物背景系のクエリは `search_news_yahoo`、それ以外のニュース系は
   `search_news_jiji` を選ぶ。
2. 候補 URL のうちネタになりそうなものを `summarize_url` で要約取得
   （focus には「直接発言」「人間味エピソード」「対立・批判発言」など
   漫画化に効く観点を書く）
3. 既存 findings と統合し、**強化された完全な findings 全体** を出力

既存 findings:

{findings}

---

# CRITICAL RULES

- snippet だけで判断せず、重要 URL は必ず `summarize_url` で要約取得
  （コンパクトな要約しか返さないので何度呼んでも安全）
- 配信日が対象日 {target_date} 当日またはそれ以前の記事のみ使用
- 既存 findings の有用な情報は残し、新規情報を統合（差し替えではなく増強）
- 出典 URL を必ず明記
- `summarize_url` は 10-15 回程度まで使ってよい
- **形式的な短時間面会を無理に膨らませない**。
  空白でも漫画化に支障なければそのままで OK

---

# 最終出力

Phase 1 と同じ markdown 構造
（漫画ネタ候補 / 主要人物プロフィール / 各面会・会議の詳細 / 周辺政治情勢 / 参照ソース）
で、欠落項目を埋めた **完全版 findings** 全体を応答テキストとして出力。
"""


_enhanced_search_executor = LlmAgent(
    name="enrich_enhanced_search_executor",
    model=DEFAULT_TEXT_MODEL,
    description="follow-up queries を全消化して findings を更新。",
    instruction=_enhanced_search_instruction,
    tools=_build_research_tools(),
    output_key=_FINDINGS_KEY,
    # 過去の section_researcher / 前 iteration の履歴は不要
    # (findings/eval を state から取る)
    include_contents="none",
)


def _news_composer_instruction(context: ReadonlyContext) -> str:
    target_date = context.state.get("temp:target_date", "")
    dosei_text = context.state.get("temp:dosei_text", "")
    findings = context.state.get(_FINDINGS_KEY, "")
    return f"""\
あなたは漫画台本作家への資料を組み立てる編集者です。
下記「首相動静本文」と「research findings」から、最終 news.md を作ります。

# ゴール再確認

これは漫画台本生成エージェントが読む資料です。
**漫画ネタとキャラ造形** が最優先で、官報のような網羅性は二の次。
findings の良い部分を **漫画作家にとって読みやすく** 並べ直す。

対象日: {target_date}

首相動静本文 (**変更禁止**、そのまま冒頭に置く):

{dosei_text}

research findings:

{findings}

---

# 出力構造

body フィールドに、以下の構造の markdown 全体を入れる:

```
（首相動静本文をそのまま冒頭に転記）

---

## 漫画ネタ候補

findings の `## 漫画ネタ候補` セクションをそのまま流用。
漫画作家が真っ先に読むので **冒頭に置く**。
各ネタは「[A/B/C ジャンル] キャッチコピー + 何が起きたか（直接 quote 含む） + 出典」
のシンプルな構成。
**コマ案・演出指示は書かない**（後段の scenario writer の仕事）。

## 主要人物プロフィール

- **氏名 (フリガナ)** (性別・年齢・所属):
  - 経歴の要点
  - 最近の直接発言: 「〜〜〜」(出典: 配信日, URL)
  - 人間味要素: 〜〜 (取れたものだけ書く、無理に作らない)

（複数人物を列挙、findings の人物カードを整形）

## 各面会・会議の詳細

動静の面会を時系列に並べる。
議題・決定事項・主要な発言があれば事実ベースで簡潔に記録する。
読者がニュースを学ぶ手がかりになるので、形式的だからといって安易に切り捨てない。
テンプレートを強制せず、findings に書かれた粒度で書く:

### 〇時〇分〜〇時〇分 〇〇との面会
議題・背景・決定事項・直接 quote を簡潔に。直接 quote はそのまま「」付きで収録。
[出典: タイトル (配信日) URL]

## 周辺政治情勢

実際の日付（M月D日）を見出しに埋める。"対象日の前日" のような文字列リテラルは使わない:

### 前日 (M月D日)
- ...

### 当日 (M月D日)
- ...

### 翌日 (M月D日)
- ... （**対象日以前に配信された記事に書かれた翌日の予定** を載せる）

## 参照した関連ニュース

- タイトル: ...
  配信日: YYYY年MM月DD日
  URL: https://...
```

# 「## 参照した関連ニュース」の埋め方

findings 末尾の **「## 参照ソース」セクションをそのまま流用** すること。

- findings の `[src-N] タイトル | 配信日: YYYY年MM月DD日 | URL: https://...`
  を下記フォーマットに変換:
  ```
  - タイトル: <タイトル>
    配信日: <YYYY年MM月DD日>
    URL: <URL>
  ```
- タイトルや配信日が「不明」「出典情報」のような汎用文字列・空のものは載せない
  (URL のみは省く)
- 本文中の inline `[出典: タイトル (配信日) URL]` で参照ソース未記載のものは追加してよい
- 重複 URL は 1 件にまとめる

---

# CRITICAL RULES

- 首相動静本文は冒頭に **そのまま転記** (修正・要約禁止)
- 配信日が対象日 {target_date} より後のものは含めない
- 出典 URL を明記できない情報は記載しない
- 創作禁止、findings に無い情報は書かない
- **直接 quote は「」付きで原文ママ**、改変禁止
- findings に `カタカナ（English）` のように **原語併記** された外国企業・
  外国製品・外国サービス・外国 AI モデル等の固有名詞は、本文 (漫画ネタ候補 /
  主要人物プロフィール / 各面会・会議の詳細) の **初出時に併記を維持** する。
  後段の漫画台本作家がカタカナ/原語のどちらで描画するか判断できるようにするため

---

# 応答フォーマット (output_schema=StepOutput)

- findings が充実しており news.md を組み立てられた場合:
  `body` に上記 markdown 全体、`error` は空文字
- findings が空または不十分で組み立て不可の場合: `body` は空文字、`error` に理由
"""


_news_composer = LlmAgent(
    name="enrich_news_composer",
    model=DEFAULT_TEXT_MODEL,
    description="findings + 首相動静から最終 news.md を組み立てる。",
    instruction=_news_composer_instruction,
    output_schema=StepOutput,
    output_key=_OUTPUT_KEY,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    # 直前 sub_agents の tool 履歴は不要 (findings/dosei は state 経由)
    include_contents="none",
)


async def _before(callback_context: CallbackContext):
    return await prepare_step(
        callback_context,
        step=_STEP,
        required_artifacts=_REQUIRED,
        load_prior={"temp:dosei_text": "dosei.md"},
    )


async def _after(callback_context: CallbackContext):
    return await save_step_output(
        callback_context,
        step=_STEP,
        output_key=_OUTPUT_KEY,
        artifact_name=_ARTIFACT,
    )


_agent = SequentialAgent(
    name=_STEP,
    description=_DESCRIPTION,
    sub_agents=[
        _section_researcher,
        LoopAgent(
            name="enrich_iterative_refinement_loop",
            max_iterations=_MAX_REFINEMENT_ITERATIONS,
            sub_agents=[
                _research_evaluator,
                EscalationChecker(name="enrich_escalation_checker"),
                _enhanced_search_executor,
            ],
        ),
        _news_composer,
    ],
    before_agent_callback=_before,
    after_agent_callback=_after,
)


enrich_news_tool = AgentTool(agent=_agent)
