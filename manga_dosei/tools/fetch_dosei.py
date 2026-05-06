"""fetch_dosei: 対象日の jiji.com 首相動静を取得して dosei.md を保存する。"""

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import FunctionTool
from google.adk.tools.agent_tool import AgentTool

from manga_dosei import DEFAULT_TEXT_MODEL
from manga_dosei.tools._common import (
    StepInput,
    StepOutput,
    prepare_step,
    save_step_output,
)
from manga_dosei.tools._fetch_url import fetch_url
from manga_dosei.tools._tavily import build_tavily_toolset

_fetch_url_tool = FunctionTool(func=fetch_url)


_STEP = "fetch_dosei"
_ARTIFACT = "dosei.md"
_OUTPUT_KEY = "temp:fetch_dosei_output"


_DESCRIPTION = """\
対象日の jiji.com 首相動静を取得して dosei.md を artifact として保存するツール。

前提: なし（ワークフローの最初のステップ）。
引数: target_date は YYYYMMDD 形式の対象日。

完了時は処理結果の要約を構造化レスポンスとして返す。失敗時はエラー詳細を含む。
"""


def _build_prompt(target_date: str) -> str:
    year = target_date[:4]
    month = int(target_date[4:6])
    day = int(target_date[6:8])
    date_jp = f"{year}年{month}月{day}日"
    md_jp = f"{month}月{day}日"
    return f"""
{target_date} (YYYYMMDD形式 = {date_jp}、配信日ではなく対象日)
の首相動静を jiji.com から取得してください。

取得手順:
1. `tavily_search` で対象日の首相動静記事を検索する。クエリ例:
   - 「首相動静 {md_jp} jiji」
   - 「首相動静（{md_jp}）」
   - 「{date_jp} 首相動静 時事通信」
   ヒットしないときはクエリを変えて複数回試すこと。
2. 検索結果から jiji.com の記事 URL
   (例: `https://www.jiji.com/jc/article?...`) を 1 つ選び、
   `fetch_url` でその URL の本文を取得する。
   `fetch_url` の戻り値の `content` フィールドが本文 plain text。
3. 取得した本文の中から「首相動静」記事部分をそのまま転記する。
   記事末尾には「2026年MM月DD日HH時MM分配信」のような
   **配信日時表記がそのまま含まれている** ので、それも見落とさず転記する。
4. どうしても jiji.com 本サイトの記事本文に到達できなければ、body は空のまま、
   error に試したクエリと回数を書いて失敗報告すること。

【重要】絶対に守ること:
- 取得した記事の内容を一切創作・加工・要約・省略しないこと
- 記事の文章をそのまま正確に転記すること
- 存在しない情報を追加しないこと
- 検索結果の snippet だけから本文を再構成しないこと
  （必ず `fetch_url` で URL の本文を取得すること）
- 配信日時の数字を勝手に省略・補完しないこと
  （content に出てくる文字列を一字一句そのまま）

応答フォーマット (output_schema):
- 取得できた場合: body に記事本文（余分なヘッダーやフッター、説明文は含めない）
  + 末尾に **content から見つけた配信日時** とページ URL。error は空文字。
- 取得できなかった場合: body は空文字、error に理由を記載。

body の出力例:

首相動静（１月１日）

午前１０時現在、公邸。

同１０時１６分、公邸発。同２３分、皇居着。新年祝賀の儀に出席。同１１時８分、皇居発。同１５分、公邸着。

午後５時現在、公邸。

同１０時現在、公邸。

2026年01月01日22時10分配信
https://www.jiji.com/jc/article?k=2026010100169
""".strip()


def _build_instruction(context: ReadonlyContext) -> str:
    return _build_prompt(context.state.get("temp:target_date", ""))


async def _before(callback_context: CallbackContext):
    return await prepare_step(callback_context, step=_STEP)


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
    tools=[
        build_tavily_toolset(tool_filter=["tavily_search"]),
        _fetch_url_tool,
    ],
    before_agent_callback=_before,
    after_agent_callback=_after,
)


fetch_dosei_tool = AgentTool(agent=_agent)
