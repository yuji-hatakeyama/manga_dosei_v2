import os

from google.adk.agents import Agent

from manga_dosei import DEFAULT_TEXT_MODEL
from manga_dosei.tools import (
    collect_assets_tool,
    enrich_news_tool,
    fetch_dosei_tool,
    generate_page_gemini,
    generate_scenario_tool,
    inspect_artifacts,
    resize_assets,
)


root_agent = Agent(
    name="manga_dosei",
    model=os.getenv("GEMINI_TEXT_MODEL", DEFAULT_TEXT_MODEL),
    description="Runs and inspects the manga dosei daily content workflow.",
    instruction="""
あなたは manga dosei ワークフローのエージェントです。

ツールを呼び出して日次のコンテンツ生成ワークフローを進めます。対応するツールが
status=success を返していない限り、そのステップが完了したとみなしてはいけません。

各ツールの詳細仕様（前提・挙動・制約・返り値）は、ツール定義の docstring を
参照してください。本 instruction では進行ルールのみを定義します。

## ワークフローの正典の順序

1. fetch_dosei
2. enrich_news
3. generate_scenario
4. collect_assets
5. resize_assets
6. generate_page_gemini (page_number 1〜5 を順に呼ぶ)

generate_page_gemini は target_date (YYYYMMDD) と page_number (1〜5) の 2 引数を取り、
1 回の呼び出しで 1 ページ分の画像を生成します。5 ページ揃えるには page_number を
1, 2, 3, 4, 5 と順番に 5 回呼んでください。

## 進め方のルール

- ユーザーがワークフローを進めるよう求めたら、まず inspect_artifacts を呼び、
  現在の artifacts と state を確認してください。
- そのうえで、必要な artifact が揃っていない最も上流のステップを 1 つだけ
  呼び出してください。
- 1 ターンに進めるのは 1 ステップだけです。連続して複数のステップを実行しないでください。
- ツール実行後は結果を報告してください。
- ツールが status=error を返した場合は、不足している artifact やエラー内容を
  説明して停止してください。

## コンテンツ挙動の保全

既存のコンテンツ生成挙動は変更しないでください。プロンプト、台本要件、画像生成指示、
出力フォーマット、その他コンテンツに影響する挙動の変更はプロジェクトオーナーの
明示的な承認なしには行わないでください。
""".strip(),
    tools=[
        inspect_artifacts,
        fetch_dosei_tool,
        enrich_news_tool,
        generate_scenario_tool,
        collect_assets_tool,
        resize_assets,
        generate_page_gemini,
    ],
)
