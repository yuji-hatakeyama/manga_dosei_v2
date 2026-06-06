# manga_dosei

報道された首相動静をもとに、その日の出来事を 1 ページの漫画にまとめる日次パイプラインです。

[English README](README.md)

![出力サンプル](assets/samples/sample.jpg)

> 生成されたページは公開されている予定情報をもとに AI が作成したフィクションです。描かれている人物・団体から承認・推薦・後援を受けたものではなく、これらと提携しておらず、これらの公式な見解を代表するものでもありません。内容についていかなる責任も負いません。

## 概要

`manga_dosei` は ADK を使ったワークフローで、対象日の首相の予定を 1 ページの漫画にまとめます。各実行は `target_date` (`YYYYMMDD`) をキーとした 1 セッションに対応し、各ステップが artifact を書き出して、次のステップがそれを入力にします。中間成果物をファイルとして残しているので、任意のステップだけを `adk web` から再実行することもできます。

## パイプライン

```mermaid
flowchart TD
    CLI([uv run manga_dosei YYYYMMDD])

    FD[fetch_dosei<br/>LlmAgent]

    subgraph EN["enrich_news · SequentialAgent"]
        direction TB
        ENR[section_researcher<br/>LlmAgent]
        subgraph ENL["LoopAgent · max_iterations=2"]
            direction TB
            ENV[research_evaluator<br/>LlmAgent] --> ENG{grade?}
            ENG -- fail --> ENX[enhanced_search_executor<br/>LlmAgent]
        end
        ENO[news_composer<br/>LlmAgent]
        ENR --> ENL
        ENG -- pass --> ENO
    end

    SU[summarize_url<br/>AgentTool · LlmAgent]
    ENR -. uses .-> SU
    ENX -. uses .-> SU

    GS[generate_scenario<br/>LlmAgent]
    CA[collect_assets<br/>LlmAgent]
    DL[define_layout<br/>LlmAgent]
    CB[compose_image_brief<br/>LlmAgent]
    GP[generate_page_gemini × 5<br/>Gemini Image]

    CLI ==> FD
    FD == dosei.md ==> EN
    EN == news.md ==> GS
    GS == scenario.md ==> CA
    CA == "assets/* + manifests/assets.json (リサイズ後 ≤1024px)" ==> DL
    DL == layout.md ==> CB
    CB == image_brief.md ==> GP
    GP ==> OUT[/"pages/gemini_1..5"/]
```

各ステップは ADK の 1 ツールに対応し、それぞれ後段が依存する artifact を出力します。**ストーリー**・**版下構造**・**画像生成ブリーフ** を別ファイルに分けてあるので、1 つだけ作り直しても全体を再生成する必要はありません。図を見やすくするために省いた非 LLM ヘルパー (Tavily / httpx / Wikimedia API など) は「使用ツール」列にまとめています。

| # | ステップ | 出力 artifact | 使用ツール | 役割 |
|---|---|---|---|---|
| 1 | `fetch_dosei` | `dosei.md` | `search_jiji_for_dosei` (Tavily / jiji.com 限定)、`fetch_url` (httpx) | 記事 URL を探して本文 HTML を取得し、対象日の `首相動静` を末尾の配信日時行まで一字一句転記する。要約はしない。 |
| 2 | `enrich_news` | `news.md` | `search_news_jiji` / `search_news_yahoo` (Tavily)、`summarize_url` | `SequentialAgent` が 3 段階で深掘りする: **section_researcher** (初期調査) → **LoopAgent** (最大 2 周。**research_evaluator** が pass/fail を判定し、fail なら **enhanced_search_executor** が再調査) → **news_composer** (調査結果と `dosei.md` を markdown にまとめる)。漫画ネタ候補・人物プロフィール・面会の文脈・周辺の政治情勢を出典付きで含む。`summarize_url` は子 `LlmAgent` 内で本文を要約し、生 HTML を親 context に漏らさない。 |
| 3 | `generate_scenario` | `scenario.md` | — | `news.md` を元に漫画台本を起こす: タイトル、コマタイトル、コマごとの 状況 / イラスト / セリフ (verbatim・番号付き)、登場人物一覧、X 投稿用テキスト。**ストーリーの正典** で、以降のステップはここに書かれた内容を組み直したり描画したりするだけ。 |
| 4 | `collect_assets` | `assets/<name>.<ext>` + `manifests/assets.json` | `wiki_image_search` / `wiki_image_info` (Wikimedia)、`download_image` (httpx) | `scenario.md` の登場人物一覧を読み、参考画像を最大 7 枚 (人物優先) 集める。manifest にはソース URL、ライセンス、MIME type を記録する。 |
| 5 | `resize_assets` | `assets/<name>.<ext>` の新バージョン | Pillow (LANCZOS) | 長辺が 1024px を超える参考画像のみ縮小し、新バージョンとして書き戻す。元バージョンは artifact ストアに残る。 |
| 6 | `define_layout` | `layout.md` | — | コマ数と展開からレイアウトカタログ (`assets/layouts/{3a,4a,4b,4c,4d,5a}/`) の `pattern_id` を 1 つ選び、ASCII 図と段配置を一字一句転記する。さらにセリフ番号順から導いたコマ別キャラ配置 (画面左 / 画面右) を書く。**版下構造のみ** で、タイトル・セリフ・視覚要素は持たない。 |
| 7 | `compose_image_brief` | `image_brief.md` | — | `scenario.md` + `layout.md` + `news.md` + `manifests/assets.json` を統合した、画像生成のための単一ブリーフ。ページヘッダ (verbatim)、登場人物プロフィール (参照画像あり/なしフラグ付き)、layout の ASCII 図、コマ別仕様 (位置 + キャラ配置 + 状況/イラストの手がかり + verbatim 吹き出し) を含む。`(verbatim)` と書かれたフィールドだけがページに文字として描画される。 |
| 8 | `generate_page_gemini` (×5) | `pages/gemini_<N>.<jpg\|png>` | Gemini Image | `image_brief.md` と 3 種類の参照画像 (該当 `pattern_id` のレイアウトサンプル、高市早苗キャラ参照 `assets/samples/sanae.jpg`、リサイズ済み `assets/*`) を渡す。内部リトライなしで `PAGE_VARIANT_COUNT` 回呼ばれ、得られたバリアントから手動で最良を選ぶ。 |

日次 CLI は現状 Gemini Image のみを利用します。`generate_page_gpt` (OpenAI GPT Image、`PAGE_VARIANT_COUNT=2`) は ADK エージェントには登録されていて `adk web` から呼び出せますが、`uv run manga_dosei` 経由では呼ばれません (構図と文字描画の安定性が Gemini のほうが高いため)。

## 使用技術

- **言語**: Python 3.12
- **ツール / ライブラリ**: uv, httpx, BeautifulSoup, Pillow ほか
- **フレームワーク**: [Google ADK](https://github.com/google/adk-python)
- **LLM (text)**: Gemini (デフォルト `gemini-3.1-pro-preview`)
- **LLM (image)**: Gemini Image (デフォルト `gemini-3-pro-image-preview`)。OpenAI GPT Image (`gpt-image-2`) はエージェント経由で利用可能ですが日次 CLI からは呼びません
- **Web 検索**: [Tavily REST API](https://docs.tavily.com/) を `make_tavily_search_tool` でコード側パラメータ固定にして利用 (LLM は `query` のみ決める)
- **データソース**: jiji.com、Yahoo!ニュース (news.yahoo.co.jp)、Wikimedia Commons

## 必要環境

日次 CLI には Gemini と Tavily の API キーと、Wikimedia 用の連絡先メールアドレスが必要です。`OPENAI_API_KEY` は `adk web` から `generate_page_gpt` を呼ぶ場合のみ必要です (`.env.example` 参照)。

## セットアップ

```bash
uv sync
cp .env.example .env   # GEMINI_API_KEY, OPENAI_API_KEY, TAVILY_API_KEY, WIKIMEDIA_CONTACT_EMAIL を記入
```

## 使い方

対象日 (`YYYYMMDD`) を指定してパイプラインを一括実行します:

```bash
uv run manga_dosei 20260410
```

セッションと artifact は `manga_dosei` パッケージディレクトリの親に固定された `.adk/` 配下に保存されます。書き出し先は CWD に依存しないので、サブディレクトリから実行しても同じ場所に書き込まれます。デフォルトの editable な `uv sync` インストールでは `<repo>/.adk/` に解決されます。

### 別 session id で再実行する

session id はデフォルトで `target_date` と同じため、同じ日付を再実行すると `.adk/` 配下のセッションと artifact が前回の分と混ざります。`--session-id` を渡すと別セッションとして独立に残せます:

```bash
uv run manga_dosei 20260410 --session-id 20260410_retry1
```

値は `^\d{8}(_[A-Za-z0-9_-]+)?$` にマッチし、かつ先頭 8 文字が `target_date` と一致する必要があります (session id は artifact ストレージのサブディレクトリ名にも使われるため、サフィックスはファイルシステム安全な文字に限定されます)。

### 成果物のディレクトリへの書き出し

`--publish-dir PATH` を渡すと、パイプライン完了後に各 artifact の最新バージョンを `PATH` 配下に、artifact 名のスラッシュ階層をそのまま保持して書き出します (例: `pages/gemini_1.jpg` → `PATH/pages/gemini_1.jpg`):

```bash
uv run manga_dosei 20260410 --publish-dir /tmp/manga-out
```

### GitHub のアーカイブ repo へ push

`manga_dosei-publish` はディレクトリの中身を GitHub の private repo に 1 fast-forward commit でまとめて push する独立した CLI です。日次ランをアーカイブ用 private repo に送る用途に使います:

```bash
GITHUB_OUTPUT_TOKEN=<fine-grained PAT> \
uv run manga_dosei-publish \
  --source /tmp/manga-out \
  --repo owner/manga_dosei_v2_artifacts \
  --dest 2026/04/20260410 \
  --message "publish: 20260410"
```

- `--source` 配下で名前が `.` で始まるファイル / ディレクトリはスキップされ、シンボリックリンクも辿りません (`.env` / `.git/` / `.adk/` / エディタの swap file などは除外されます)。同じコミットに同梱したいログ (リダイレクトした stdout/stderr など) は `.` で始めないでください。
- `--dest` はクリーンな相対 POSIX パスである必要があります (詳細な検証ルールは `manga_dosei-publish --help` 参照)。空 / `/` を渡すと repo ルートに push されます。

token はプロセスの `GITHUB_OUTPUT_TOKEN` 環境変数からのみ読み、CLI 引数では受け付けません。さらに `manga_dosei-publish` はディスク上の `.env` を意図的に読みません。`--source` ディレクトリや CI の作業ディレクトリにまぎれた `.env` から token が注入される事故を防ぐためです。パイプライン本体 (`uv run manga_dosei`) は通常どおり CWD の `.env` を読みますが、publish CLI のみこの挙動を opt-out しています。

### 対話的にセッションを操作する

ADK Web UI から同じストレージを参照して状態を確認・再開する場合:

```bash
uv run adk web \
  --session_service_uri="sqlite:///$(pwd)/.adk/sessions.db" \
  --artifact_service_uri="file://$(pwd)/.adk/artifacts" \
  .
```

CLI が書き出す `.adk/` (上記のとおり通常は `<repo-root>/.adk/`) と同じディレクトリから `adk web` を起動し、`$(pwd)/.adk/...` が同じパスに解決されるようにしてください。絶対パスが必須です — `file://./...` の形式だと `.` が host として解釈されて「`file://` artifact URIs must reference the local filesystem.」で起動に失敗します。

## テスト

```bash
uv run pytest
```

`tests/` 配下の純粋な単体テストのみ (LLM・ネットワーク・`.adk/` ストレージには触れません)。
