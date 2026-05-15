# manga_dosei

報道された首相動静をもとに、その日の出来事を 1 ページの漫画にまとめる日次パイプラインです。

[English README](README.md)

![出力サンプル](assets/samples/sample.jpg)

> 生成されたページは公開情報をもとに AI が作成したフィクションです。登場する人物・団体とは一切関係なく、内容について一切の責任を負いません。

## 概要

`manga_dosei` は ADK を使ったワークフローで、対象日の首相の予定を 1 ページの漫画にまとめます。処理の流れ:

1. JIJI.COM (www.jiji.com) から対象日の `首相動静` を取得
2. Tavily で関連ニュースを調査して肉付け
3. 漫画台本を生成
4. 登場人物・場所の参考画像を Wikimedia Commons から収集してリサイズ
5. 台本からページレイアウト（コマ並び・各コマのキャラ配置・吹き出し読み順）を構造化して書き出し
6. Gemini Image で漫画ページを複数バリアント生成（その中から手動で選択）

日次 CLI は現状 Gemini Image のみを利用します。OpenAI GPT Image バックエンドは ADK エージェントには登録されていて `adk web` から呼び出せますが、`uv run manga_dosei` 経由では呼ばれません。

## 使用技術

- **言語**: Python 3.12
- **ツール / ライブラリ**: uv, httpx, BeautifulSoup, Pillow ほか
- **フレームワーク**: [Google ADK](https://github.com/google/adk-python)
- **LLM (text)**: Gemini (デフォルト `gemini-3.1-pro-preview`)
- **LLM (image)**: Gemini Image (デフォルト `gemini-3-pro-image-preview`)。OpenAI GPT Image (`gpt-image-2`) はエージェント経由で利用可能ですが日次 CLI からは呼びません
- **Web 検索 (MCP)**: [Tavily](https://docs.tavily.com/)
- **データソース**: JIJI.COM (www.jiji.com)、Wikimedia Commons

## 必要環境

Gemini、OpenAI、Tavily の API キー (`.env.example` 参照)。

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

セッションと artifact は `.adk/` 配下に保存されます。

ADK Web UI から同じストレージを参照して状態を確認・再開する場合:

```bash
uv run adk web \
  --session_service_uri="sqlite:///$(pwd)/.adk/sessions.db" \
  --artifact_service_uri="file://$(pwd)/.adk/artifacts" \
  .
```

リポジトリのルートで実行してください (CLI が書く `.adk/` と同じディレクトリを `$(pwd)` で参照させるため)。絶対パスが必須です — `file://./...` の形式だと `.` が host として解釈されて「`file://` artifact URIs must reference the local filesystem.」で起動失敗します。
