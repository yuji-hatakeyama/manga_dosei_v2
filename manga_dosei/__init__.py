"""ADK implementation for manga dosei workflows."""

from pathlib import Path

from dotenv import load_dotenv

# tools モジュールが import 時に Tavily 等の環境変数を要求するため、
# パッケージ読み込み時に `.env` を一度だけロードしておく。
# 既存の環境変数は上書きしない (シェルから渡された値が優先)。
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

APP_NAME = "manga_dosei"
DEFAULT_USER_ID = "daily"
DEFAULT_TEXT_MODEL = "gemini-3.1-pro-preview"
