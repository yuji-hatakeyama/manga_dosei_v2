"""ADK implementation for manga dosei workflows."""

from pathlib import Path

from dotenv import load_dotenv

from manga_dosei.config import Settings, get_settings
from manga_dosei.names import ArtifactName, StateKey

# tools モジュールが import 時に Tavily 等の環境変数を要求するため、
# パッケージ読み込み時に `.env` を一度だけロードしておく
# (既存の環境変数は上書きしない — シェル経由の値を優先するため)。
# NOTE: ここでの `load_dotenv` は副作用で os.environ を書き換えるので、
# テストでローカル `.env` の値を遮断したい場合は `manga_dosei` を import
# する前に env を scrub する必要がある (tests/conftest.py 参照)。
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

APP_NAME = "manga_dosei"
DEFAULT_USER_ID = "daily"

__all__ = [
    "APP_NAME",
    "DEFAULT_USER_ID",
    "Settings",
    "get_settings",
    "ArtifactName",
    "StateKey",
]
