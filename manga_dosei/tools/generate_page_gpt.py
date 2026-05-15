"""generate_page_gpt: OpenAI GPT Image を使って 1 ページ分の漫画画像を生成する。

generate_page_gemini の OpenAI 版。1 回呼ばれたら 1 回だけ
OpenAI Images Edit API を呼び、画像を artifact として保存する。
内部 retry は持たない（CLI 側で page_number を変えながら複数回呼ぶ）。

入力は前段 compose_image_brief が生成する image_brief.md artifact 一本。
これに ページタイトル / 登場人物プロフィール / ページレイアウト / コマ別仕様
(verbatim セリフ・視覚要素含む) / 描画前チェックリスト / 免責文言 が
すべて入っているため、本ツールの prompt template は画風・参照画像の使い方など
描画固有ルールに専念する。
"""

import base64
import os
import re
from pathlib import Path
from typing import Any

from google.adk.tools import ToolContext
from google.genai import types
from openai import AsyncOpenAI

from manga_dosei.validation import validate_target_date

_STEP = "generate_page_gpt"
_MODEL_LABEL = "gpt"
_DEFAULT_IMAGE_MODEL = "gpt-image-2"
_IMAGE_SIZE = "1024x1536"
_IMAGE_QUALITY = "high"
_OUTPUT_FORMAT = "png"
_OUTPUT_MIME = "image/png"
_OUTPUT_EXT = ".png"

# モデルごとに当たり外れの幅が異なるので、生成バリアント数はツール側で持つ。
PAGE_VARIANT_COUNT = 2

_REPO_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
_LAYOUTS_DIR = _REPO_ASSETS_DIR / "layouts"
_CHARACTER_REF_PATH = _REPO_ASSETS_DIR / "samples" / "sanae.jpg"

_PATTERN_ID_RE = re.compile(r"^-\s*pattern_id:\s*([A-Za-z0-9_-]+)\s*$", re.MULTILINE)


_PROMPT_INTRO = """## 指示

下記の「画像生成ブリーフ」を元に、漫画ページを生成してください。
ブリーフには ページタイトル・登場人物プロフィール・ページレイアウト・
コマ別仕様 (verbatim セリフ/視覚要素含む)・描画前チェックリスト・
免責文言 がすべて含まれています。

### ブリーフの読み方 (最重要)

* `(verbatim)` と注釈が付いたフィールド (日付表記 / 主タイトル本文 /
  コマタイトル / 吹き出し / ページ下部) は **画像内に文字として
  一字一句そのまま描画** してください。漢字・ひらがな・数字・記号を
  含め、誤字・字形変更・改行追加・空白挿入は禁止です。
* それ以外のフィールド (登場人物プロフィール / ページレイアウト /
  位置 / キャラ配置 / 視覚要素 / 描画前チェックリスト) は
  **画像生成 AI への描画手がかり** であり、**これらの語句を文字として
  ページに描画してはいけません**。
* 「ページレイアウト」の ASCII 図と「## 段ごとの配置」は **コマ枠の
  並び順の絶対指示** です。各コマの「位置」フィールドと矛盾なく描画
  してください。コマを勝手に全幅化したり段数を変えたりしないこと。
  添付の【ページ例】は同じ pattern_id のサンプル画像で **ASCII 図と完全に
  同じコマ配置** になっているので、コマ枠の数・並び・分割の比率は
  【ページ例】に忠実に従ってください (内容ではなくコマ枠の形を真似る)。
* 「描画前チェックリスト」は描画前の最終確認用です。各コマの実際の
  キャラ左右配置と必ず一致させてください。
* ブリーフに無いキャプション・横断幕・看板の説明文・字幕・キャラ名
  ラベル・日付ラベル等を背景や前景に追加しないでください。背景文字や
  小道具の文字は「視覚要素」に明記されたものに限定。

### 吹き出しの描画ルール

* 吹き出しに描画するのは `<話者名>「<本文>」` の **`「」` 内の本文のみ**。
  以下は **描画しない** こと:
  * 先頭の番号プレフィックス (`1.`, `2.` など) — 読み順マーカー
  * 話者ラベル (「」の前の名前) — 発言者指定のメタ情報
  * `（心の声）` 等の補助注釈 — 必要なら吹き出し形状や表情で表現
* 吹き出しのテール (しっぽ) は **「キャラ配置」で指定された話者の
  口元へ** 必ず向けること。別人から出ているように見える配置は不可。

### 画風と文字要素

* ページは A4 サイズ (縦長)。
* **ヘッダ (日付とタイトル)**: ページ上部に 2 行構成で配置。
  1 行目=日付 (タイトルより小さめの黒色太字)、
  2 行目=タイトル本文 (日付より **明確に大きい** 黒色太字)。
  改行位置・フォント・サイズ比・配置・行間・余白は添付【ページ例】に
  忠実に従う。
* 各コマの左上に **コマタイトル** を `#F0E68C` 背景色・黒色太字
  ゴシック体で描画。フォーマット詳細は【ページ例】に揃える。
* ウォーターマークを入れない。
* **画風**: 日本のフラットなアニメ/セル画調。均一なベタ塗りを基本とし、
  立体的なグラデーション・強いハイライト・髪や肌のツヤ・写実的な
  ライティングや影は使わない。背景もアニメ背景として描画し、写真の
  ようなディテール・質感は入れない。全体のトーンは【ページ例】に
  忠実に揃える (Pixar 風・3D CG 風・セミリアル調にはしない)。
* **文字要素 (フォント・サイズ・色・太さ・装飾)**: すべての文字要素
  について【ページ例】に忠実に揃える。独自の装飾的フォントや派手な
  色は使わず、【ページ例】の書体・配色ルールを厳密に模倣すること。

### 参照画像について

このプロンプトには以下の画像が添付されています。各ラベルに従って使用
してください。

* **【ページ例】**: pattern_id に対応する正準サンプル画像で、ブリーフの
  「ページレイアウト」ASCII 図と **同じコマ配置** になっています。
  **コマ枠レイアウト** (コマ数・段数・分割比率) と **画風・フォーマット**
  (タイトル書式、コマタイトルのデザイン、フォント、色使い、線の太さ、
  塗り方、陰影、全体のビジュアルスタイル) の両方の参考にしてください。
  **内容を同じにする必要はありません** が、画風・質感・トーン・文字要素の
  フォント・サイズ・色・配置ルール・コマ枠配置はこの画像を忠実に模倣
  してください。
* **【参考資料: ○○】**: 登場人物のキャラクターデザイン。すべてのコマで
  この画像の顔の特徴を忠実に再現し、別人に見えないようにすること。
  ブリーフの「登場人物プロフィール」で「参照画像: なし」となっている
  キャラは、視覚特徴の手がかりだけを参考に違和感のない人物を描く。

### 画像生成ブリーフ

```
"""


# OpenAI Images Edit API は画像をフラットなリストでしか受け取れないため
# (Gemini のようにテキスト/画像を交互配置できない)、添付順を別途宣言する。
_IMAGE_ORDER_TEMPLATE = """

### 添付画像の順序（重要）

このリクエストには複数の参照画像を添付順に渡しています。下のリストの
順序で各画像のラベル・用途を解釈してください。

{lines}
"""


_FINAL_REMINDER = """## 🛑 描画開始直前の最終確認

1. すべての複数人コマで、ブリーフの「🔴 描画前チェックリスト」の左右位置を
   満たしているか。
2. 各吹き出しのテールが、ブリーフの「キャラ配置」で指定された発言者の
   口元へ向いているか。
3. ページレイアウトの ASCII 図および各コマの「位置」と矛盾せず、コマを
   勝手に全幅化したり段数を変えたりしていないか。

これらを満たさない場合、漫画として成立しない。生成前に必ず再確認すること。
"""


async def generate_page_gpt(
    target_date: str,
    page_number: int,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """1 ページ分の漫画画像を OpenAI GPT Image で 1 回生成し、artifact 保存する。

    前提:
        image_brief.md artifact が存在すること
        (compose_image_brief 完了想定)。
        必要に応じて assets/* artifact を OpenAI に参考画像として添付する。

    挙動:
        - 1 回だけ OpenAI Images Edit API を呼ぶ（内部 retry なし）。
        - 成功時は pages/gpt_<N>.png として artifact 保存する。
        - 失敗時は status=error を返し、CLI 側で必要なら 1 回 retry される。

    引数:
        target_date: YYYYMMDD 形式の対象日。
        page_number: バリアント番号（artifact 名 pages/gpt_<N>.png に使う）。

    返り値（成功時）:
        {"status": "success", "step": "generate_page_gpt",
         "page_number": int, "artifact": "pages/gpt_<N>.png",
         "version": int, "bytes": int, "mime_type": "image/png"}

    返り値（失敗時）:
        {"status": "error", "step": "generate_page_gpt",
         "page_number": int, "message": str}
    """
    try:
        validate_target_date(target_date)
    except ValueError as error:
        return _error(str(error), page_number=page_number)

    if not _CHARACTER_REF_PATH.exists():
        return _error(
            f"character reference image missing: {_CHARACTER_REF_PATH}",
            page_number=page_number,
        )

    brief_part = await tool_context.load_artifact("image_brief.md")
    if brief_part is None or brief_part.text is None:
        return _error(
            "image_brief.md is missing or unreadable (run compose_image_brief first)",
            page_number=page_number,
        )
    brief_text = brief_part.text

    pattern_id_match = _PATTERN_ID_RE.search(brief_text)
    if not pattern_id_match:
        return _error(
            "pattern_id not found in image_brief.md "
            "(expected `- pattern_id: <id>` near the top)",
            page_number=page_number,
        )
    pattern_id = pattern_id_match.group(1)
    sample_page_path = _LAYOUTS_DIR / pattern_id / "sample.jpg"
    if not sample_page_path.exists():
        return _error(
            f"layout sample image missing for pattern_id={pattern_id}: "
            f"{sample_page_path}",
            page_number=page_number,
        )

    images: list[tuple[str, bytes, str]] = [
        ("sample.jpg", sample_page_path.read_bytes(), "image/jpeg"),
        ("sanae.jpg", _CHARACTER_REF_PATH.read_bytes(), "image/jpeg"),
    ]
    image_labels: list[str] = ["【ページ例】", "【参考資料: 高市早苗】"]

    asset_keys = sorted(
        key for key in await tool_context.list_artifacts() if key.startswith("assets/")
    )
    for key in asset_keys:
        asset_part = await tool_context.load_artifact(key)
        if (
            asset_part is None
            or asset_part.inline_data is None
            or not asset_part.inline_data.data
        ):
            continue
        mime = (asset_part.inline_data.mime_type or "image/jpeg").lower()
        # OpenAI Images Edit が受け付けるのは png/jpeg/webp のみ。
        if mime not in {"image/png", "image/jpeg", "image/webp"}:
            continue
        filename = key.removeprefix("assets/") or "asset"
        asset_name = filename.rsplit(".", 1)[0] if "." in filename else filename
        images.append((filename, asset_part.inline_data.data, mime))
        image_labels.append(f"【参考資料: {asset_name}】")

    order_lines = "\n".join(f"{i + 1}. {label}" for i, label in enumerate(image_labels))
    prompt = (
        f"{_PROMPT_INTRO}{brief_text}\n```"
        f"{_IMAGE_ORDER_TEMPLATE.format(lines=order_lines)}"
        f"\n\n---\n\n{_FINAL_REMINDER}"
    )

    model_name = os.getenv("OPENAI_IMAGE_MODEL", _DEFAULT_IMAGE_MODEL)
    try:
        client = AsyncOpenAI()
        response = await client.images.edit(
            model=model_name,
            image=list(images),
            prompt=prompt,
            size=_IMAGE_SIZE,
            quality=_IMAGE_QUALITY,
            output_format=_OUTPUT_FORMAT,
            n=1,
        )
    except Exception as error:
        return _error(f"openai call failed: {error}", page_number=page_number)

    data_items = getattr(response, "data", None) or []
    if not data_items or not getattr(data_items[0], "b64_json", None):
        return _error("no image data in response", page_number=page_number)

    try:
        image_bytes = base64.b64decode(data_items[0].b64_json)
    except (ValueError, TypeError) as error:
        return _error(f"failed to decode image: {error}", page_number=page_number)

    artifact_name = f"pages/{_MODEL_LABEL}_{page_number}{_OUTPUT_EXT}"
    version = await tool_context.save_artifact(
        artifact_name,
        types.Part(inline_data=types.Blob(data=image_bytes, mime_type=_OUTPUT_MIME)),
    )

    tool_context.state.update(
        {
            "target_date": target_date,
            f"page_{_MODEL_LABEL}_{page_number}_artifact": artifact_name,
            f"page_{_MODEL_LABEL}_{page_number}_version": version,
            "last_error": None,
        }
    )

    return {
        "status": "success",
        "step": _STEP,
        "page_number": page_number,
        "artifact": artifact_name,
        "version": version,
        "bytes": len(image_bytes),
        "mime_type": _OUTPUT_MIME,
    }


def _error(message: str, *, page_number: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": "error", "step": _STEP, "message": message}
    if page_number is not None:
        payload["page_number"] = page_number
    return payload
