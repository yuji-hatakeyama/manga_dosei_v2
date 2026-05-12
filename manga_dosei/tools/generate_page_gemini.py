"""generate_page_gemini: Gemini Image を使って 1 ページ分の漫画画像を生成する。

1 回呼ばれたら 1 回だけ Gemini Image API を呼び、画像を artifact として保存する。
内部 retry は持たない（CLI 側で page_number を変えながら複数回呼ぶ）。

レイアウト指示は前段の define_layout が生成する layout.md artifact から取得し、
プロンプト冒頭にそのまま貼り込む。コマ配置・読み順・吹き出し順は layout.md が
カバーするため、本ツールの prompt template は画風・文字・参照画像・台本転記等の
描画固有ルールに専念する。
"""

import os
from pathlib import Path
from typing import Any

from google import genai
from google.adk.tools import ToolContext
from google.genai import types

from manga_dosei.validation import validate_target_date

_STEP = "generate_page_gemini"
_MODEL_LABEL = "gemini"
_DEFAULT_IMAGE_MODEL = "gemini-3-pro-image-preview"
_ALLOWED_IMAGE_MIME = {"image/jpeg", "image/png"}

# モデルごとに当たり外れの幅が異なるので、生成バリアント数はツール側で持つ。
PAGE_VARIANT_COUNT = 5

_SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "samples"
_SAMPLE_PAGE_PATH = _SAMPLES_DIR / "sample.jpg"
_CHARACTER_REF_PATH = _SAMPLES_DIR / "sanae.jpg"


_PROMPT_TEMPLATE = """## 指示

冒頭の「📐 レイアウト指示書」と下記の「対象の台本」を元に、漫画ページを生成してください。
**ページ構成（コマの並び順、各コマ内のキャラ配置、吹き出しの読み順）は
レイアウト指示書に従ってください。** 台本はセリフ本文・場面描写の原典として参照します。

### 要件

* ページは A4サイズ (縦長の方向)です。
* **ヘッダ（日付とタイトル）**: ページ上部のヘッダは**2行構成**で配置してください。
  * 1行目: 日付のみ（例: `2026年04月18日`）。
    タイトルより小さめの黒色太字。
  * 2行目: タイトル本文（例: `静養！公邸でじっくり英気を養う土曜日や！`）。
    日付より**明確に大きい**黒色太字。
  * 台本の「タイトル」欄は `YYYY年MM月DD日 <タイトル本文>` の形式で記載されているので、
    先頭の日付部分を1行目、それ以降を2行目に分けて描画してください。
  * 改行位置・フォント・サイズ比・配置（左寄せ/中央など）・行間・余白は
    添付された【ページ例】に忠実に従ってください。
* 各コマの左上には「コマタイトル」を記載します。
  台本に書かれたコマタイトルを、`#F0E68C` の背景色、黒色の太字ゴシック体で、
  そのまま正確に描画してください。
  詳細なフォーマット（フォント、サイズ、配置など）は
  添付された「ページ例」のスタイルに揃えてください（内容は台本に従ってください）。
* ウォーターマークは入れないでください
* **台本の文章を一字一句そのまま使用**:
  台本に記載されているコマタイトル、セリフ、ナレーション、擬音、説明文などの文章は、
  一切変更・創作せずにそのまま正確に描画してください。
  漢字・ひらがな・数字を含め、誤字や字形の間違いは避けてください。
* **描画してはいけない台本上の記法**:
  台本のセリフは「`1. 高市総理「...セリフ本文...」`」のような形式で書かれています。
  **吹き出しに描画するのは「」内の本文のみ**です。以下は**描画しないでください**:
  * 先頭の番号プレフィックス（`1.`、`2.`、`3.` など）—
    これはコマ内でのセリフの読み順を示すマーカーであり、吹き出しに含めてはいけません。
  * 話者ラベル（`高市総理`、`小泉防衛相`、`マールズ副首相` など「」の前に書かれた名前）—
    これは発言者を指定するためのメタ情報であり、吹き出しに描画する文字ではありません。
    話者ラベルとレイアウト指示書に従い、**吹き出しのしっぽを該当キャラクターへ向けてください**。
  * `（心の声）`、`（資料を読みながら）` などの補助注釈は、
    必要に応じて吹き出しの形（思考バルーン等）や表情・仕草で表現し、
    文字として吹き出し内に描画する必要はありません。
* **吹き出しのテール（しっぽ）は発言者の口元へ**:
  各吹き出しのテールは、レイアウト指示書で指定された発言者キャラクターの口元へ
  必ず向けてください。別人から出ているように見える配置は不可。
  特に高市総理（女性）のセリフを男性閣僚から出すような、性別を跨ぐ取り違えは
  絶対に避けてください。
* **キャラクターリファレンス**:
  添付された「キャラクターリファレンス」画像を参照して、
  登場人物の顔や外見を統一してください。
  台本に登場する各キャラクターについて、対応するリファレンス画像がある場合は、
  すべてのコマで同じキャラクターデザインを使用してください。
  表情やポーズは変えても構いませんが、**髪型・髪色・年齢感・眼鏡の有無・
  特徴的な顔の輪郭**はリファレンス画像と一致させ、別人に見えないように
  してください（リアル写真はアニメ調にトレースし、写実的にはしない）。
  複数のキャラクターリファレンスが添付されている場合は、
  それぞれのキャラクターを正しく識別して使用してください。
* **画風**:
  日本のフラットなアニメ/セル画調で描画してください。
  均一なベタ塗りを基本とし、
  立体的なグラデーション・強いハイライト・髪や肌のツヤ・写実的なライティングや影は
  使わないでください。
  背景もアニメ背景として描画し、写真のようなディテールや質感を入れないでください。
  全体のトーンは添付された【ページ例】に忠実に揃えてください
  （Pixar 風・3D CG 風・セミリアル調にはしない）。
* **文字要素（フォント・サイズ・色）**:
  タイトル、コマタイトル、セリフ、ナレーション、擬音、説明文、免責事項など
  **すべての文字要素**について、
  **フォント（書体）・サイズ・色・太さ・装飾**を【ページ例】に忠実に揃えてください。
  独自の装飾的フォントや派手な色は使わず、
  【ページ例】で使用されている書体・配色のルールを厳密に模倣してください。

### 参照画像について

このプロンプトには以下の画像が添付されています。各ラベルに従って使用してください。

* **【ページ例】**: レイアウト、フォーマット、スタイル（タイトルの書式、コマタイトルの
  デザイン、フォント、色使い、線の太さ、塗り方、陰影の付け方、全体的なビジュアルスタイルなど）の参考です。
  **内容を同じにする必要はありません**が、**画風・質感・トーン、および文字要素のフォント・サイズ・色・配置ルールはこの画像を忠実に模倣してください**。
  特に「フラットなアニメ/セル画調」「均一なベタ塗り」「控えめな陰影」のスタイルは厳密に再現し、
  写実的・立体的・3D CG 的な質感に寄せないでください。
  台本の内容に基づいて描画してください。
* **【参考資料: ○○】**: 登場人物のキャラクターデザインです。○○の部分に
  キャラクター名が入ります。すべてのコマで、この画像の顔の特徴（目、鼻、口、髪型など）を
  忠実に再現してください。複数のキャラクターリファレンスがある場合は、それぞれ対応する
  キャラクターに使用してください。

### 対象の台本

```
"""


_LAYOUT_HEADER = """## 📐 レイアウト指示書（絶対遵守）

下記はこのページを描画するための **版下設計書** です。
コマの並び順、各コマ内のキャラクター配置、吹き出しの読み順は
**この指示書に厳密に従ってください**。
画像生成の創造性を発揮するのは「視覚要素」「ポーズ」「表情」「背景の描き方」
といった描画ディテールに限定し、配置・順序の判断はこの指示書で確定済みです。

"""


async def generate_page_gemini(
    target_date: str,
    page_number: int,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """1 ページ分の漫画画像を Gemini Image で 1 回生成し、artifact 保存する。

    前提:
        scenario.md および layout.md artifact が存在すること
        （generate_scenario / define_layout 完了想定）。
        必要に応じて assets/* artifact を Gemini に参考画像として添付する。

    挙動:
        - 1 回だけ Gemini Image API を呼ぶ（内部 retry なし）。
        - 成功時は pages/gemini_<N>.<ext> として artifact 保存する。
        - 失敗時は status=error を返し、CLI 側で必要なら 1 回 retry される。

    引数:
        target_date: YYYYMMDD 形式の対象日。
        page_number: バリアント番号（artifact 名 pages/gemini_<N>.<ext> に使う）。

    返り値（成功時）:
        {"status": "success", "step": "generate_page_gemini",
         "page_number": int, "artifact": "pages/gemini_<N>.<ext>",
         "version": int, "bytes": int, "mime_type": str}

    返り値（失敗時）:
        {"status": "error", "step": "generate_page_gemini",
         "page_number": int, "message": str}
    """
    try:
        validate_target_date(target_date)
    except ValueError as error:
        return _error(str(error), page_number=page_number)

    if not _SAMPLE_PAGE_PATH.exists() or not _CHARACTER_REF_PATH.exists():
        return _error(
            "sample images missing in adk/assets/samples/",
            page_number=page_number,
        )

    scenario_part = await tool_context.load_artifact("scenario.md")
    if scenario_part is None or scenario_part.text is None:
        return _error(
            "scenario.md is missing or unreadable",
            page_number=page_number,
        )
    scenario_text = scenario_part.text

    layout_part = await tool_context.load_artifact("layout.md")
    if layout_part is None or layout_part.text is None:
        return _error(
            "layout.md is missing or unreadable (run define_layout first)",
            page_number=page_number,
        )
    layout_text = layout_part.text

    contents: list[Any] = [
        f"{_LAYOUT_HEADER}{layout_text}\n\n---\n\n{_PROMPT_TEMPLATE}{scenario_text}\n```",
        "【ページ例】",
        types.Part.from_bytes(
            data=_SAMPLE_PAGE_PATH.read_bytes(),
            mime_type="image/jpeg",
        ),
        "【参考資料: 高市早苗】",
        types.Part.from_bytes(
            data=_CHARACTER_REF_PATH.read_bytes(),
            mime_type="image/jpeg",
        ),
    ]

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
        asset_name = key.removeprefix("assets/")
        if "." in asset_name:
            asset_name = asset_name.rsplit(".", 1)[0]
        contents.append(f"【参考資料: {asset_name}】")
        contents.append(
            types.Part(
                inline_data=types.Blob(
                    data=asset_part.inline_data.data,
                    mime_type=asset_part.inline_data.mime_type or "image/jpeg",
                )
            )
        )

    model_name = os.getenv("GEMINI_IMAGE_MODEL", _DEFAULT_IMAGE_MODEL)
    try:
        client = genai.Client()
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                image_config=types.ImageConfig(image_size="2K"),
            ),
        )
    except Exception as error:
        return _error(f"gemini call failed: {error}", page_number=page_number)

    image_bytes: bytes | None = None
    image_mime: str | None = None
    parts = response.parts or []
    for part in parts:
        if part.inline_data and part.inline_data.data:
            image_bytes = part.inline_data.data
            image_mime = (part.inline_data.mime_type or "").lower() or "image/jpeg"
            break

    if not image_bytes:
        return _error("no image part in response", page_number=page_number)
    if image_mime not in _ALLOWED_IMAGE_MIME:
        return _error(
            f"unexpected image mime: {image_mime}",
            page_number=page_number,
        )

    extension = ".jpg" if image_mime == "image/jpeg" else ".png"
    artifact_name = f"pages/{_MODEL_LABEL}_{page_number}{extension}"
    version = await tool_context.save_artifact(
        artifact_name,
        types.Part(inline_data=types.Blob(data=image_bytes, mime_type=image_mime)),
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
        "mime_type": image_mime,
    }


def _error(message: str, *, page_number: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": "error", "step": _STEP, "message": message}
    if page_number is not None:
        payload["page_number"] = page_number
    return payload
