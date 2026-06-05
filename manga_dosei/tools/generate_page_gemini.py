"""generate_page_gemini: Gemini Image を使って 1 ページ分の漫画画像を生成する。

1 回呼ばれたら 1 回だけ Gemini Image API を呼び、画像を artifact として保存する。
内部 retry は持たない（CLI 側で page_number を変えながら複数回呼ぶ）。

入力は前段 compose_image_brief が生成する image_brief.md artifact 一本。
これに ページタイトル / 登場人物プロフィール / ページレイアウト / コマ別仕様
(verbatim セリフ・視覚要素含む) / 描画前チェックリスト / 免責文言 が
すべて入っているため、本ツールの prompt template は画風・参照画像の使い方など
描画固有ルールに専念する。

【ページ例】は image_brief.md 先頭の `- pattern_id: <id>` から ID を読み取り、
`assets/layouts/<id>/sample.jpg` を選択添付する。サンプルとブリーフの ASCII 図は
同じレイアウトを表しており、Gemini に対しては両者を遵守させる。
"""

from typing import Any

from google import genai
from google.adk.tools import ToolContext
from google.genai import types

from manga_dosei import paths
from manga_dosei.config import get_settings
from manga_dosei.tools._common import build_error_result
from manga_dosei.tools._image_gen import (
    load_asset_parts,
    load_brief_and_layout_sample,
    save_page_artifact,
)

_STEP = "generate_page_gemini"
_MODEL_LABEL = "gemini"
_ALLOWED_IMAGE_MIME = {"image/jpeg", "image/png"}

# モデルごとに当たり外れの幅が異なるので、生成バリアント数はツール側で持つ。
PAGE_VARIANT_COUNT = 5


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
  **コマ枠レイアウト** (コマ数・段数・分割比率) と、**画風・フォーマット**
  (タイトル書式、コマタイトルのデザイン、フォント、色使い、線の太さ、
  塗り方、陰影、全体のビジュアルスタイル) の両方の参考にしてください。
  **内容を同じにする必要はありません** が、画風・質感・トーン・文字要素の
  フォント・サイズ・色・配置ルール・コマ枠配置はこの画像を忠実に模倣
  してください。
* **【参考資料: ○○】**: 登場人物のキャラクターデザイン。すべてのコマで
  この画像の顔の特徴 (目、鼻、口、髪型、年齢感、眼鏡の有無、特徴的な
  顔の輪郭) を忠実に再現し、別人に見えないようにしてください
  (リアル写真はアニメ調にトレースし、写実的にはしない)。
  複数のリファレンスがある場合は、それぞれ対応するキャラクターに
  使用してください。
  ブリーフの「登場人物プロフィール」で「参照画像: なし」となっている
  キャラは、視覚特徴の手がかりだけを参考に違和感のない人物を描く。

### 画像生成ブリーフ

```
"""


def _describe_empty_response(response: Any) -> str:
    """SDK response shape varies across versions / finish states, so chain getattr."""
    parts: list[str] = []

    candidates = getattr(response, "candidates", None) or []
    if candidates:
        finish_reason = getattr(candidates[0], "finish_reason", None)
        if finish_reason is not None:
            parts.append(f"finish_reason={getattr(finish_reason, 'name', finish_reason)}")
        safety_ratings = getattr(candidates[0], "safety_ratings", None) or []
        if safety_ratings:
            first = safety_ratings[0]
            category = getattr(first, "category", None)
            probability = getattr(first, "probability", None)
            parts.append(
                f"safety[0]={getattr(category, 'name', category)}/"
                f"{getattr(probability, 'name', probability)}"
            )

    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback is not None:
        block_reason = getattr(prompt_feedback, "block_reason", None)
        if block_reason is not None:
            parts.append(f"block_reason={getattr(block_reason, 'name', block_reason)}")

    return ", ".join(parts) if parts else "no diagnostic fields available"


async def generate_page_gemini(
    target_date: str,
    page_number: int,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """1 ページ分の漫画画像を Gemini Image で 1 回生成し、artifact 保存する。

    前提:
        image_brief.md artifact が存在すること
        (compose_image_brief 完了想定)。
        必要に応じて assets/* artifact を Gemini に参考画像として添付する。

    挙動:
        - 1 回だけ Gemini Image API を呼ぶ（内部 retry なし）。
        - 成功時は pages/gemini_<N>.<ext> として artifact 保存する。
        - 失敗時は status=error を返し、CLI 側で必要なら 1 回 retry される。

    引数:
        target_date: YYYYMMDD 形式の対象日。
        page_number: バリアント番号（artifact 名 pages/gemini_<N>.<ext> に使う）。

    返り値（成功時）:
        {"status": "success", "step": "generate_page_gemini", "message": str,
         "page_number": int, "artifact": "pages/gemini_<N>.<ext>",
         "version": int, "bytes": int, "mime_type": str}

    返り値（失敗時）:
        {"status": "error", "step": "generate_page_gemini",
         "page_number": int, "message": str}
    """
    preflight = await load_brief_and_layout_sample(
        tool_context,
        step=_STEP,
        target_date=target_date,
        page_number=page_number,
    )
    if isinstance(preflight, dict):
        return preflight
    brief_text, sample_page_path = preflight

    contents: list[Any] = [
        f"{_PROMPT_INTRO}{brief_text}\n```",
        "【ページ例】",
        types.Part.from_bytes(
            data=sample_page_path.read_bytes(),
            mime_type="image/jpeg",
        ),
        "【参考資料: 高市早苗】",
        types.Part.from_bytes(
            data=paths.CHARACTER_REF_PATH.read_bytes(),
            mime_type="image/jpeg",
        ),
    ]

    for asset in await load_asset_parts(tool_context):
        contents.append(f"【参考資料: {asset.asset_name}】")
        contents.append(
            types.Part(inline_data=types.Blob(data=asset.data, mime_type=asset.mime))
        )

    model_name = get_settings().gemini_image_model
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
        # NOTE: include exception type so ResourceExhausted / ConnectionError /
        # auth errors are distinguishable without grepping logs.
        return build_error_result(
            tool_context,
            _STEP,
            f"gemini call failed ({type(error).__name__}): {error}",
            page_number=page_number,
        )

    image_bytes: bytes | None = None
    image_mime: str | None = None
    parts = response.parts or []
    for part in parts:
        if part.inline_data and part.inline_data.data:
            image_bytes = part.inline_data.data
            image_mime = (part.inline_data.mime_type or "").lower() or "image/jpeg"
            break

    if not image_bytes:
        # NOTE: empty image_part is almost always safety/recitation/MAX_TOKENS;
        # surface finish_reason / block_reason / first safety rating so blind
        # retry has a chance of being skipped at the CLI layer.
        return build_error_result(
            tool_context,
            _STEP,
            f"no image part in response ({_describe_empty_response(response)})",
            page_number=page_number,
        )
    if image_mime not in _ALLOWED_IMAGE_MIME:
        return build_error_result(
            tool_context,
            _STEP,
            f"unexpected image mime: {image_mime}",
            page_number=page_number,
        )

    extension = ".jpg" if image_mime == "image/jpeg" else ".png"
    return await save_page_artifact(
        tool_context,
        step=_STEP,
        model_label=_MODEL_LABEL,
        page_number=page_number,
        target_date=target_date,
        image_bytes=image_bytes,
        mime=image_mime,
        extension=extension,
    )
