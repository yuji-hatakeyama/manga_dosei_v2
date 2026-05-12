"""generate_page_gpt: OpenAI GPT Image を使って 1 ページ分の漫画画像を生成する。

generate_page_gemini の OpenAI 版。1 回呼ばれたら 1 回だけ
OpenAI Images Edit API を呼び、画像を artifact として保存する。
内部 retry は持たない（CLI 側で page_number を変えながら複数回呼ぶ）。
"""

import base64
import os
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

_SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "samples"
_SAMPLE_PAGE_PATH = _SAMPLES_DIR / "sample.jpg"
_CHARACTER_REF_PATH = _SAMPLES_DIR / "sanae.jpg"


_PROMPT_TEMPLATE = """## 指示

以下の「対象の台本」を元に実際の漫画のページを生成してください。

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
* 漫画のコマは「右から左へ」、そして「上から下へ」読みます。
  読みやすいコマ割りを意識してください。
* 各コマの左上には「コマタイトル」を記載します。
  台本に書かれたコマタイトルを、`#F0E68C` の背景色、黒色の太字ゴシック体で、
  そのまま正確に描画してください。
  詳細なフォーマット（フォント、サイズ、配置など）は
  添付された「ページ例」のスタイルに揃えてください（内容は台本に従ってください）。
* 「漫画のコマ割り（パネルレイアウト）のルール」を遵守してください。
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
    話者ラベルに従って、**吹き出しのしっぽを該当キャラクターへ向けてください**
    （「吹き出し配置ルール」参照）。
  * `（心の声）`、`（資料を読みながら）` などの補助注釈は、
    必要に応じて吹き出しの形（思考バルーン等）や表情・仕草で表現し、
    文字として吹き出し内に描画する必要はありません。
* **キャラクターリファレンス**:
  添付された「キャラクターリファレンス」画像を参照して、
  登場人物の顔や外見を統一してください。
  台本に登場する各キャラクターについて、対応するリファレンス画像がある場合は、
  すべてのコマで同じキャラクターデザインを使用してください。
  表情やポーズは変えても構いませんが、
  顔の特徴（目、鼻、口、髪型など）はリファレンス画像に忠実に再現してください。
  複数のキャラクターリファレンスが添付されている場合は、
  それぞれのキャラクターを適切に識別して使用してください。
* 「吹き出し（セリフ）の配置ルール」を遵守してください
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

### 漫画のコマ割り（パネルレイアウト）のルール

以下のASCIIアートは、日本スタイルの漫画における**「コマを読む順序のロジック（右から左、上から下）」を示すための抽象的な例**です。

※日本では諸外国と異なり「右から左」に読まれます。非常に間違いやすいので気をつけてください。

この特定の6コマ配置を厳密に再現する必要はありません。内容はシナリオに合わせて柔軟に変更してください。

ただし、**「読み順のルール（視線誘導）」**については、この例に従って厳密に守ってください。

## 読み順のルール参照図 (Reference for Reading Flow Logic)

※ これは読み順のルールを示すための例です (Example Only)

```
+-----------------------+
|  [End of Row] | [Start] |  <-- 1. まず右上のコマから始まる
|     Panel 2   | Panel 1 |  <-- 2. 横のラインがある限り、左へ進む
|               |         |  <-- 3. いきなり下の段（Panel 3）に行ってはいけない
+-----------------------+
|                       |
|        Panel 3        |  <-- 4. 右から左へ突き当たったら、次の段へ
|                       |
+-----------------------+
|           |           |
|  Panel 5  |  Panel 4  |  <-- 5. 同様に右(4)から左(5)へ
|-----------|           |
|  Panel 6  |           |  <-- 6. 縦に割れている場合も、横の流れが優先される
+-----------------------+
```

**制作指示**

* 上記の図はあくまで「読み順のルール」の参考です。
  実際のコマ数や配置は、シーンの内容に最適なものを生成してください。
* 日本の漫画の文法（右から左へ読む、突き当たるまで下の段に行かない）を遵守してください。

**レイアウトロジック**

* 行1: パネル1（右）、パネル2（左）
* 行2: パネル3（全幅）
* 行3: パネル4（右下縦）、パネル5（左上）、パネル6（左下）

**ルール**: 常に右から左へ流れ、その後下へ。

### 吹き出し（セリフ）の配置ルール

#### ルール1: 吹き出しの読み順（右から左）

コマ内に複数の吹き出しがある場合、**右から左の順**に読まれます。
台本のセリフ番号（1→2→3…）の順に、**右→左**へ配置してください。

※ コマの読み順と同じく「右から左」です。非常に間違いやすいので注意してください。

```
正しい配置例（1コマ内に3つの吹き出し）:
+-------------------------------+
|  [3番目のセリフ]  [2番目]  [1番目]  |  ← 読者は右から左へ読む
|    ＼          ｜      ／      |
|   キャラC   キャラB  キャラA     |
+-------------------------------+
読み順: 1番目(右) → 2番目(中央) → 3番目(左)

誤った配置例:
+-------------------------------+
|  [1番目のセリフ]  [2番目]  [3番目]  |  ← ✗ 左から右になっている
+-------------------------------+
```

#### ルール2: 吹き出しは発言者から出す

吹き出しには必ず**しっぽ（テール）**を付け、**発言しているキャラクターの口元**に向けてください。
どのキャラクターが発言しているか一目で分かるように配置します。

**発言者は台本の話者ラベルで決まります**。
台本のセリフは「`<番号>. <話者ラベル>「...本文...」`」形式で記載されており、
`<話者ラベル>` が発言者です。
例えば「`1. 小泉防衛相「自由で開かれた…」`」の発言者は小泉防衛相であり、
吹き出しのしっぽは小泉防衛相へ向けてください。
マールズ副首相のセリフを小泉防衛相に付けるような**発言者の取り違えは厳禁**です。

```
正しい例:            誤った例:
+-------------+     +-------------+
| 「セリフ」    |     | 「セリフ」    |
|   ＼        |     |        ／   |
|   キャラA    |     |   キャラA    |  ← しっぽがキャラAに
|      キャラB |     |      キャラB |    向いていない
+-------------+     +-------------+
キャラAのセリフ →       しっぽの向きが
しっぽがキャラAへ       間違っている
```

#### ルール3: 発言キャラ同士の配置も読み順に合わせる

**1コマ内で複数のキャラクターが発言する場合のみ**、発言者同士の
立ち位置を読み順（右→左、上→下）に揃えてください。台本のセリフ
番号順に、先に話す人ほど右（または上）、後に話す人ほど左（または下）
に配置します。これによりしっぽが他の吹き出しを跨いで交差すること
を防げます。

セリフを持たないキャラクターの配置は構図優先で構いません。

```
例（A→B→Cの順で発言）:
+---------------------------------+
| [3番目]  [2番目]  [1番目]       |
|   ＼       ｜       ／          |
|  キャラC  キャラB  キャラA       |
+---------------------------------+
```

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


# OpenAI Images Edit API は画像をフラットなリストでしか受け取れないため
# (Gemini のようにテキスト/画像を交互配置できない)、添付順を別途宣言する。
_IMAGE_ORDER_TEMPLATE = """

### 添付画像の順序（重要）

このリクエストには複数の参照画像を添付順に渡しています。下のリストの
順序で各画像のラベル・用途を解釈してください。

{lines}
"""


async def generate_page_gpt(
    target_date: str,
    page_number: int,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """1 ページ分の漫画画像を OpenAI GPT Image で 1 回生成し、artifact 保存する。

    前提:
        scenario.md artifact が存在すること（generate_scenario 完了想定）。
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

    images: list[tuple[str, bytes, str]] = [
        ("sample.jpg", _SAMPLE_PAGE_PATH.read_bytes(), "image/jpeg"),
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
        f"{_PROMPT_TEMPLATE}{scenario_text}\n```"
        f"{_IMAGE_ORDER_TEMPLATE.format(lines=order_lines)}"
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
