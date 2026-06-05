"""Contract tests for `manga_dosei.tools.compose_image_brief._build_prompt`.

Pins two AGENTS.md-level invariants that downstream image-gen depends on:

  1. 「高市早苗」と `assets/samples/sanae.jpg` の参照は news.md にプロフィールが
     無くても常にプロンプトに含まれる (sanae 参照画像が常に配線される契約)。
  2. ページ下部の免責文言は逐語で固定 — 後段がそのまま赤字で描画する対象なので
     一字でも揺れると整合性が崩れる。

`_build_prompt` is a pure function so it is fair game for classical-school
direct testing (no LLM, no I/O).
"""

from __future__ import annotations

from manga_dosei.tools.compose_image_brief import _build_prompt

# Verbatim disclaimer string as specified by AGENTS.md / compose_image_brief.py.
# Any drift here means the image-gen layer will render different on-page text.
_DISCLAIMER = (
    "※報道の首相動静を元にAIで創作したフィクションです。"
    "内容について一切の責任を負いません。"
)


def _build_minimal_prompt(*, news_text: str = "") -> str:
    return _build_prompt(
        target_date="20260101",
        scenario_text="",
        layout_text="- pattern_id: 4a\n",
        news_text=news_text,
        assets_manifest="{}",
    )


def test_compose_image_brief_prompt_always_injects_sanae() -> None:
    # news.md is empty — sanae must still be wired up so the downstream image-gen
    # layer always has a character reference for the Prime Minister.
    prompt = _build_minimal_prompt(news_text="")
    assert "高市早苗" in prompt
    assert "assets/samples/sanae.jpg" in prompt


def test_compose_image_brief_prompt_carries_verbatim_disclaimer() -> None:
    prompt = _build_minimal_prompt()
    assert _DISCLAIMER in prompt
