"""Unit tests for `manga_dosei.tools._image_gen` shared helpers (refactor unit U7).

Locks the contract of the helpers shared by generate_page_gemini and
generate_page_gpt — pattern_id parsing (verbatim error message), the
assets/* fan-out skip rules, repo-relative path constants, and the
`save_page_artifact` tail-end contract — so the two backends keep matching
behavior after the dedup.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from google.genai import types

from manga_dosei.paths import CHARACTER_REF_PATH, LAYOUTS_DIR, REPO_ASSETS_DIR
from manga_dosei.tools._image_gen import (
    AssetPart,
    load_asset_parts,
    load_brief_and_layout_sample,
    parse_pattern_id,
    save_page_artifact,
)

# --------------------------------------------------------------------------- #
# parse_pattern_id
# --------------------------------------------------------------------------- #


def test_parse_pattern_id_extracts_id_from_top_line() -> None:
    brief = "- pattern_id: 4a\n\n## ページタイトル\n..."
    assert parse_pattern_id(brief) == "4a"


def test_parse_pattern_id_raises_value_error_with_verbatim_message() -> None:
    # The exact wording is part of the tool contract — callers surface it through
    # _error() and the CLI retry path; do not relax this assertion.
    with pytest.raises(ValueError) as excinfo:
        parse_pattern_id("no marker here\n")
    assert str(excinfo.value) == (
        "pattern_id not found in image_brief.md "
        "(expected `- pattern_id: <id>` near the top)"
    )


# --------------------------------------------------------------------------- #
# load_asset_parts
# --------------------------------------------------------------------------- #


def _png_part(data: bytes = b"\x89PNG\r\n\x1a\n", mime: str = "image/png") -> types.Part:
    return types.Part(inline_data=types.Blob(data=data, mime_type=mime))


async def _run_load(tool_context) -> list[AssetPart]:
    return await load_asset_parts(tool_context)


def test_load_asset_parts_returns_assets_only_in_sorted_order(stub_tool_context) -> None:
    ctx = stub_tool_context(
        initial_artifacts={
            "assets/zeta.png": _png_part(b"z"),
            "assets/alpha.jpg": _png_part(b"a", "image/jpeg"),
            # Non-assets artifacts must be ignored entirely.
            "scenario.md": types.Part(text="..."),
            "image_brief.md": types.Part(text="..."),
        }
    )
    result = asyncio.run(_run_load(ctx))
    keys = [item.key for item in result]
    assert keys == ["assets/alpha.jpg", "assets/zeta.png"]


def test_load_asset_parts_strips_extension_for_asset_name(stub_tool_context) -> None:
    ctx = stub_tool_context(
        initial_artifacts={
            "assets/takaichi.jpg": _png_part(b"x", "image/jpeg"),
            "assets/no_ext_file": _png_part(b"y"),
        }
    )
    result = asyncio.run(_run_load(ctx))
    name_by_key = {item.key: item.asset_name for item in result}
    assert name_by_key["assets/takaichi.jpg"] == "takaichi"
    # Files without extensions keep their full name.
    assert name_by_key["assets/no_ext_file"] == "no_ext_file"


def test_load_asset_parts_exposes_basename_with_extension(stub_tool_context) -> None:
    ctx = stub_tool_context(
        initial_artifacts={
            "assets/takaichi.jpg": _png_part(b"x", "image/jpeg"),
            "assets/no_ext_file": _png_part(b"y"),
        }
    )
    result = asyncio.run(_run_load(ctx))
    basename_by_key = {item.key: item.basename for item in result}
    assert basename_by_key["assets/takaichi.jpg"] == "takaichi.jpg"
    assert basename_by_key["assets/no_ext_file"] == "no_ext_file"


def test_load_asset_parts_skips_zero_byte_inline_data(stub_tool_context) -> None:
    # Empty blobs make Gemini / OpenAI requests fail; the original loop skipped
    # them and the dedup must preserve that.
    ctx = stub_tool_context(
        initial_artifacts={
            "assets/good.png": _png_part(b"\x00", "image/png"),
            "assets/empty.png": types.Part(
                inline_data=types.Blob(data=b"", mime_type="image/png")
            ),
        }
    )
    result = asyncio.run(_run_load(ctx))
    keys = [item.key for item in result]
    assert keys == ["assets/good.png"]


def test_load_asset_parts_skips_missing_inline_data(stub_tool_context) -> None:
    ctx = stub_tool_context(
        initial_artifacts={
            "assets/good.png": _png_part(b"\x01"),
            # Text-only Part (no inline_data) must be skipped.
            "assets/text.md": types.Part(text="not an image"),
        }
    )
    result = asyncio.run(_run_load(ctx))
    keys = [item.key for item in result]
    assert keys == ["assets/good.png"]


def test_load_asset_parts_returns_empty_when_no_assets(stub_tool_context) -> None:
    ctx = stub_tool_context(initial_artifacts={"scenario.md": types.Part(text="x")})
    assert asyncio.run(_run_load(ctx)) == []


def test_load_asset_parts_exposes_inline_data_bytes_on_asset_part(
    stub_tool_context,
) -> None:
    # AssetPart's `data` field is the iter-3 contract that lets backends skip
    # `part.inline_data.data` indirection — pin that it carries the original bytes.
    raw = b"\x89PNG\r\n\x1a\nORIGINAL"
    ctx = stub_tool_context(
        initial_artifacts={"assets/x.png": _png_part(raw, "image/png")}
    )
    [asset] = asyncio.run(_run_load(ctx))
    assert asset.data == raw


def test_load_asset_parts_defaults_mime_to_image_jpeg_when_missing(
    stub_tool_context,
) -> None:
    # `inline_data.mime_type` may be None / "" depending on upstream serialization;
    # the helper must default to image/jpeg so backends never send a blank mime.
    ctx = stub_tool_context(
        initial_artifacts={
            "assets/none_mime.bin": types.Part(
                inline_data=types.Blob(data=b"\x01", mime_type=None)
            ),
            "assets/blank_mime.bin": types.Part(
                inline_data=types.Blob(data=b"\x02", mime_type="")
            ),
        }
    )
    result = asyncio.run(_run_load(ctx))
    mime_by_key = {item.key: item.mime for item in result}
    assert mime_by_key["assets/none_mime.bin"] == "image/jpeg"
    assert mime_by_key["assets/blank_mime.bin"] == "image/jpeg"


def test_load_asset_parts_preserves_explicit_mime_type(stub_tool_context) -> None:
    ctx = stub_tool_context(
        initial_artifacts={
            "assets/p.png": _png_part(b"\x01", "image/png"),
            "assets/j.jpg": _png_part(b"\x02", "image/jpeg"),
            "assets/w.webp": _png_part(b"\x03", "image/webp"),
        }
    )
    result = asyncio.run(_run_load(ctx))
    mime_by_key = {item.key: item.mime for item in result}
    assert mime_by_key == {
        "assets/p.png": "image/png",
        "assets/j.jpg": "image/jpeg",
        "assets/w.webp": "image/webp",
    }


# --------------------------------------------------------------------------- #
# Repo-relative path constants
# --------------------------------------------------------------------------- #


def test_repo_assets_dir_resolves_to_repo_root_assets() -> None:
    # The constant lives in `manga_dosei/paths.py` and points at the repo
    # `assets/` directory.
    expected = Path(__file__).resolve().parent.parent.parent / "assets"
    assert REPO_ASSETS_DIR == expected


def test_layouts_dir_is_repo_assets_layouts() -> None:
    assert LAYOUTS_DIR == REPO_ASSETS_DIR / "layouts"


def test_character_ref_path_is_repo_assets_samples_sanae_jpg() -> None:
    assert CHARACTER_REF_PATH == REPO_ASSETS_DIR / "samples" / "sanae.jpg"


# --------------------------------------------------------------------------- #
# load_brief_and_layout_sample — CR1 / CR2 regression
# --------------------------------------------------------------------------- #


def test_load_brief_invalid_target_date_preserves_upstream_last_error(
    stub_tool_context,
) -> None:
    # CR1 regression: malformed target_date is wiring (not a step failure)
    # — must NOT touch state["last_error"] (would clobber upstream context).
    stale = {"step": "upstream_step", "message": "real failure"}
    ctx = stub_tool_context(initial_state={"last_error": stale})

    result = asyncio.run(
        load_brief_and_layout_sample(
            ctx,
            step="generate_page_gemini",
            target_date="bad-date",
            page_number=1,
        )
    )

    assert isinstance(result, dict)
    assert result["status"] == "error"
    assert result["step"] == "generate_page_gemini"
    assert result["page_number"] == 1
    # Upstream last_error survives untouched.
    assert ctx.state["last_error"] == stale


def test_load_brief_returns_error_when_character_ref_image_missing(
    stub_tool_context, monkeypatch, tmp_path
) -> None:
    # CR2 regression: helpers must read CHARACTER_REF_PATH via attribute access
    # on `paths` so monkeypatching the module attribute takes effect. The
    # filesystem path is a legitimate I/O boundary — monkeypatch is the right
    # tool here.
    from manga_dosei import paths as paths_module

    missing = tmp_path / "nonexistent_sanae.jpg"
    monkeypatch.setattr(paths_module, "CHARACTER_REF_PATH", missing)

    ctx = stub_tool_context()
    result = asyncio.run(
        load_brief_and_layout_sample(
            ctx,
            step="generate_page_gemini",
            target_date="20260101",
            page_number=2,
        )
    )

    assert isinstance(result, dict)
    assert result["status"] == "error"
    assert "character reference image missing" in result["message"]
    assert str(missing) in result["message"]


# --------------------------------------------------------------------------- #
# save_page_artifact
# --------------------------------------------------------------------------- #


def test_save_page_artifact_writes_canonical_pages_path(stub_tool_context) -> None:
    ctx = stub_tool_context()
    payload = asyncio.run(
        save_page_artifact(
            ctx,
            step="generate_page_gemini",
            model_label="gemini",
            page_number=3,
            target_date="20260101",
            image_bytes=b"\x89PNG\r\n\x1a\n...",
            mime="image/jpeg",
            extension=".jpg",
        )
    )
    assert payload["artifact"] == "pages/gemini_3.jpg"
    # Artifact actually landed in the in-memory store under the canonical key.
    assert "pages/gemini_3.jpg" in ctx._artifact_store


def test_save_page_artifact_clears_last_error_and_sets_target_date(
    stub_tool_context,
) -> None:
    ctx = stub_tool_context(
        initial_state={"last_error": {"step": "prev", "message": "stale"}},
    )
    asyncio.run(
        save_page_artifact(
            ctx,
            step="generate_page_gpt",
            model_label="gpt",
            page_number=1,
            target_date="20260515",
            image_bytes=b"\x00\x01",
            mime="image/png",
            extension=".png",
        )
    )
    assert ctx.state["last_error"] is None
    assert ctx.state["target_date"] == "20260515"


def test_save_page_artifact_returns_payload_with_required_keys(
    stub_tool_context,
) -> None:
    ctx = stub_tool_context()
    image_bytes = b"\xff\xd8\xff" + b"x" * 100  # JPEG-ish 103-byte body
    payload = asyncio.run(
        save_page_artifact(
            ctx,
            step="generate_page_gemini",
            model_label="gemini",
            page_number=2,
            target_date="20260101",
            image_bytes=image_bytes,
            mime="image/jpeg",
            extension=".jpg",
        )
    )
    assert payload["status"] == "success"
    assert payload["step"] == "generate_page_gemini"
    assert payload["artifact"] == "pages/gemini_2.jpg"
    assert payload["version"] == 0
    assert payload["page_number"] == 2
    assert payload["bytes"] == len(image_bytes)
    assert payload["mime_type"] == "image/jpeg"
