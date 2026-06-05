"""Unit tests for `manga_dosei.publish._normalize_dest`.

Security-adjacent: this helper validates the `--dest` prefix that becomes part
of every tree path written to the archive repo. A bypass (``..`` segment,
absolute path, embedded backslash) would let a stray CLI invocation overwrite
files outside the intended `<year>/<month>/<date>` slot. Cover both the
"what is rejected" and "round-trip cleans correctly" sides.
"""

from __future__ import annotations

import pytest

from manga_dosei.publish import _normalize_dest


def test_normalize_dest_strips_trailing_slash_from_clean_prefix() -> None:
    assert _normalize_dest("2026/05/20260515") == "2026/05/20260515"


def test_normalize_dest_strips_leading_and_trailing_slashes() -> None:
    assert _normalize_dest("/2026/05/") == "2026/05"


def test_normalize_dest_empty_string_returns_empty_prefix() -> None:
    # Documented "push to repo root" sentinel.
    assert _normalize_dest("") == ""


def test_normalize_dest_single_slash_returns_empty_prefix() -> None:
    assert _normalize_dest("/") == ""


def test_normalize_dest_rejects_backslash_segment() -> None:
    with pytest.raises(SystemExit):
        _normalize_dest("a\\b")


def test_normalize_dest_rejects_parent_directory_segment() -> None:
    with pytest.raises(SystemExit):
        _normalize_dest("2026/../etc")


def test_normalize_dest_rejects_current_directory_segment() -> None:
    with pytest.raises(SystemExit):
        _normalize_dest("2026/./05")


def test_normalize_dest_rejects_absolute_style_with_parent_traversal() -> None:
    with pytest.raises(SystemExit):
        _normalize_dest("/../etc/passwd")


def test_normalize_dest_rejects_empty_segment_via_double_slash() -> None:
    with pytest.raises(SystemExit):
        _normalize_dest("a//b")


def test_normalize_dest_rejects_segment_with_internal_whitespace_padding() -> None:
    with pytest.raises(SystemExit):
        _normalize_dest("2026/ 05 /20260515")


def test_normalize_dest_rejects_leading_whitespace_on_whole_value() -> None:
    with pytest.raises(SystemExit):
        _normalize_dest(" 2026/05")


def test_normalize_dest_rejects_trailing_whitespace_on_whole_value() -> None:
    with pytest.raises(SystemExit):
        _normalize_dest("2026/05 ")
