"""Unit tests for the lazy layout-catalog loader in `manga_dosei.tools.define_layout`
(refactor unit U9).

The loader must:
  1. Not touch the filesystem at module import — a missing or broken
     `assets/layouts/` directory must not break `import define_layout`
     (the LlmAgent at `adk web` startup pulls this module transitively).
  2. Memoise the catalog so repeated calls share one dict object.
  3. Return content that matches what is actually on disk under
     `assets/layouts/`.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from manga_dosei import paths
from manga_dosei.tools import define_layout


@pytest.fixture
def fresh_define_layout(monkeypatch: pytest.MonkeyPatch):
    """Reimport `define_layout` so `get_layout_catalog`'s lru_cache starts clean.

    Tests that swap `LAYOUTS_DIR` need a cache that has not seen the real
    catalog yet, otherwise the cached real entries leak across tests.
    """
    sys.modules.pop("manga_dosei.tools.define_layout", None)
    module = importlib.import_module("manga_dosei.tools.define_layout")
    yield module
    # Restore the canonical module so later tests share state with the rest
    # of the suite.
    sys.modules.pop("manga_dosei.tools.define_layout", None)
    importlib.import_module("manga_dosei.tools.define_layout")


# --------------------------------------------------------------------------- #
# Import-time side effects
# --------------------------------------------------------------------------- #


def test_import_does_not_touch_filesystem_when_catalog_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Point LAYOUTS_DIR at a non-existent path and re-import the module.
    # Import must succeed; only an explicit `get_layout_catalog()` call would
    # surface the missing-catalog error.
    missing = tmp_path / "does-not-exist"
    monkeypatch.setattr(paths, "LAYOUTS_DIR", missing)
    sys.modules.pop("manga_dosei.tools.define_layout", None)
    try:
        module = importlib.import_module("manga_dosei.tools.define_layout")
        assert module.define_layout_tool is not None
        # lru_cache must be empty right after import — disk has not been read.
        assert module.get_layout_catalog.cache_info().currsize == 0
    finally:
        sys.modules.pop("manga_dosei.tools.define_layout", None)
        importlib.import_module("manga_dosei.tools.define_layout")


def test_get_layout_catalog_raises_when_directory_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fresh_define_layout
) -> None:
    # Catalog dir absent → the lazy loader must raise (not silently return
    # an empty catalog) so `_before` can surface it via last_error.
    missing = tmp_path / "does-not-exist"
    monkeypatch.setattr(paths, "LAYOUTS_DIR", missing)
    with pytest.raises(FileNotFoundError):
        fresh_define_layout.get_layout_catalog()


# --------------------------------------------------------------------------- #
# Memoisation
# --------------------------------------------------------------------------- #


def test_get_layout_catalog_is_cached_across_calls() -> None:
    define_layout.get_layout_catalog.cache_clear()
    first = define_layout.get_layout_catalog()
    second = define_layout.get_layout_catalog()
    assert first is second


# --------------------------------------------------------------------------- #
# Content matches disk
# --------------------------------------------------------------------------- #


def _read_disk_catalog() -> dict[str, dict]:
    """Independently re-derive the expected catalog from `assets/layouts/`."""
    out: dict[str, dict] = {}
    for entry in sorted(paths.LAYOUTS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        meta = json.loads((entry / "meta.json").read_text(encoding="utf-8"))
        ascii_text = (entry / "ascii.txt").read_text(encoding="utf-8").rstrip("\n")
        out[meta["id"]] = {
            "id": meta["id"],
            "panels": meta["panels"],
            "name": meta["name"],
            "when_to_use": meta["when_to_use"],
            "rows": meta["rows"],
            "ascii": ascii_text,
        }
    return out


def test_catalog_ids_match_directories_on_disk() -> None:
    define_layout.get_layout_catalog.cache_clear()
    expected_ids = {entry.name for entry in paths.LAYOUTS_DIR.iterdir() if entry.is_dir()}
    assert set(define_layout.get_layout_catalog().keys()) == expected_ids


def test_catalog_content_matches_files_on_disk() -> None:
    define_layout.get_layout_catalog.cache_clear()
    assert define_layout.get_layout_catalog() == _read_disk_catalog()


def test_catalog_entries_carry_required_fields() -> None:
    # Downstream code (`_format_catalog_for_prompt`) indexes these keys
    # directly — locking them keeps that contract explicit.
    define_layout.get_layout_catalog.cache_clear()
    required_fields = {"id", "panels", "name", "when_to_use", "rows", "ascii"}
    for pattern_id, entry in define_layout.get_layout_catalog().items():
        assert required_fields <= set(entry.keys()), pattern_id
        assert isinstance(entry["rows"], list) and entry["rows"]
        assert isinstance(entry["panels"], int)
        assert entry["ascii"]  # non-empty ASCII figure


# --------------------------------------------------------------------------- #
# Loader uses LAYOUTS_DIR (sanity check on monkeypatch path)
# --------------------------------------------------------------------------- #


def test_catalog_picks_up_tmp_catalog_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fresh_define_layout
) -> None:
    # Build a minimal one-pattern catalog in tmp and verify the loader reads
    # from there when LAYOUTS_DIR is swapped — proves the path is the only
    # I/O coupling.
    fake_root = tmp_path / "layouts"
    pattern_dir = fake_root / "9z"
    pattern_dir.mkdir(parents=True)
    (pattern_dir / "meta.json").write_text(
        json.dumps(
            {
                "id": "9z",
                "panels": 1,
                "name": "test-only",
                "when_to_use": "unit test",
                "rows": ["段 1: 全幅 = ①"],
            }
        ),
        encoding="utf-8",
    )
    (pattern_dir / "ascii.txt").write_text("+---+\n| ① |\n+---+\n", encoding="utf-8")

    monkeypatch.setattr(paths, "LAYOUTS_DIR", fake_root)

    catalog = fresh_define_layout.get_layout_catalog()
    assert set(catalog.keys()) == {"9z"}
    assert catalog["9z"]["ascii"] == "+---+\n| ① |\n+---+"
    assert catalog["9z"]["rows"] == ["段 1: 全幅 = ①"]


def test_catalog_raises_when_pattern_dir_is_incomplete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fresh_define_layout
) -> None:
    # Drop a pattern dir with only meta.json — loader must raise (not silent
    # skip) so an image-gen "pattern_id unknown" debug session isn't required
    # later. This matches the explicit guard in `_load_catalog`.
    fake_root = tmp_path / "layouts"
    pattern_dir = fake_root / "bad"
    pattern_dir.mkdir(parents=True)
    (pattern_dir / "meta.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(paths, "LAYOUTS_DIR", fake_root)

    with pytest.raises(FileNotFoundError):
        fresh_define_layout.get_layout_catalog()
