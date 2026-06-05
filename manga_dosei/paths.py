"""Repo-relative path constants shared across tools.

Lives outside `tools/` so non-image-gen modules (e.g. `define_layout`) can
import the layout catalog root without depending on `_image_gen.py`.
"""

from __future__ import annotations

from pathlib import Path

REPO_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
LAYOUTS_DIR = REPO_ASSETS_DIR / "layouts"
CHARACTER_REF_PATH = REPO_ASSETS_DIR / "samples" / "sanae.jpg"


__all__: list[str] = [
    "CHARACTER_REF_PATH",
    "LAYOUTS_DIR",
    "REPO_ASSETS_DIR",
]
