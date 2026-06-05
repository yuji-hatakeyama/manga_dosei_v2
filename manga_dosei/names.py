"""Stable artifact filenames and session state keys.

Single source of truth so renames are one-line changes; see AGENTS.md
"Artifact names are stable contracts between steps".
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ArtifactName(StrEnum):
    DOSEI = "dosei.md"
    NEWS = "news.md"
    SCENARIO = "scenario.md"
    LAYOUT = "layout.md"
    IMAGE_BRIEF = "image_brief.md"
    ASSETS_MANIFEST = "manifests/assets.json"


class StateKey(StrEnum):
    TARGET_DATE = "target_date"
    STATUS = "status"
    LAST_ERROR = "last_error"
    ASSET_MANIFEST_ARTIFACT = "asset_manifest_artifact"
    ASSET_COUNT = "asset_count"


TEMP_PREFIX = "temp:"


def temp_key(name: StateKey | str) -> str:
    return f"{TEMP_PREFIX}{name}"


TEMP_TARGET_DATE = temp_key(StateKey.TARGET_DATE)


def persistent_state(state_mapping: dict[str, Any]) -> dict[str, Any]:
    """Return `state_mapping` minus `temp:`-prefixed entries.

    `temp:` keys are ADK-ephemeral; surfacing them to readers as
    "persistent state" misleads downstream consumers (LLM, inspectors).
    """
    return {k: v for k, v in state_mapping.items() if not k.startswith(TEMP_PREFIX)}
