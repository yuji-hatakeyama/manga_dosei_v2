"""Unit tests for `manga_dosei.names` (refactor unit U2).

Pins the enum values to the literal strings the existing codebase uses for
artifact filenames and session state keys. These tests guard later refactor
units that swap string literals for these enums — any rename has to be done
deliberately and updates these tests in the same commit.
"""

from __future__ import annotations

from manga_dosei.names import (
    TEMP_TARGET_DATE,
    ArtifactName,
    StateKey,
    temp_key,
)


def test_artifact_name_values_match_canonical_filenames() -> None:
    assert ArtifactName.DOSEI == "dosei.md"
    assert ArtifactName.NEWS == "news.md"
    assert ArtifactName.SCENARIO == "scenario.md"
    assert ArtifactName.LAYOUT == "layout.md"
    assert ArtifactName.IMAGE_BRIEF == "image_brief.md"
    assert ArtifactName.ASSETS_MANIFEST == "manifests/assets.json"


def test_state_key_values_match_run_daily_literals() -> None:
    # run_daily.py writes these exact keys into session.state (see
    # `_ensure_session` initial state and `_record_error`'s state_delta).
    assert StateKey.TARGET_DATE == "target_date"
    assert StateKey.STATUS == "status"
    assert StateKey.LAST_ERROR == "last_error"
    assert StateKey.ASSET_MANIFEST_ARTIFACT == "asset_manifest_artifact"
    assert StateKey.ASSET_COUNT == "asset_count"


def test_temp_key_prefixes_arbitrary_string_with_temp_namespace() -> None:
    assert temp_key("dosei_text") == "temp:dosei_text"
    assert temp_key("foo") == "temp:foo"


def test_temp_key_accepts_state_key_enum_member() -> None:
    # StrEnum members format as their value, so `temp_key(StateKey.TARGET_DATE)`
    # produces the same string a plain "target_date" would.
    assert temp_key(StateKey.TARGET_DATE) == "temp:target_date"


def test_temp_target_date_constant_matches_existing_scratch_key() -> None:
    assert TEMP_TARGET_DATE == "temp:target_date"
