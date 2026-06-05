"""Unit tests for `manga_dosei.tools._tavily.resolve_date_range` (refactor unit U5).

Covers the pure date arithmetic extracted out of the FunctionTool closure:
offsets in both directions, `None` passthrough, month/year boundary wrap,
and YYYYMMDD validation via `validate_target_date`.
"""

from __future__ import annotations

import pytest

from manga_dosei.tools._tavily import resolve_date_range


def test_resolve_date_range_applies_both_offsets() -> None:
    start, end = resolve_date_range("20260101", 2, 1)

    assert start == "2025-12-30"
    assert end == "2026-01-02"


def test_resolve_date_range_returns_none_for_none_offsets() -> None:
    start, end = resolve_date_range("20260101", None, None)

    assert start is None
    assert end is None


def test_resolve_date_range_supports_zero_offsets() -> None:
    start, end = resolve_date_range("20260515", 0, 0)

    assert start == "2026-05-15"
    assert end == "2026-05-15"


def test_resolve_date_range_supports_one_sided_offsets() -> None:
    start_only, end_none = resolve_date_range("20260515", 7, None)
    start_none, end_only = resolve_date_range("20260515", None, 3)

    assert start_only == "2026-05-08"
    assert end_none is None
    assert start_none is None
    assert end_only == "2026-05-18"


def test_resolve_date_range_crosses_year_boundary() -> None:
    start, end = resolve_date_range("20260101", 5, 5)

    assert start == "2025-12-27"
    assert end == "2026-01-06"


def test_resolve_date_range_rejects_non_yyyymmdd() -> None:
    with pytest.raises(ValueError, match="target_date must be YYYYMMDD"):
        resolve_date_range("2026-01-01", 1, 1)


def test_resolve_date_range_rejects_short_input() -> None:
    with pytest.raises(ValueError, match="target_date must be YYYYMMDD"):
        resolve_date_range("2026011", 1, 1)


# NOTE: validate_target_date only enforces the \d{8} format; calendar validity is
# enforced by date(...) construction inside resolve_date_range (see AGENTS.md
# "Conventions specific to this repo"). Pin that split here so a future parser
# swap cannot silently start accepting invalid calendar dates.
def test_resolve_date_range_rejects_invalid_day_of_month() -> None:
    with pytest.raises(ValueError):
        resolve_date_range("20260230", 1, 1)


def test_resolve_date_range_rejects_invalid_month() -> None:
    with pytest.raises(ValueError):
        resolve_date_range("20261301", 1, 1)


# --------------------------------------------------------------------------- #
# F8: target-date-from-state resolvers must fall back to no-filter on
# malformed state values rather than propagating ValueError.
# --------------------------------------------------------------------------- #


import types as _types  # noqa: E402

from manga_dosei.tools._tavily import (  # noqa: E402
    end_date_offset_from_target,
    start_date_offset_from_target,
)


def _stub_ctx(target_date: str) -> _types.SimpleNamespace:
    return _types.SimpleNamespace(state={"temp:target_date": target_date})


def test_start_date_resolver_returns_empty_for_malformed_target_date() -> None:
    resolver = start_date_offset_from_target(days_before=2)
    assert resolver(_stub_ctx("abc")) == ""


def test_end_date_resolver_returns_empty_for_malformed_target_date() -> None:
    resolver = end_date_offset_from_target(days_after=1)
    assert resolver(_stub_ctx("not-a-date")) == ""


def test_start_date_resolver_returns_empty_for_invalid_calendar_date() -> None:
    # 20260230 passes the \d{8} format check but fails date() construction.
    # F8 says: do not let that crash the tool — return "" (no filter).
    resolver = start_date_offset_from_target(days_before=1)
    assert resolver(_stub_ctx("20260230")) == ""


def test_resolvers_still_return_iso_date_for_valid_state() -> None:
    # Sanity: the no-filter fallback must not regress the happy path.
    start_resolver = start_date_offset_from_target(days_before=2)
    end_resolver = end_date_offset_from_target(days_after=1)
    assert start_resolver(_stub_ctx("20260101")) == "2025-12-30"
    assert end_resolver(_stub_ctx("20260101")) == "2026-01-02"
