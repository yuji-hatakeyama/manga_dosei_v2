"""Unit tests for the pure helpers extracted from `manga_dosei.publish`.

These exercise the file-enumeration and destination-path assembly without
hitting PyGithub or the network. Covers refactor unit U4: the tree-building
pure helpers (`enumerate_files`, `build_dest_paths`).
"""

from __future__ import annotations

from pathlib import Path

from manga_dosei.publish import build_dest_paths, enumerate_files


def test_enumerate_files_returns_sorted_relative_posix_paths(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    sub = tmp_path / "pages"
    sub.mkdir()
    (sub / "gemini_2.jpg").write_bytes(b"\x00\x01")
    (sub / "gemini_1.jpg").write_bytes(b"\x02\x03")

    result = enumerate_files(tmp_path)

    assert [rel for rel, _ in result] == [
        "a.txt",
        "b.txt",
        "pages/gemini_1.jpg",
        "pages/gemini_2.jpg",
    ]
    for rel, local in result:
        assert local.is_file()
        assert local.relative_to(tmp_path).as_posix() == rel


def test_enumerate_files_skips_directories(tmp_path: Path) -> None:
    (tmp_path / "only_dir").mkdir()
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")

    result = enumerate_files(tmp_path)

    assert [rel for rel, _ in result] == ["file.txt"]


def test_enumerate_files_empty_source_returns_empty_list(tmp_path: Path) -> None:
    # Empty source dir is not an error here — the CLI layer (`main`) is what
    # decides empty == failure; `enumerate_files` itself just returns [].
    assert enumerate_files(tmp_path) == []


def test_enumerate_files_walks_deeply_nested_directories(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    (deep / "leaf.txt").write_text("leaf", encoding="utf-8")
    (tmp_path / "a" / "b" / "mid.md").write_text("mid", encoding="utf-8")
    (tmp_path / "top.md").write_text("top", encoding="utf-8")

    result = enumerate_files(tmp_path)

    assert [rel for rel, _ in result] == [
        "a/b/c/d/leaf.txt",
        "a/b/mid.md",
        "top.md",
    ]


def test_enumerate_files_uses_posix_separators_on_all_platforms(tmp_path: Path) -> None:
    # Even on Windows, the resulting tree paths must use '/' because they are
    # repo paths sent to the GitHub Git Data API.
    nested = tmp_path / "pages"
    nested.mkdir()
    (nested / "gemini_1.jpg").write_bytes(b"")

    result = enumerate_files(tmp_path)

    assert result[0][0] == "pages/gemini_1.jpg"
    assert "\\" not in result[0][0]


def test_enumerate_files_skips_dotfiles_and_dot_directories(tmp_path: Path) -> None:
    # Dotfiles and dot-directories must be skipped so a stray `.env`, `.git/`,
    # `.adk/`, or editor swap file in the publish-dir cannot leak into the
    # public archive repo. See `enumerate_files` docstring.
    (tmp_path / ".hidden").write_text("h", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("v", encoding="utf-8")
    dot_dir = tmp_path / ".adk"
    dot_dir.mkdir()
    (dot_dir / "sessions.db").write_bytes(b"\x00")
    nested_dot = tmp_path / "pages" / ".DS_Store"
    nested_dot.parent.mkdir()
    nested_dot.write_bytes(b"")
    (nested_dot.parent / "gemini_1.jpg").write_bytes(b"")

    result = enumerate_files(tmp_path)

    assert [rel for rel, _ in result] == ["pages/gemini_1.jpg", "visible.txt"]


def test_enumerate_files_skips_symlinks(tmp_path: Path) -> None:
    # Symlinks must not be followed and must not be uploaded as their own
    # blobs. A symlink whose realpath escapes `source` would otherwise raise
    # in `relative_to(source)`, masking the actual leak.
    real = tmp_path / "real.txt"
    real.write_text("real", encoding="utf-8")

    outside = tmp_path.parent / "outside_target.txt"
    outside.write_text("outside", encoding="utf-8")
    try:
        (tmp_path / "link_inside").symlink_to(real)
        (tmp_path / "link_outside").symlink_to(outside)
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "real2.txt").write_text("r2", encoding="utf-8")
        (tmp_path / "linked_dir").symlink_to(subdir, target_is_directory=True)

        result = enumerate_files(tmp_path)

        assert [rel for rel, _ in result] == ["real.txt", "subdir/real2.txt"]
    finally:
        outside.unlink(missing_ok=True)


def test_enumerate_files_returns_absolute_local_paths_pointing_at_real_files(
    tmp_path: Path,
) -> None:
    local = tmp_path / "sub" / "x.bin"
    local.parent.mkdir()
    local.write_bytes(b"\xde\xad\xbe\xef")

    [(rel, returned_path)] = enumerate_files(tmp_path)

    assert rel == "sub/x.bin"
    assert returned_path.read_bytes() == b"\xde\xad\xbe\xef"


def test_build_dest_paths_prepends_prefix(tmp_path: Path) -> None:
    local_a = tmp_path / "a.md"
    local_a.write_text("a", encoding="utf-8")
    local_b = tmp_path / "pages" / "g_1.jpg"
    local_b.parent.mkdir()
    local_b.write_bytes(b"")

    files = [("a.md", local_a), ("pages/g_1.jpg", local_b)]

    result = build_dest_paths(files, "2026/05/20260515")

    assert result == [
        ("2026/05/20260515/a.md", local_a),
        ("2026/05/20260515/pages/g_1.jpg", local_b),
    ]


def test_build_dest_paths_joined_paths_have_single_separator(tmp_path: Path) -> None:
    # Guard against the classic "prefix ends with '/'" or "rel starts with '/'"
    # join bug — every joined path must contain exactly one '/' between the
    # prefix and the relative path.
    local = tmp_path / "a.md"
    local.write_text("a", encoding="utf-8")

    [(joined, _)] = build_dest_paths([("a.md", local)], "2026/05/20260515")

    assert "//" not in joined
    assert not joined.startswith("/")
    assert not joined.endswith("/")
    assert joined == "2026/05/20260515/a.md"


def test_build_dest_paths_empty_prefix_is_passthrough(tmp_path: Path) -> None:
    local = tmp_path / "a.md"
    local.write_text("a", encoding="utf-8")
    files = [("a.md", local)]

    assert build_dest_paths(files, "") == [("a.md", local)]


def test_build_dest_paths_returns_new_list(tmp_path: Path) -> None:
    # Caller should be free to mutate the result without affecting input.
    local = tmp_path / "a.md"
    local.write_text("a", encoding="utf-8")
    files = [("a.md", local)]

    result = build_dest_paths(files, "")

    assert result is not files


def test_build_dest_paths_empty_files_returns_empty(tmp_path: Path) -> None:
    assert build_dest_paths([], "2026/05/20260515") == []
    assert build_dest_paths([], "") == []


def test_build_dest_paths_preserves_input_order(tmp_path: Path) -> None:
    # `enumerate_files` returns sorted, but `build_dest_paths` itself must be
    # order-preserving so the final tree elements stay in deterministic order.
    files = [
        ("z.md", tmp_path / "z.md"),
        ("a.md", tmp_path / "a.md"),
        ("m.md", tmp_path / "m.md"),
    ]

    result = build_dest_paths(files, "prefix")

    assert [rel for rel, _ in result] == ["prefix/z.md", "prefix/a.md", "prefix/m.md"]
