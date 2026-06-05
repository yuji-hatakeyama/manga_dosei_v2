"""Push every file under --source into a GitHub repo as one fast-forward commit.

Token is read only from GITHUB_OUTPUT_TOKEN in the process env so it never
appears in CLI args or job logs and cannot be injected by a stray `.env`
in the publisher's CWD.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path, PurePosixPath

from github import Auth, Github, GithubException, InputGitTreeElement

from manga_dosei.config import Settings


def main() -> None:
    args = _parse_args()

    # NOTE: `_env_file=None` disables pydantic-settings' `.env` lookup so a
    # stray `.env` in the CI / invocation CWD cannot inject GITHUB_OUTPUT_TOKEN
    # (AGENTS.md publisher contract). One-shot CLI, so no caching needed.
    token_secret = Settings(_env_file=None).github_output_token
    if token_secret is None:
        raise SystemExit("env var GITHUB_OUTPUT_TOKEN is required")
    token = token_secret.get_secret_value()

    source = Path(args.source).resolve()
    if not source.is_dir():
        raise SystemExit(f"--source {source} is not a directory")

    dest_prefix = _normalize_dest(args.dest)
    enumerated = enumerate_files(source)
    paths = build_dest_paths(enumerated, dest_prefix)
    if not paths:
        raise SystemExit(f"no files found under {source}")

    files = [(repo_path, local.read_bytes()) for repo_path, local in paths]

    try:
        commit_sha = push_tree(
            repo_full=args.repo,
            token=token,
            branch=args.branch,
            message=args.message,
            files=files,
        )
    except GithubException as error:
        print(
            f"GitHub API error: status={error.status} data={error.data}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    print(
        f"pushed {len(files)} file(s) → "
        f"https://github.com/{args.repo}/commit/{commit_sha}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="manga_dosei-publish",
        description=(
            "Push every regular file under --source into <repo>:<dest>/... "
            "as a single fast-forward commit on --branch. "
            "Token is read from the GITHUB_OUTPUT_TOKEN env var."
        ),
    )
    parser.add_argument("--source", required=True, help="local directory to upload")
    parser.add_argument("--repo", required=True, help="GitHub repo (owner/name)")
    parser.add_argument(
        "--dest",
        required=True,
        help="directory path inside the repo (e.g. 2026/05/20260515)",
    )
    parser.add_argument("--message", required=True, help="commit message")
    parser.add_argument(
        "--branch",
        default="main",
        help="branch to update (default: main, fast-forward only)",
    )
    return parser.parse_args()


def _normalize_dest(raw: str) -> str:
    """Validate --dest and return a clean POSIX-style prefix (no leading/trailing '/').

    Rejects values that would produce surprising tree paths: backslashes,
    leading/trailing whitespace on the whole value or any segment, empty
    segments (``a//b``), absolute paths, and ``.`` / ``..`` segments. An
    empty / ``/`` dest is allowed and means "push to repo root".
    """
    if raw != raw.strip():
        raise SystemExit(f"--dest must not have leading/trailing whitespace: {raw!r}")
    if "\\" in raw:
        raise SystemExit(f"--dest must not contain backslashes: {raw!r}")

    stripped = raw.strip("/")
    if not stripped:
        return ""

    segments = stripped.split("/")
    for segment in segments:
        if not segment:
            raise SystemExit(f"--dest must not contain empty segments: {raw!r}")
        if segment in (".", ".."):
            raise SystemExit(f"--dest must not contain '.' or '..' segments: {raw!r}")
        if segment != segment.strip():
            raise SystemExit(
                f"--dest segments must not have leading/trailing whitespace: {raw!r}"
            )

    # Sanity check via PurePosixPath: the normalized form must equal what we built,
    # otherwise something exotic (e.g. embedded NULs) slipped past the explicit checks.
    normalized = PurePosixPath(stripped)
    if normalized.is_absolute() or str(normalized) != stripped:
        raise SystemExit(f"--dest is not a clean relative path: {raw!r}")

    return stripped


def enumerate_files(source: Path) -> list[tuple[str, Path]]:
    """Walk *source* and return `(relative posix path, absolute path)` pairs.

    Sorted by the POSIX relative path so commit contents are deterministic
    across platforms (macOS APFS is case-insensitive, Windows traversal order
    differs again — sorting `Path` objects directly would leak that).

    Symlinks are not followed and dotfiles / dot-directories are skipped:
    the documented use case feeds the publish-dir straight from `run_daily.py`,
    so a stray `.env`, `.git/`, `.adk/`, or editor swap file picked up by
    redirected stdout/stderr must not end up in the public archive repo.
    """
    result: list[tuple[str, Path]] = []
    source_str = str(source)
    for dirpath, dirnames, filenames in os.walk(source_str, followlinks=False):
        # Prune dot-directories in-place so os.walk doesn't recurse into them.
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            abs_path = Path(dirpath) / name
            # Skip symlinks (incl. dangling ones) and non-regular files;
            # is_file() follows symlinks, so guard with is_symlink() first.
            if abs_path.is_symlink() or not abs_path.is_file():
                continue
            rel_posix = abs_path.relative_to(source).as_posix()
            result.append((rel_posix, abs_path))
    result.sort(key=lambda item: item[0])
    return result


def build_dest_paths(
    files: list[tuple[str, Path]],
    dest_prefix: str,
) -> list[tuple[str, Path]]:
    if not dest_prefix:
        return list(files)
    return [(f"{dest_prefix}/{rel}", local) for rel, local in files]


def push_tree(
    *,
    repo_full: str,
    token: str,
    branch: str,
    message: str,
    files: list[tuple[str, bytes]],
) -> str:
    # Git Data API (blob → tree → commit → ref.edit) avoids a clone, so upload
    # cost is constant in repo size. ref.edit refuses non-fast-forward updates,
    # so concurrent pushes fail loudly.
    auth = Auth.Token(token)
    gh = Github(auth=auth)
    try:
        repo = gh.get_repo(repo_full)
        ref = repo.get_git_ref(f"heads/{branch}")
        base_commit = repo.get_git_commit(ref.object.sha)
        base_tree = base_commit.tree

        elements: list[InputGitTreeElement] = []
        for path, data in files:
            blob = repo.create_git_blob(
                content=base64.b64encode(data).decode("ascii"),
                encoding="base64",
            )
            elements.append(
                InputGitTreeElement(
                    path=path,
                    mode="100644",
                    type="blob",
                    sha=blob.sha,
                )
            )

        new_tree = repo.create_git_tree(elements, base_tree=base_tree)
        new_commit = repo.create_git_commit(
            message=message,
            tree=new_tree,
            parents=[base_commit],
        )
        ref.edit(sha=new_commit.sha)
        return new_commit.sha
    finally:
        gh.close()


if __name__ == "__main__":
    main()
