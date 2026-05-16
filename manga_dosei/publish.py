"""Push every file under --source into a GitHub repo as one fast-forward commit.

Token is read only from GITHUB_OUTPUT_TOKEN env so it never appears in CLI args
or job logs.
"""

import argparse
import base64
import os
import sys
from pathlib import Path

from github import Auth, Github, GithubException, InputGitTreeElement


def main() -> None:
    args = _parse_args()
    token = os.environ.get("GITHUB_OUTPUT_TOKEN")
    if not token:
        raise SystemExit("env var GITHUB_OUTPUT_TOKEN is required")

    source = Path(args.source).resolve()
    if not source.is_dir():
        raise SystemExit(f"--source {source} is not a directory")

    files = _collect_files(source, args.dest.strip("/"))
    if not files:
        raise SystemExit(f"no files found under {source}")

    try:
        commit_sha = _push_to_github(
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


def _collect_files(source: Path, dest_prefix: str) -> list[tuple[str, bytes]]:
    files: list[tuple[str, bytes]] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        repo_path = f"{dest_prefix}/{relative}" if dest_prefix else relative
        files.append((repo_path, path.read_bytes()))
    return files


def _push_to_github(
    *,
    repo_full: str,
    token: str,
    branch: str,
    message: str,
    files: list[tuple[str, bytes]],
) -> str:
    # Uses the Git Data API (blob → tree → commit → ref.edit) instead of cloning
    # so the upload cost is constant regardless of repo size. ref.edit refuses
    # non-fast-forward updates, so concurrent pushes will fail loudly.
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
