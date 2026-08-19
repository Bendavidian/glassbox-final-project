"""Report the CI result for a commit, so nobody has to assume it.

Three consecutive reports were written on an assumed-green CI because ``gh`` is not
authenticated here and the repository is private, so the unauthenticated API returns 404.
The credential that pushes the commit is the same one that can read the run, and git will
hand it over on request — so there is no reason to guess.

Usage::

    python scripts/ci_status.py            # the current HEAD
    python scripts/ci_status.py <sha>      # a specific commit
    python scripts/ci_status.py --recent 5 # the last few runs, whatever the commit

Exit codes: 0 the run succeeded, 1 it failed or was cancelled, 2 no run was found for that
commit yet, 3 the credential or the API was unavailable. So it is usable in a shell
condition rather than only by reading it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request

try:
    from glassbox.config.loader import DEFAULT_SETTINGS_PATH  # noqa: F401
except ImportError:  # pragma: no cover - the wrong interpreter
    print(
        "ci_status: run this with the project's interpreter, e.g.\n"
        "  .venv/Scripts/python.exe scripts/ci_status.py",
        file=sys.stderr,
    )
    raise SystemExit(3) from None

API = "https://api.github.com"
CONCLUSION_EXIT = {"success": 0, None: 2}


def _repo_slug() -> str:
    """``owner/name`` from the origin remote, however it is spelled."""
    url = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    slug = url.removesuffix(".git")
    if slug.startswith("git@"):
        slug = slug.split(":", 1)[1]
    else:
        slug = "/".join(slug.split("/")[-2:])
    return slug


def _token() -> str:
    """The credential git already holds for github.com. Never printed."""
    filled = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for line in filled.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    raise SystemExit(
        "ci_status: git holds no credential for github.com. Push once, or run "
        "`gh auth login`, then try again."
    )


def _api(path: str, token: str) -> dict:
    request = urllib.request.Request(
        API + path,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "glassbox-ci-status",
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def _describe(run: dict) -> str:
    return (
        f"{run['head_sha'][:7]}  {run['status']:<11} {run['conclusion']!s:<10} "
        f"{run['name']}  #{run['run_number']}  {run['created_at']}\n"
        f"          {run['html_url']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/ci_status.py")
    parser.add_argument("sha", nargs="?", default=None, help="commit (default: HEAD)")
    parser.add_argument("--recent", type=int, default=0, metavar="N")
    args = parser.parse_args(argv)

    try:
        token = _token()
        slug = _repo_slug()
    except subprocess.CalledProcessError as failure:
        print(f"ci_status: git failed: {failure}", file=sys.stderr)
        return 3

    try:
        runs = _api(f"/repos/{slug}/actions/runs?per_page=30", token)["workflow_runs"]
    except (urllib.error.URLError, KeyError) as failure:
        print(f"ci_status: the API did not answer: {failure}", file=sys.stderr)
        return 3

    if args.recent:
        for run in runs[: args.recent]:
            print(_describe(run))
        return 0

    sha = (
        args.sha
        or subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )

    for run in runs:
        if run["head_sha"].startswith(sha) or sha.startswith(run["head_sha"][:7]):
            print(_describe(run))
            if run["status"] != "completed":
                print("  still running")
                return 2
            return CONCLUSION_EXIT.get(run["conclusion"], 1)

    print(f"no CI run found for {sha[:7]} yet (checked the last {len(runs)})")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
