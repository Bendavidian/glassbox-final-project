"""No tracked file contains anything shaped like a credential.

Written after a real Alpaca paper key reached a local commit on 14 Aug 2026 through a
blanket ``git add -A``. The keys were revoked and the commit was rewritten before it was
ever pushed, but a rule that depends on someone reading a diff carefully is not a rule.
This is the check that would have caught it, and it runs on every commit.

Detection is by **shape**, not by matching known strings. A test that knows the current
secrets is useless against the next one, and would have to contain them to work.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

# A vendor key prefix followed by a long uppercase-alphanumeric run. Alpaca stamps `PK`
# on paper keys and `AK` on live ones.
VENDOR_KEY = re.compile(r"\b(?:PK|AK)[A-Z0-9]{16,}\b")

# A long unbroken alphanumeric run mixing case and digits — the shape of a generated
# secret. Deliberately requires all three character classes: a 40-character git SHA is
# lowercase hex and a package hash is lowercase base16, so neither trips it.
SECRET_SHAPED = re.compile(
    r"\b(?=[A-Za-z0-9]{32,}\b)"
    r"(?=[A-Za-z0-9]*[a-z])"
    r"(?=[A-Za-z0-9]*[A-Z])"
    r"(?=[A-Za-z0-9]*[0-9])"
    r"[A-Za-z0-9]+\b"
)

PATTERNS = (("vendor key prefix", VENDOR_KEY), ("secret-shaped token", SECRET_SHAPED))

# An `.env.example` value that is one opaque alphanumeric run. Every honest placeholder
# has separators — `your-paper-api-key-here` has hyphens, an endpoint has `:` and `/` —
# so this catches a pasted credential of any vendor, not only the two prefixes above.
ENV_OPAQUE_VALUE = re.compile(r"^[A-Za-z0-9]{16,}$")


def scan(text: str) -> list[tuple[int, str, int]]:
    """Return ``(line number, pattern name, match length)`` for every hit.

    The matched text is deliberately **not** returned. A failing security test that prints
    the secret it found has moved the secret into a CI log.
    """
    hits = []
    for number, line in enumerate(text.splitlines(), start=1):
        for name, pattern in PATTERNS:
            hits.extend((number, name, len(m.group())) for m in pattern.finditer(line))
    return hits


def tracked_files(repo_root: Path) -> list[Path]:
    """Every file git is tracking. Untracked and ignored files are not our problem."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        capture_output=True,
        check=True,
        text=True,
    )
    return [repo_root / name for name in result.stdout.split("\0") if name]


def read_text(path: Path) -> str | None:
    """The file's text, or ``None`` if it is binary and cannot hold a pasted key."""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def test_no_tracked_file_contains_a_secret_shaped_string(repo_root: Path) -> None:
    """The repository-wide check. Cheap to run, and it is the one that would have caught
    the 14 Aug incident at the point of commit rather than after it."""
    offenders = []
    for path in tracked_files(repo_root):
        text = read_text(path)
        if text is None:
            continue
        offenders.extend(
            f"{path.relative_to(repo_root).as_posix()}:{number} — {name}, {length} chars"
            for number, name, length in scan(text)
        )

    assert not offenders, "credential-shaped strings in tracked files:\n" + "\n".join(
        offenders
    )


def test_env_example_holds_only_placeholders(repo_root: Path) -> None:
    """Every value in the committed template must be obviously not a credential."""
    template = repo_root / ".env.example"
    assert template.is_file()

    opaque = [
        line.split("=", 1)[0].strip()
        for line in template.read_text(encoding="utf-8").splitlines()
        if "=" in line
        and not line.strip().startswith("#")
        and ENV_OPAQUE_VALUE.match(line.split("=", 1)[1].strip().strip("\"'"))
    ]

    assert not opaque, (
        f"{opaque} in .env.example look like pasted credentials rather than "
        "placeholders. Real values belong in .env, which is git-ignored."
    )


def test_env_is_not_tracked(repo_root: Path) -> None:
    """The file that holds the real keys must never be in the index."""
    assert (repo_root / ".env.example") in tracked_files(repo_root)
    assert (repo_root / ".env") not in tracked_files(repo_root)


# ── the scan has teeth ───────────────────────────────────────────────────────
#
# The planted values below are assembled from fragments at runtime, so no
# credential-shaped literal appears in this file for the repository scan to find.


def test_the_scan_detects_a_vendor_key() -> None:
    planted = "PK" + "A1B2" * 5
    assert scan(f"ALPACA_API_KEY={planted}")


def test_the_scan_detects_an_opaque_secret() -> None:
    planted = "aB1" * 12
    assert scan(f"ALPACA_SECRET_KEY={planted}")


@pytest.mark.parametrize(
    "innocent",
    [
        "commit 787e1c3 introduced the keystone",
        "2dee07bb1031f81eb269fdac314984b973f98561",  # a full git SHA: lowercase hex
        "--hash=sha256:" + "0123456789abcdef" * 4,  # a lockfile hash: lowercase hex
        "ALPACA_API_KEY=your-paper-api-key-here",
        "https://paper-api.alpaca.markets",
        "the quick brown fox jumps over the lazy dog",
    ],
)
def test_the_scan_does_not_cry_wolf(innocent: str) -> None:
    """A check that fires on git hashes gets switched off, and then guards nothing."""
    assert not scan(innocent)
