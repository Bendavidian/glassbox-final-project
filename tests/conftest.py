"""Shared fixtures for the GlassBox test suite.

Paths are built with pathlib so the suite runs identically on Windows and POSIX.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root, resolved from this file rather than the working directory."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def package_root(repo_root: Path) -> Path:
    """The ``glassbox`` package directory."""
    return repo_root / "glassbox"
