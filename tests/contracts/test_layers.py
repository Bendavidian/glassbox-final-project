"""GB-3 acceptance: the layer rule of spec 3.1 is machine-enforced, not documented.

The contracts live in pyproject.toml; this runs them, so a violation fails the suite
rather than waiting to be noticed in review.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from importlinter import configuration
from importlinter.application import use_cases

# import-linter's CLI does this before any use case; calling the API directly means
# doing it ourselves, or the option readers are unregistered.
configuration.configure()

CONTRACT_NAMES = (
    "Layers: data flows strictly upward",
    "The live path never imports the validation harness",
)


def _run_contracts(repo_root: Path) -> bool:
    """Run every import-linter contract. Caching is off so a stale graph cannot pass."""
    return use_cases.lint_imports(
        config_filename=str(repo_root / "pyproject.toml"),
        cache_dir=None,
    )


def test_no_layer_violations(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No module imports from a layer above it, and the live path ignores the harness."""
    monkeypatch.chdir(repo_root)
    assert _run_contracts(
        repo_root
    ), "import-linter reported a layer violation; see the report printed above"


@pytest.mark.parametrize("name", CONTRACT_NAMES)
def test_contract_is_configured(repo_root: Path, name: str) -> None:
    """Both contracts are actually registered, so the suite cannot pass vacuously."""
    options = use_cases.read_user_options(
        config_filename=str(repo_root / "pyproject.toml")
    )
    configured = [contract["name"] for contract in options.contracts_options]
    assert name in configured


def test_root_package_is_glassbox(repo_root: Path) -> None:
    """reference/ is outside the contract by construction, not by an exclusion list."""
    options = use_cases.read_user_options(
        config_filename=str(repo_root / "pyproject.toml")
    )
    assert options.session_options["root_packages"] == ["glassbox"]
