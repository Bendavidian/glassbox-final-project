"""GB-1 acceptance: the package tree matches spec 3.4 exactly.

An empty suite makes pytest exit 5, which fails CI, so the scaffold asserts the one
thing GB-1 actually delivers: that every module named in the spec exists and that
nothing extra has been added to the tree.
"""

from pathlib import Path

import pytest

# Every module named in docs/GLASSBOX_PROJECT_SPEC.md 3.4, relative to glassbox/.
SPEC_MODULES = (
    "config/settings.yaml",
    "config/loader.py",
    "contracts/schemas.py",
    "contracts/protocols.py",
    "data/historical.py",
    "data/live.py",
    "data/quality.py",
    "features/indicators.py",
    "features/wavelets.py",
    "features/builder.py",
    "model/persistence.py",
    "model/ltsf.py",
    "model/fits.py",
    "model/train.py",
    "model/predict.py",
    "explain/channel.py",
    "explain/spectral.py",
    "explain/narrate.py",
    "engine/signal.py",
    "engine/rank.py",
    "engine/risk.py",
    "engine/executor.py",
    "backtest/engine.py",
    "backtest/walkforward.py",
    "backtest/metrics.py",
    "experiments/study.py",
    "experiments/report.py",
    "live_loop.py",
    "replay.py",
    "smoke_offline.py",
    "dashboard/app.py",
)

PACKAGE_DIRS = (
    "config",
    "contracts",
    "data",
    "features",
    "model",
    "explain",
    "engine",
    "backtest",
    "experiments",
    "dashboard",
)


def test_package_imports() -> None:
    """The package is installed and importable."""
    import glassbox

    assert glassbox.__doc__


@pytest.mark.parametrize("relative_path", SPEC_MODULES)
def test_spec_module_exists(package_root: Path, relative_path: str) -> None:
    """Every module named in spec 3.4 exists."""
    assert (package_root / relative_path).is_file()


@pytest.mark.parametrize("relative_path", PACKAGE_DIRS)
def test_package_dir_is_a_package(package_root: Path, relative_path: str) -> None:
    """Every package directory carries an __init__.py."""
    assert (package_root / relative_path / "__init__.py").is_file()


def test_every_script_guards_its_imports(repo_root: Path) -> None:
    """A script run with the wrong interpreter must say so, not raise ModuleNotFoundError.

    This bites on any fresh machine, including GB-59's clean-clone audit.
    """
    unguarded = [
        path.name
        for path in (repo_root / "scripts").glob("*.py")
        if "except ImportError" not in path.read_text(encoding="utf-8")
    ]
    assert not unguarded


def test_tree_has_no_extra_modules(package_root: Path) -> None:
    """Nothing beyond spec 3.4 has been added. New modules need a spec entry first."""
    allowed = {package_root / p for p in SPEC_MODULES}
    allowed |= {package_root / d / "__init__.py" for d in PACKAGE_DIRS}
    allowed.add(package_root / "__init__.py")

    found = {
        path
        for path in package_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert found == allowed
