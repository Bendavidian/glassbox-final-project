"""GB-1 acceptance: the package tree matches spec 3.4 exactly.

An empty suite makes pytest exit 5, which fails CI, so the scaffold asserts the one
thing GB-1 actually delivers: that every module named in the spec exists and that
nothing extra has been added to the tree.
"""

import re
from pathlib import Path

import pytest

# Every module named in docs/GLASSBOX_PROJECT_SPEC.md 3.4, relative to glassbox/.
SPEC_MODULES = (
    "config/settings.yaml",
    "config/loader.py",
    "contracts/schemas.py",
    "contracts/protocols.py",
    "data/historical.py",
    "data/http.py",
    "data/live.py",
    "data/quality.py",
    "features/indicators.py",
    "features/wavelets.py",
    "features/builder.py",
    "model/persistence.py",
    "model/ltsf.py",
    "model/fits.py",
    "model/wits.py",
    "model/train.py",
    # GB-15: the per-epoch loss record. A module of its own because `ltsf.py` produces it
    # and `train.py` writes it, and `train.py` already imports `ltsf` through the package
    # `__init__` — putting the type in either would make the pair circular.
    "model/history.py",
    "model/predict.py",
    "explain/channel.py",
    "explain/spectral.py",
    "explain/narrate.py",
    "engine/signal.py",
    "engine/rank.py",
    "engine/risk.py",
    "engine/executor.py",
    # GB-23: reconcile-from-truth. Its own module rather than part of `executor.py`
    # because submission and reconciliation fail differently and are read separately —
    # one is "did the order go", the other is "is what we believe still true".
    "engine/reconcile.py",
    "backtest/engine.py",
    "backtest/walkforward.py",
    "backtest/metrics.py",
    # GB-20: per-fold threshold calibration. At the harness layer rather than inside
    # `engine/signal.py` because the ruling is that it scores candidates with the real
    # backtester, and the engine layer may not import the harness - nor may the live path,
    # which imports `signal`.
    "backtest/calibrate.py",
    "experiments/study.py",
    # GB-51: the paired significance tests. Its own module rather than part of
    # `study.py` because it is a pure function from a results table to a test table -
    # it runs nothing, trains nothing and needs no cache - and because `report.py`
    # must be able to derive it from `results.csv` alone.
    "experiments/stats.py",
    "experiments/report.py",
    # GB-39: the retry policy. Its own module because its callers sit at opposite ends of
    # the layer stack - `data/live.py` is L1 and `engine/executor.py` is L5 - so neither
    # can import it from the other without inverting the stack, and a backoff written
    # twice is a backoff that drifts.
    "faults.py",
    # GB-29: decision records and the live trade log. Top-level beside `live_loop.py`
    # because the live loop and the dashboard both read it and neither may reach the
    # harness - it is named in the forbidden-import contract for that reason.
    "records.py",
    "live_loop.py",
    # GB-26 / 26 Aug 2026: the one-loop-per-state-directory lock. Its own module
    # because it must be importable and callable before `live_loop` has loaded a
    # config, and because it imports nothing from `glassbox` in return.
    "live_lock.py",
    "replay.py",
    "smoke_offline.py",
    "dashboard/app.py",
    # GB-63c: the palette, split out of `app.py`. Its own module because the contrast test
    # has to walk every colour without knowing their names - a palette scattered through a
    # 2,400-line view module can only be checked by a test that lists what it expects to
    # find, and that list and the palette then drift apart while both stay internally
    # consistent.
    "dashboard/tokens.py",
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


def test_every_declared_dependency_is_pinned_in_the_lock(repo_root: Path) -> None:
    """Two lists that must agree, and nothing made them agree until CI went red.

    ``pyproject.toml`` **declares** dependencies; CI **installs** from
    ``requirements.lock`` and then adds the package with ``--no-deps``. So a dependency
    added to ``pyproject.toml`` alone is present on the developer's machine and absent in
    CI — which is exactly how matplotlib produced a green local suite and a red run #17 on
    20 Aug 2026, and it is the same defect class as ``smoke_offline``'s third copy of the
    model registry: a fact written in two places with no mechanism keeping them equal.

    What is asserted is the **declared** set, not the transitive closure: the lock holds
    the closure and ``pyproject.toml`` deliberately does not, so requiring equality in
    both directions would fail on every indirect pin. Every name the project asks for by
    name must be pinned somewhere in the lock.
    """
    import tomllib

    manifest = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    declared = list(manifest["project"]["dependencies"])
    for extra in manifest["project"].get("optional-dependencies", {}).values():
        declared.extend(extra)

    locked = {
        _normalised(line.split("==")[0])
        for line in (repo_root / "requirements.lock")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.startswith("#")
    }

    missing = sorted(
        {
            _normalised(re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0])
            for requirement in declared
        }
        - locked
    )

    assert not missing, (
        f"declared in pyproject.toml and not pinned in requirements.lock: {missing}. "
        "CI installs from the lock, so these would be missing there and present here"
    )


def _normalised(name: str) -> str:
    """PEP 503 normalisation, so ``python_dotenv`` and ``python-dotenv`` compare equal."""
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def test_the_data_snapshot_is_tracked_so_gated_tests_run_in_ci(
    repo_root: Path,
) -> None:
    """A mechanism that can be skipped by a flag is a mechanism only when the flag is off.

    Roughly fourteen tests in this suite are gated on ``data_cache/`` — the train/live
    parity sweep, the offline smoke path, the whole study grid — and while the snapshot
    was git-ignored every one of them **skipped in CI**. They ran on one laptop and
    nowhere else, which is how a defect reached ``results.csv`` past a test that had
    already been written to catch it (20 Aug 2026, DECISIONS).

    Tracking the snapshot is what makes them run, and this test is what stops it being
    quietly un-tracked again. It also makes ``data_snapshot_last_bar`` a reference rather
    than a name: §7 requires every result to say which snapshot produced it.
    """
    import subprocess

    from glassbox.config.loader import load_config

    cfg = load_config()
    listed = subprocess.run(
        ["git", "ls-files", "-z", cfg.data.cache_dir],
        cwd=repo_root,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.split("\0")
    tracked = {Path(name).name for name in listed if name}

    missing = sorted(
        f"{symbol}.parquet"
        for symbol in cfg.universe
        if f"{symbol}.parquet" not in tracked
    )

    assert not missing, (
        f"the snapshot for {missing} is not tracked, so every cache-gated test skips in "
        "CI. See CLAUDE.md §3 and the .gitignore comment"
    )
