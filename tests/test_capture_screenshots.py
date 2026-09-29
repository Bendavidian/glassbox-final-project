"""The capture script's refusals, tested without a browser.

``scripts/capture_screenshots.py`` exists because six committed figures had no producing
code a clone could run (PROGRESS.md, 28 Sep 2026). What makes it safe to run is what it
refuses: to overwrite, to guess which decision to expand, and to photograph a panel whose
emptiness would read as a broker failure. Those are pure functions and are tested here, in
CI, without Playwright - a refusal that only runs on the capture machine is a note.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from glassbox.dashboard import app


@pytest.fixture(scope="module")
def script(repo_root: Path) -> ModuleType:
    """``scripts/capture_screenshots.py``, loaded by path."""
    path = repo_root / "scripts" / "capture_screenshots.py"
    spec = importlib.util.spec_from_file_location("capture_screenshots", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs, because `dataclasses` resolves the string annotations
    # `from __future__ import annotations` produces through `sys.modules`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _state(script: ModuleType, **overrides) -> object:
    healthy = {
        "session": "LIVE",
        "showing": script.BROKER,
        "positions_empty": False,
        "activity_empty": False,
        "readings": 4,
        "decision_matches": 1,
    }
    return script.PageState(**(healthy | overrides))


def test_no_shot_can_write_a_retained_frame(script: ModuleType) -> None:
    written = {shot.filename for shot in script.SHOTS.values()}
    assert not written & script.RETAINED
    assert script.RETAINED == {
        "01-overview.png",
        "03-forecast-path.png",
        "06-reflow-1280.png",
    }


def test_out_has_no_default(script: ModuleType) -> None:
    with pytest.raises(SystemExit):
        script.main(["http://localhost:8540", "--shot", "02"])


def test_an_existing_target_is_refused(script: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "04-attribution.png").write_bytes(b"")
    with pytest.raises(script.Refused, match="overwrite"):
        script.targets(tmp_path, ["03b", "04"])
    assert set(script.targets(tmp_path, ["03b"])) == {"03b"}


def test_a_decision_id_is_the_expander_title_prefix(script: ModuleType) -> None:
    assert script.expander_prefix("20260922-NVDA") == "2026-09-22  NVDA  "
    with pytest.raises(script.Refused):
        script.expander_prefix("NVDA")


def test_the_prefix_matches_the_dashboards_own_title(script: ModuleType) -> None:
    """Two writings of one format: the script's parse and the dashboard's title."""
    source = Path(app.__file__).read_text(encoding="utf-8")
    assert 'f"{record.as_of:%Y-%m-%d}  {record.symbol}  "' in source


def test_a_healthy_page_is_not_refused(script: ModuleType) -> None:
    keys = ["07", "02", "03b", "04", "06b"]
    assert script.refusals(keys, _state(script), "20260922-NVDA") == []


@pytest.mark.parametrize("session", [app.NOT_RUNNING, app.NOT_RESPONDING])
def test_the_activity_frame_refuses_a_dead_loop(
    script: ModuleType, session: str
) -> None:
    assert script.refusals(["02"], _state(script, session=session), None)


def test_the_activity_frame_refuses_an_empty_positions_panel(
    script: ModuleType,
) -> None:
    """GB-55 §55.11: a failed broker read renders exactly like a flat book."""
    reasons = script.refusals(["02"], _state(script, positions_empty=True), None)
    assert any("broker read failure" in reason for reason in reasons)


def test_the_first_expander_is_never_the_choice(script: ModuleType) -> None:
    assert script.refusals(["04"], _state(script), None)
    assert script.refusals(["04"], _state(script, decision_matches=0), "20260922-NVDA")
    assert script.refusals(["04"], _state(script, decision_matches=2), "20260922-NVDA")


@pytest.mark.parametrize("readings", [None, 1, app.EQUITY_THIN_READINGS, 40])
def test_the_young_session_frame_refuses_outside_its_window(
    script: ModuleType, readings: int | None
) -> None:
    assert script.refusals(["07"], _state(script, readings=readings), None)


def test_a_live_frame_refuses_a_replay_page_and_back(script: ModuleType) -> None:
    replay = _state(script, showing="REPLAY:FOLD-13")
    assert script.refusals(["03b"], replay, None)
    assert script.refusals(["05"], replay, "20250701-AAPL") == []
    assert script.refusals(["05"], _state(script), "20250701-AAPL")


def test_every_anchor_is_something_the_dashboard_renders(script: ModuleType) -> None:
    """A renamed region fails here rather than producing a frame of the wrong thing."""
    source = Path(app.__file__).read_text(encoding="utf-8")
    for label in (script.POSITIONS, script.ACTIVITY, script.SESSION_EQUITY):
        assert f'"{label}",' in source
    for text in (script.FORECASTS, script.PER_CHANNEL, script.PER_FREQUENCY):
        assert text in source
    assert "READINGS THIS SESSION" in source
