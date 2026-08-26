"""One loop per state directory.

The hazard is concrete: on 25 Aug 2026 a stale terminal relaunched a rehearsal against the
deployed session's state directory. Nothing was harmed, and nothing *prevented* the harm -
the bar the rehearsal would have decided had already been decided, so it found nothing to
do. A new bar completes every session, and that accident does not repeat.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from glassbox.live_lock import (
    LOCK_FILE,
    LockRefused,
    acquire,
    adopt,
    lock_path,
    process_is_alive,
    read_lock,
    release,
)


@pytest.fixture
def dead_pid() -> int:
    """A PID that certainly is not running, because we watched it exit.

    Not a made-up large number: PIDs are recycled, and a test that depends on 999999 being
    free is a test that fails on somebody else's machine one day for no reason.
    """
    finished = subprocess.Popen([sys.executable, "-c", ""])
    assert finished.wait(timeout=60) == 0
    return finished.pid


# ── liveness ─────────────────────────────────────────────────────────────────


def test_this_process_is_alive() -> None:
    assert process_is_alive(os.getpid())


def test_a_finished_process_is_not_alive(dead_pid: int) -> None:
    assert not process_is_alive(dead_pid)


def test_probing_a_live_process_does_not_kill_it() -> None:
    """The reason ``os.kill(pid, 0)`` is not used on Windows.

    CPython's Windows ``os.kill`` ignores the signal for anything but a console-control
    event and calls ``TerminateProcess``, so the obvious existence probe terminates what it
    is asking about. Here that would be the deployed session. The probe is run against a
    child that would notice.
    """
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE,
    )
    try:
        assert process_is_alive(child.pid)
        assert child.poll() is None, "the liveness probe terminated the process"
    finally:
        child.kill()
        child.wait(timeout=10)


# ── acquire and refuse ───────────────────────────────────────────────────────


def test_acquiring_a_free_directory_writes_the_pid_and_the_mode(tmp_path) -> None:
    holder = acquire(tmp_path, mode="deployed", command="python -m glassbox.live_loop")

    assert holder.pid == os.getpid()
    assert holder.mode == "deployed"
    on_disk = json.loads((tmp_path / LOCK_FILE).read_text(encoding="utf-8"))
    assert on_disk["pid"] == os.getpid()
    assert on_disk["mode"] == "deployed"
    assert on_disk["command"] == "python -m glassbox.live_loop"


def test_a_second_launch_against_the_same_directory_refuses(tmp_path) -> None:
    """The mechanism itself: the deployed session holds it, a rehearsal is turned away."""
    acquire(
        tmp_path, mode="deployed", command="python -m glassbox.live_loop --sessions 3"
    )

    with pytest.raises(LockRefused) as refusal:
        acquire(tmp_path, mode="rehearsal:gate2-execution-path")

    assert str(os.getpid()) in str(refusal.value)
    assert "deployed" in str(refusal.value)


def test_the_refusal_names_the_directory_it_is_refusing(tmp_path) -> None:
    acquire(tmp_path, mode="deployed")
    with pytest.raises(LockRefused, match=str(tmp_path.name)):
        acquire(tmp_path, mode="deployed")


def test_a_different_state_directory_is_not_blocked(tmp_path) -> None:
    """The lock is per directory, so an independent run stays possible."""
    acquire(tmp_path / "live", mode="deployed")
    holder = acquire(tmp_path / "sandbox", mode="rehearsal:something")
    assert holder.pid == os.getpid()


# ── stale locks ──────────────────────────────────────────────────────────────


def test_a_stale_lock_is_reclaimed(tmp_path, dead_pid: int, caplog) -> None:
    """A loop killed by the host never gets to clean up. The next launch must not be
    blocked forever by a file its owner cannot remove."""
    adopted = {
        "pid": dead_pid,
        "mode": "deployed",
        "command": "python -m glassbox.live_loop --sessions 3",
        "acquired": "2026-08-25T17:47:24Z",
        "note": "",
    }
    (tmp_path / LOCK_FILE).write_text(json.dumps(adopted), encoding="utf-8")

    with caplog.at_level("WARNING"):
        holder = acquire(tmp_path, mode="deployed")

    assert holder.pid == os.getpid()
    assert read_lock(tmp_path).pid == os.getpid()
    assert "stale" in caplog.text, "reclaiming silently is how the mechanism disappears"
    assert str(dead_pid) in caplog.text


def test_an_unreadable_lock_refuses_rather_than_being_reclaimed(tmp_path) -> None:
    """The asymmetry, stated as a test.

    A lock this module cannot parse might name a running loop. Treating it as absent is
    the one wrong guess that costs a duplicated order, so it refuses and tells a person to
    look. The cost of being wrong the other way is one deleted file.
    """
    (tmp_path / LOCK_FILE).write_text("{not json", encoding="utf-8")

    with pytest.raises(LockRefused, match="cannot be read"):
        acquire(tmp_path, mode="deployed")


def test_a_lock_missing_its_fields_is_unreadable_too(tmp_path) -> None:
    (tmp_path / LOCK_FILE).write_text('{"pid": 1}', encoding="utf-8")
    with pytest.raises(LockRefused, match="cannot be read"):
        acquire(tmp_path, mode="deployed")


# ── release ──────────────────────────────────────────────────────────────────


def test_release_frees_the_directory(tmp_path) -> None:
    holder = acquire(tmp_path, mode="deployed")
    assert release(tmp_path, holder)
    assert not lock_path(tmp_path).exists()
    assert acquire(tmp_path, mode="rehearsal:x").pid == os.getpid()


def test_release_does_not_remove_somebody_elses_lock(tmp_path, dead_pid: int) -> None:
    """A process exiting late must not unlock a directory another loop has since taken."""
    stale = acquire(tmp_path, mode="deployed", pid=dead_pid)
    successor = acquire(tmp_path, mode="deployed")

    assert not release(tmp_path, stale)
    assert read_lock(tmp_path).pid == successor.pid


def test_release_on_an_already_free_directory_is_not_an_error(tmp_path) -> None:
    holder = acquire(tmp_path, mode="deployed")
    lock_path(tmp_path).unlink()
    assert not release(tmp_path, holder)


# ── adoption ─────────────────────────────────────────────────────────────────


def test_a_lock_can_be_written_for_an_already_running_process(tmp_path) -> None:
    """The transitional case: the deployed run predates the lock and must not restart."""
    holder = adopt(
        tmp_path,
        pid=os.getpid(),
        mode="deployed",
        command="python -m glassbox.live_loop --sessions 3",
        note="adopted",
    )
    assert holder.note == "adopted"
    with pytest.raises(LockRefused):
        acquire(tmp_path, mode="rehearsal:gate2-execution-path")


def test_adopting_a_dead_process_is_refused(tmp_path, dead_pid: int) -> None:
    """Otherwise a typo blocks the directory until somebody notices the file."""
    with pytest.raises(LockRefused, match="not running"):
        adopt(tmp_path, pid=dead_pid, mode="deployed", command="x", note="y")
    assert not lock_path(tmp_path).exists()


def test_adoption_refuses_to_overwrite_an_existing_lock(tmp_path) -> None:
    acquire(tmp_path, mode="deployed")
    with pytest.raises(LockRefused, match="already exists"):
        adopt(tmp_path, pid=os.getpid(), mode="deployed", command="x", note="y")


# ── the CLI, end to end ──────────────────────────────────────────────────────


def test_the_cli_refuses_a_second_launch_and_exits_non_zero(tmp_path) -> None:
    """Through ``main``, because the module being right is not the claim being made.

    The claim is that *running the command twice* refuses, and that it refuses **before**
    loading a config or a checkpoint - which is why this passes with neither present.
    """
    from glassbox.live_lock import EXIT_REFUSED
    from glassbox.live_loop import main

    state = tmp_path / "live"
    acquire(state, mode="deployed", command="python -m glassbox.live_loop --sessions 3")

    code = main(
        [
            "--state-dir",
            str(state),
            "--log-dir",
            str(tmp_path / "logs"),
            "--rehearsal",
            "gate2-execution-path",
        ]
    )

    assert code == EXIT_REFUSED
    assert code != 0
