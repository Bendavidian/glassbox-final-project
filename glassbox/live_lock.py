"""One loop per state directory, enforced rather than remembered.

On 25 Aug 2026 a stale terminal relaunched a rehearsal against the same state directory
as the deployed session. It was harmless **only** because the bar it would have decided
had already been decided, so the "one decision per completed bar" rule refused it - an
accident, not a mechanism, and one that expires the moment a new bar completes. Two loops
sharing ``pending.json`` is two orders from one approval.

The mechanism is a lock file in the state directory holding the PID and the launch mode.
A second process reads it, asks whether that PID is still running, and either refuses by
name or reclaims a lock whose owner is gone.

**The failure directions are not symmetric, and the code leans deliberately.** Refusing a
launch that should have been allowed costs a person one message and one deleted file.
Allowing a second loop costs a duplicated order against a live account. So every case
this module cannot resolve - an unreadable lock, a process it cannot query - resolves to
*refuse*.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

LOGGER = logging.getLogger(__name__)

LOCK_FILE = "live.lock"

#: ``main`` returns this when another loop holds the state directory. Distinct from the
#: general failure code so a runbook can tell "somebody is already running it" from "the
#: config is wrong" without reading the message.
EXIT_REFUSED = 3


class LockRefused(RuntimeError):
    """Another loop holds this state directory, or the lock cannot be reasoned about."""


@dataclass(frozen=True)
class LockHolder:
    """Who holds the state directory, in the terms a person needs to act on it."""

    pid: int
    mode: str
    command: str
    acquired: str
    #: Set only for a lock written on a running process's behalf. See
    #: :func:`adopt`; empty for every lock a loop takes for itself.
    note: str = ""

    def describe(self) -> str:
        held = f"PID {self.pid} (mode {self.mode!r}) since {self.acquired}"
        return f"{held} [{self.note}]" if self.note else held


def _now() -> str:
    """UTC, matching every other timestamp the loop writes (spec 4.1)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def lock_path(state_dir: str | Path) -> Path:
    return Path(state_dir) / LOCK_FILE


def process_is_alive(pid: int) -> bool:
    """Is ``pid`` a running process?

    **Never use ``os.kill(pid, 0)`` for this on Windows.** POSIX treats signal 0 as an
    existence probe, but CPython's Windows ``os.kill`` ignores the signal number for
    anything that is not a console-control event and calls ``TerminateProcess(handle,
    sig)`` - so the "probe" kills the process it was asking about. The deployed session
    this module exists to protect would be the first casualty.

    Returns ``True`` when the answer cannot be determined, per the module docstring: an
    unanswerable question about a possible live loop resolves to refuse.
    """
    if pid <= 0:
        return False

    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        ERROR_ACCESS_DENIED = 5

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            # Access denied means the process exists and belongs to somebody else, which
            # is still "alive". Anything else - typically ERROR_INVALID_PARAMETER - means
            # there is no such process.
            return ctypes.get_last_error() == ERROR_ACCESS_DENIED
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists, owned by somebody else.
        return True
    return True


def read_lock(state_dir: str | Path) -> LockHolder | None:
    """The current holder, or ``None`` if the directory is free.

    Raises:
        LockRefused: The lock exists but cannot be read or parsed. Deliberately not
            treated as absent - a file this module cannot understand may name a running
            loop, and the refusal tells a person to look rather than guessing for them.
    """
    path = lock_path(state_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return LockHolder(**payload)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError) as failure:
        raise LockRefused(
            f"the lock at {path} cannot be read ({failure}). Refusing rather than "
            "assuming it is stale. If no loop is running, delete it."
        ) from failure


def _payload(holder: LockHolder) -> str:
    return json.dumps(asdict(holder), indent=2) + "\n"


def acquire(
    state_dir: str | Path,
    *,
    mode: str,
    command: str | None = None,
    pid: int | None = None,
) -> LockHolder:
    """Take the state directory, or refuse and say who holds it.

    A lock whose PID is no longer running is **reclaimed**, not obeyed: a loop killed by
    the host, by a reboot or by ``taskkill /F`` never gets to clean up after itself, and a
    lock that outlived its process would turn every hard stop into a manual step.

    Raises:
        LockRefused: A running process holds it, or the lock cannot be read.
    """
    directory = Path(state_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = lock_path(directory)
    mine = LockHolder(
        pid=os.getpid() if pid is None else pid,
        mode=mode,
        command=" ".join(sys.argv) if command is None else command,
        acquired=_now(),
    )

    # O_EXCL rather than exists()-then-write: two loops launched in the same second must
    # not both find the directory free.
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        pass
    else:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(_payload(mine))
        return mine

    holder = read_lock(directory)
    if holder is None:
        # Released between the O_EXCL failure and the read. Try once more from the top;
        # a second FileExistsError means somebody genuinely beat us to it.
        return acquire(directory, mode=mode, command=mine.command, pid=mine.pid)

    if process_is_alive(holder.pid):
        raise LockRefused(
            f"{holder.describe()} already holds {directory}. Two loops on one state "
            "directory is two orders from one approval. Stop it first, or run against a "
            "different --state-dir."
        )

    LOGGER.warning(
        "reclaiming a stale lock on %s: %s is no longer running",
        directory,
        holder.describe(),
    )
    replacement = directory / f"{LOCK_FILE}.{mine.pid}.tmp"
    replacement.write_text(_payload(mine), encoding="utf-8")
    os.replace(replacement, path)

    # Two processes can reach the reclaim together and both replace. Whoever's PID is in
    # the file afterwards owns it; the other refuses rather than running unlocked.
    confirmed = read_lock(directory)
    if confirmed is None or confirmed.pid != mine.pid:
        raise LockRefused(
            f"lost the race to reclaim the stale lock on {directory}: it is now held by "
            f"{confirmed.describe() if confirmed else 'another process'}."
        )
    return mine


def release(state_dir: str | Path, holder: LockHolder) -> bool:
    """Give up the lock, but only if it is still ours.

    Returns ``False`` without touching anything when the lock has been reclaimed by
    somebody else - deleting *their* lock on the way out would leave a running loop
    unprotected, which is worse than leaving a stale file behind.
    """
    try:
        current = read_lock(state_dir)
    except LockRefused:
        return False
    if current is None or current.pid != holder.pid:
        return False
    try:
        lock_path(state_dir).unlink()
    except OSError:
        return False
    return True


def adopt(
    state_dir: str | Path, *, pid: int, mode: str, command: str, note: str
) -> LockHolder:
    """Write a lock on behalf of a process that is already running.

    The lock landed on 26 Aug 2026, mid-way through a three-session deployed run that
    predates it. Restarting that run to make it take its own lock would have destroyed the
    thing being protected - GATE 2 criterion 6 asks for consecutive sessions in one
    process - so the lock is written *for* it instead.

    The adopted lock is an ordinary lock in every respect that matters: a second launch
    probes the PID, finds it running and refuses. When the run ends it leaves the file
    behind, and the next launch reclaims it by the stale path. ``note`` records why it was
    not self-taken, so a reader of the file is not left inferring it.

    Raises:
        LockRefused: The directory is already locked, or ``pid`` is not running - adopting
            a dead process's lock would block the directory for no reason.
    """
    if not process_is_alive(pid):
        raise LockRefused(f"PID {pid} is not running; there is nothing to adopt.")
    holder = LockHolder(pid=pid, mode=mode, command=command, acquired=_now(), note=note)
    path = lock_path(Path(state_dir))
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = read_lock(state_dir)
        raise LockRefused(
            f"{path} already exists, held by "
            f"{existing.describe() if existing else 'an unreadable lock'}."
        ) from None
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(_payload(holder))
    return holder


__all__ = [
    "EXIT_REFUSED",
    "LOCK_FILE",
    "LockHolder",
    "LockRefused",
    "acquire",
    "adopt",
    "lock_path",
    "process_is_alive",
    "read_lock",
    "release",
]
