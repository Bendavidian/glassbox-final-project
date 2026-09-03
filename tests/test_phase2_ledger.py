"""Phase 2's boundary, expressed as an invariant instead of a date.

``GLASSBOX_PHASE2_EXPANSION.md`` sets 15 September as the moment *"Phase 2 closes in
whatever state it is in"*, and annotates it *"a hard boundary, not a target"*. That is the
same construction as GB-63's timebox, which on 25 Aug 2026 **failed by expiring unused**:
the window held zero dashboard commits, so there was nothing to stop and nothing to park,
and the clause covering *"whatever has not landed by 23:00"* was a rule about a remainder
when there was no remainder. An event at a timestamp needs a reader, and nobody was
standing there.

So this is **not** a check that fires on 15 September. A check that fires on the 15th is a
check that has never run before the 15th, which is the cache-gated test of 20 Aug wearing a
calendar. It is an invariant, evaluated on every run of the suite, and the suite is its
reader. A boundary that is an invariant cannot pass unused, because there is no passing.

**Every Phase 2 task must be in exactly one of two ledgers** - recorded complete in
``PROGRESS.md``, or named in ``IDEAS_PARKED.md``. A task nobody has touched is in neither,
so **the default state is the failing state**. That is the property GB-63's ruling lacked.

**What this cannot do, stated because a guard that oversells itself is worse than none.**
The parking half is genuinely enforced: naming a task in ``IDEAS_PARKED.md`` *is* the
required act, so asserting the name checks the actual thing. "Complete" is a claim about
code, and a row asserting it is prose. :func:`test_a_completion_record_carries_a_resolvable_citation`
raises the price from *prose* to *anchored prose* and stops there. This guard cannot tell a
correct completion from a false one. It guarantees only that no Phase 2 task reaches the
boundary without someone having decided, in writing, which of two states it is in.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: The three documents this guard reads. **No missing-file guard, deliberately**: if one is
#: gone the ledger cannot be checked, and a test that skips when it cannot check is the
#: defect this one exists to close. ``read_text`` raises, which is the correct outcome.
EXPANSION = ROOT / "GLASSBOX_PHASE2_EXPANSION.md"
PROGRESS = ROOT / "PROGRESS.md"
PARKED = ROOT / "IDEAS_PARKED.md"

#: ``### GB-61 · Universe expansion, 5 → 20 symbols · 5 SP`` - the expansion's task
#: headings, which are the source of the task list. Hand-listing the six here would be the
#: third copy of the universe all over again: a GB-67 added to the expansion would be
#: invisible to a tuple written in this file.
TASK_HEADING = re.compile(r"^### (GB-\d+)\s*·", re.MULTILINE)

#: A completion declaration in a row's **title cell**. ``\b`` keeps "incomplete" out, which
#: matters: the retracted row ``~~GB-66 is incomplete against its own acceptance
#: criterion~~`` sits three lines above the real one and contains the word.
DECLARED_COMPLETE = re.compile(r"\bcomplete[ds]?\b", re.IGNORECASE)

#: A struck-through title is a withdrawn claim. Reading one as live would be the honesty
#: layer failing in the direction it exists to prevent.
WITHDRAWN = "~~"

#: Backticked tokens in a row, from which a citation is resolved.
BACKTICKED = re.compile(r"`([^`]+)`")

#: A short or full commit hash.
SHA = re.compile(r"^[0-9a-f]{7,40}$")

#: ``glassbox/dashboard/app.py:42`` or a bare path.
PATH_CITATION = re.compile(r"^([\w./\-]+\.(?:py|md|ya?ml|csv|toml))(?::\d+(?:-\d+)?)?$")

#: A test function named as a citation, with or without its module.
NODE_ID = re.compile(r"(?:^|::)(test_\w+)$")


def _tasks() -> tuple[str, ...]:
    """Every Phase 2 task ID, in the order the expansion declares them."""
    return tuple(TASK_HEADING.findall(EXPANSION.read_text(encoding="utf-8")))


def _rows(text: str) -> list[tuple[str, str]]:
    """Every markdown table row as ``(title cell, whole row)``.

    The title is cell 1 because a completion is declared where a reader sees it. A row
    whose *body* mentions ``GB-64`` in passing is not a record that GB-64 is done, and
    matching the whole row would make every cross-reference a completion.
    """
    rows = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = line.split("|")
        if len(cells) < 3:
            continue
        rows.append((cells[1].strip(), line))
    return rows


def completed(progress_text: str, tasks: tuple[str, ...]) -> dict[str, str]:
    """Tasks with a completion-declaring, non-withdrawn title row, mapped to that row."""
    found: dict[str, str] = {}
    for title, row in _rows(progress_text):
        if WITHDRAWN in title or not DECLARED_COMPLETE.search(title):
            continue
        for task in tasks:
            if re.search(rf"{task}\b", title):
                found.setdefault(task, row)
    return found


def parked(parked_text: str, tasks: tuple[str, ...]) -> set[str]:
    """Tasks named by ID anywhere in the parked-ideas file."""
    return {task for task in tasks if re.search(rf"{task}\b", parked_text)}


def unresolved_citations(row: str) -> bool:
    """Whether a row carries at least one citation that resolves.

    A commit that exists, a file that exists, or a test that is defined. It does not
    check that the citation *supports* the claim - only that the claim is anchored to
    something real, which is strictly more than prose and strictly less than proof.
    """
    for token in BACKTICKED.findall(row):
        token = token.strip()
        if SHA.match(token) and _commit_exists(token):
            return False
        path = PATH_CITATION.match(token)
        if path and (ROOT / path.group(1)).exists():
            return False
        node = NODE_ID.search(token)
        if node and _test_is_defined(node.group(1)):
            return False
    return True


def _commit_exists(sha: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _test_is_defined(name: str) -> bool:
    pattern = re.compile(rf"^def {re.escape(name)}\(", re.MULTILINE)
    return any(
        pattern.search(path.read_text(encoding="utf-8", errors="ignore"))
        for path in (ROOT / "tests").rglob("test_*.py")
    )


# ── the ledger ───────────────────────────────────────────────────────────────


def test_the_phase_2_task_list_is_read_from_the_expansion() -> None:
    """The task list comes from the governing document, and the parser is not vacuous.

    Asserted explicitly because a parser that silently matches nothing gives a test that
    passes for having checked nothing - the same reason ``SPEC_COPIES`` is pinned in
    ``tests/config/test_config.py``.
    """
    tasks = _tasks()
    assert tasks, "no `### GB-NN ·` headings found; the expansion's format has moved"
    assert len(set(tasks)) == len(tasks), f"duplicate task headings: {tasks}"
    assert set(tasks) == {"GB-61", "GB-62", "GB-63", "GB-64", "GB-65", "GB-66"}, tasks


def test_every_phase_2_task_is_either_recorded_complete_or_parked() -> None:
    """**The guard.** A task in neither ledger fails, and that is the non-start property.

    GB-63's ruling produced no event when its window closed empty, because its clause
    covered a remainder and there was none. Here the empty case *is* the alarm: a task
    nobody has decided about is in neither ledger, so it is red by default rather than
    silent by default.
    """
    tasks = _tasks()
    done = completed(PROGRESS.read_text(encoding="utf-8"), tasks)
    away = parked(PARKED.read_text(encoding="utf-8"), tasks)
    undecided = [task for task in tasks if task not in done and task not in away]
    assert not undecided, (
        f"{len(undecided)} Phase 2 task(s) in neither ledger: {', '.join(undecided)}. "
        f"Each must be recorded complete in PROGRESS.md (a title cell naming the ID and "
        f"declaring completion) or named in IDEAS_PARKED.md. This is a decision to take, "
        f"not a test to fix."
    )


def test_no_phase_2_task_is_both_complete_and_parked() -> None:
    """Both is a failure too: two records disagreeing is the two-places defect."""
    tasks = _tasks()
    done = completed(PROGRESS.read_text(encoding="utf-8"), tasks)
    away = parked(PARKED.read_text(encoding="utf-8"), tasks)
    both = sorted(set(done) & away)
    assert (
        not both
    ), f"recorded complete AND parked, so the two ledgers disagree: {both}"


def test_a_completion_record_carries_a_resolvable_citation() -> None:
    """A completion claim must anchor to a commit, a file or a test that exists.

    This is GB-57's evidence-audit discipline turned into a test, and it is the half of
    Instance 10 that can be mechanised: a suite count cannot be checked at commit time,
    an anchor can.
    """
    tasks = _tasks()
    done = completed(PROGRESS.read_text(encoding="utf-8"), tasks)
    unanchored = sorted(task for task, row in done.items() if unresolved_citations(row))
    assert not unanchored, (
        f"completion recorded with no resolvable citation: {unanchored}. "
        f"Cite a commit SHA, a `path.py:line`, or a `test_name` in backticks."
    )


def test_the_guard_can_see_a_task_in_neither_state() -> None:
    """Non-vacuity for the guard itself.

    A synthetic task present in neither ledger must be reported as undecided. Without
    this, a regex that drifts and matches nothing passes for having checked everything -
    the shape of ``test_the_containment_guard_can_see_a_mark_in_every_chart``.
    """
    synthetic = ("GB-99",)
    assert not completed(PROGRESS.read_text(encoding="utf-8"), synthetic)
    assert not parked(PARKED.read_text(encoding="utf-8"), synthetic)

    declared = "| GB-99 COMPLETE: a synthetic row | 4 Sep 2026 | Ben | `HEAD` |"
    assert completed(declared, synthetic) == {"GB-99": declared}
    assert parked("## GB-99 parked for now", synthetic) == {"GB-99"}

    withdrawn = "| ~~GB-99 COMPLETE~~ - retracted | 4 Sep 2026 | Ben | `HEAD` |"
    assert not completed(withdrawn, synthetic), "a struck-through title read as live"

    negated = "| GB-99 is incomplete against its criterion | 4 Sep 2026 | Ben | x |"
    assert not completed(negated, synthetic), "'incomplete' read as a completion"

    body_only = "| A finding | 4 Sep 2026 | Ben | GB-99 is complete, said in passing |"
    assert not completed(body_only, synthetic), "a body mention read as a declaration"

    # A declaring title for one task whose body names another. Added after breaking the
    # ID lookup from the title cell to the whole row and finding this test could not see
    # it: `body_only` is filtered out by its non-declaring title before the lookup runs,
    # so it never exercised the line that moved.
    cross = "| GB-98 COMPLETE | 4 Sep 2026 | Ben | `HEAD`, and see also GB-99 |"
    assert not completed(
        cross, synthetic
    ), "another row's body credited as a completion"

    assert unresolved_citations("| GB-99 COMPLETE | x | y | `deadbeef1` |")
    assert not unresolved_citations("| GB-99 COMPLETE | x | y | `PROGRESS.md` |")


@pytest.mark.parametrize("document", [EXPANSION, PROGRESS, PARKED])
def test_the_ledger_documents_are_present_and_are_not_skipped_when_absent(
    document: Path,
) -> None:
    """The no-skip property: a missing document is a failure, never a skip.

    Every reader above calls ``read_text`` with no existence check, so a missing document
    raises ``FileNotFoundError`` and the suite goes red rather than quietly passing. This
    names the three documents so the failure says *which* one, instead of a traceback from
    whichever guard happened to run first.
    """
    assert (
        document.exists()
    ), f"{document.name} is missing; the ledger cannot be checked"
