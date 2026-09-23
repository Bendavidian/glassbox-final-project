"""Every figure the momentum entry states is a cell of the file that produced it.

``IDEAS_PARKED.md``'s cross-sectional momentum entry declines an arm on measured grounds,
and it opens by faulting two figures in this project's history - the 16.00% gross exposure
and GB-46's ~86% - for having been computed once into prose with no producing code. It
then states nineteen figures of its own. **An entry that makes that argument owes the same
obligation**, and `scripts/momentum_probe.py` existing is only half of it: a script and a
paragraph are two places holding the same numbers, and two places diverge unless something
makes them equal. This is that something.

**Scoped to one table, deliberately, so it cannot become a prose parser.** The entry
carries exactly one table of shape ``| control | quantity | value |`` and this guard reads
that table and nothing else in the file. Prose around it may say "near zero" or "changes
sign"; those are readings of the table, not copies of it, and a guard that tried to check
them would be checking English. **What that leaves uncovered is stated rather than
implied:** the cost figures further down the entry come from ``results.csv`` and
``PROGRESS.md``, not from the probe, and nothing here pins them.

**The quantity cell is an address into the CSV**, in one of three forms:

``rebalances``
    the ``value`` column of the row with that ``quantity`` and no ``subject``.
``months_held:NVDA``
    the ``value`` column of that ``quantity`` for that ``subject``.
``top_n_minus_equal_weight@p``
    the ``p`` column of that row instead of its ``value``.

**Why a tolerance rather than an equality.** The entry prints a rounded figure and the CSV
holds full precision, so the comparison is "the entry states this value rounded to the
entry's own number of decimal places", expressed as half a unit in the last place written.
An exact ``round()`` comparison would fail on a value sitting on a rounding tie for no
reason a reader could act on, and a guard that goes red without a cause is a guard somebody
eventually weakens.

**The row count is pinned by hand and that is a second copy on purpose.** It is the
non-vacuity property: a parser that drifted and matched nothing would otherwise pass for
having checked everything, which is a defect this project has several instances of. Adding
a figure to the table means editing :data:`EXPECTED_FIGURES` in the same commit, and that
is the intended cost.

**Deliberately not a determinism test.** Re-running the probe inside the suite would check
that the machine reproduces itself; the committed CSV is what the entry cites, so the
committed CSV is what must agree with it.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

ENTRY = ROOT / "IDEAS_PARKED.md"
PROBE = ROOT / "report" / "momentum_probe.csv"

#: The table's first line, exactly. It anchors the parser, so another table elsewhere in
#: the file cannot be picked up and this one cannot be reformatted without a failure that
#: names the reason.
HEADER = "| control | quantity | value |"

#: How many figures the entry states. See the module docstring: pinned by hand, on purpose.
EXPECTED_FIGURES = 19


def figures(text: str) -> list[tuple[str, str, str]]:
    """The ``(control, quantity, value)`` rows of the one fixed-shape table.

    Raises:
        AssertionError: the header is absent, or appears more than once. A second copy of
            the table is the same defect as a second copy of a number, and a reader would
            have no way to tell which one the guard checked.
    """
    lines = text.splitlines()
    starts = [index for index, line in enumerate(lines) if line.strip() == HEADER]
    assert starts, (
        f"{ENTRY.name} has no line reading exactly {HEADER!r}. The momentum entry's "
        "figure table is the thing this guard checks; if it was reformatted, restore the "
        "shape rather than relaxing the parser."
    )
    assert len(starts) == 1, (
        f"{len(starts)} tables in {ENTRY.name} carry the header {HEADER!r}. One table, or "
        "a reader cannot tell which set of figures was checked."
    )

    rows: list[tuple[str, str, str]] = []
    for line in lines[starts[0] + 1 :]:
        if not line.lstrip().startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3 or set(cells[0]) <= {"-", ":"}:
            continue
        rows.append((cells[0], cells[1], cells[2]))
    return rows


def address(quantity: str) -> tuple[str, str, str]:
    """Split a quantity cell into ``(quantity, subject, column)``. See the docstring."""
    column = "value"
    if quantity.endswith("@p"):
        quantity, column = quantity[: -len("@p")], "p"
    name, _, subject = quantity.partition(":")
    return name, subject, column


def written_value(cell: str) -> tuple[float, int]:
    """A figure as written, and how many decimal places it was written to."""
    text = cell.strip().replace("−", "-").lstrip("+")
    _, _, fraction = text.partition(".")
    return float(text), len(fraction)


def _probe_cell(table: pd.DataFrame, control: str, quantity: str) -> float:
    name, subject, column = address(quantity)
    found = table[
        (table["control"] == control)
        & (table["quantity"] == name)
        & (table["subject"] == subject)
    ]
    assert len(found) == 1, (
        f"the entry states a figure for ({control!r}, {quantity!r}) and "
        f"{PROBE.name} holds {len(found)} row(s) matching quantity {name!r} "
        f"subject {subject!r}. Re-run `python scripts/momentum_probe.py`, or correct the "
        "entry to name a cell that exists."
    )
    return float(found[column].iloc[0])


@pytest.fixture(scope="module")
def probe() -> pd.DataFrame:
    """The committed probe output, with the empty subject as a string rather than NaN."""
    assert PROBE.is_file(), (
        f"{PROBE} is missing. The momentum entry quotes it, so the entry has no source "
        "until it is regenerated: `python scripts/momentum_probe.py`."
    )
    table = pd.read_csv(PROBE)
    table["subject"] = table["subject"].fillna("")
    return table


def test_the_entry_states_the_figures_it_is_pinned_to() -> None:
    """Non-vacuity. A drifted parser matching nothing must not pass for having checked."""
    rows = figures(ENTRY.read_text(encoding="utf-8"))
    assert len(rows) == EXPECTED_FIGURES, (
        f"the momentum table holds {len(rows)} figures and this guard is pinned to "
        f"{EXPECTED_FIGURES}. Adding or removing a figure means editing EXPECTED_FIGURES "
        "in the same commit - that is the cost of the count being explicit, and it is "
        "cheaper than a parser that silently checks nothing."
    )
    keys = [(control, quantity) for control, quantity, _ in rows]
    assert len(set(keys)) == len(keys), f"the table states a figure twice: {keys}"


def test_every_figure_in_the_entry_is_the_value_the_probe_produced(
    probe: pd.DataFrame,
) -> None:
    """The guard. Editing either side without the other goes red, naming both values."""
    wrong: list[str] = []
    for control, quantity, cell in figures(ENTRY.read_text(encoding="utf-8")):
        stated, places = written_value(cell)
        measured = _probe_cell(probe, control, quantity)
        if not math.isclose(stated, measured, abs_tol=0.5 * 10.0**-places, rel_tol=0.0):
            wrong.append(
                f"  {control} {quantity}: the entry says {cell} and "
                f"{PROBE.name} holds {measured!r}"
            )
    assert not wrong, (
        "IDEAS_PARKED.md and the probe output disagree about "
        f"{len(wrong)} figure(s):\n" + "\n".join(wrong) + "\nOne of them was edited "
        "without the other. Re-run `python scripts/momentum_probe.py` if the measurement "
        "changed, or correct the entry if it did not."
    )


def test_the_guard_can_see_an_edited_figure_and_a_missing_table() -> None:
    """Non-vacuity for the guard itself, on synthetic text rather than the real file.

    Proves the three ways it is meant to fail - a moved figure, an absent table, a
    duplicated one - without anybody having to break the repository to find out.
    """
    table = (
        f"prose above\n{HEADER}\n"
        "|---|---|---|\n"
        "| real | rebalances | 115 |\n"
        "| real | months_held:NVDA | 78 |\n"
        "\nprose below\n"
    )
    assert figures(table) == [
        ("real", "rebalances", "115"),
        ("real", "months_held:NVDA", "78"),
    ]

    with pytest.raises(AssertionError, match="no line reading exactly"):
        figures("a file with no figure table at all")
    with pytest.raises(AssertionError, match="carry the header"):
        figures(f"{table}\n{table}")

    assert address("rebalances") == ("rebalances", "", "value")
    assert address("months_held:NVDA") == ("months_held", "NVDA", "value")
    assert address("top_n_minus_equal_weight@p") == (
        "top_n_minus_equal_weight",
        "",
        "p",
    )

    assert written_value("+0.005965") == (0.005965, 6)
    assert written_value("-0.007067") == (-0.007067, 6)
    assert written_value("−0.007067") == (-0.007067, 6)
    assert written_value("115") == (115.0, 0)

    # A figure moved by one unit in its last written place must not survive the tolerance.
    stated, places = written_value("+0.005965")
    assert math.isclose(stated, 0.0059649, abs_tol=0.5 * 10.0**-places, rel_tol=0.0)
    assert not math.isclose(stated, 0.005966, abs_tol=0.5 * 10.0**-places, rel_tol=0.0)
