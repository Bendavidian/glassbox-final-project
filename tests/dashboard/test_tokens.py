"""GB-63c: the palette's contrast is the reason it was chosen, so it is computed.

A palette recorded as prose in DECISIONS and used as literals in a view module is the
two-places family with the second place in a document nobody greps. These tests close it
from the code's side: the ratios are derived from the tokens, so a colour edited toward the
background fails here rather than in somebody's eyes six weeks later.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from glassbox.dashboard import tokens

#: WCAG AA for body text.
AA = 4.5

#: WCAG AAA for body text. `DIM` is held to this rather than to AA because it carries
#: every label, column head and chart caption on the page - the text a reader scans
#: rather than reads, and the text six report screenshots are full of.
AAA = 7.0

#: The floor `DIM` must keep *away* from `TEXT`. Contrast against the ground says each is
#: legible; it does not say they are legible as two different things, and a palette that
#: fixed legibility by collapsing its own hierarchy would pass every other test here.
SECONDARY_SEPARATION = 2.0

#: The starfield's requirement is the OPPOSITE of the text's. A test that only checked
#: "bright enough" would happily pass a field bright enough to read beside a numeral.
INVISIBLE = 2.0


@pytest.mark.parametrize("colour", tokens.TEXT_TOKENS)
def test_every_text_colour_clears_aa_against_the_ground(colour: str) -> None:
    """Computed from sRGB relative luminance, never quoted.

    The figures this palette was proposed with were right in six of seven cases and wrong
    in the seventh - `#2E3742` was given as 1.27 when it is 1.57, which is `RULE`'s ratio.
    Nothing about the design changed, but a hand-checked table is a table that can be
    wrong in exactly that way, and this cannot.
    """
    ratio = tokens.contrast_ratio(colour, tokens.GROUND)

    assert ratio >= AA, f"{colour} is {ratio:.2f}:1 against the ground, below AA"


@pytest.mark.parametrize("colour", tokens.STAR_TOKENS)
def test_the_starfield_stays_under_the_threshold_of_notice(colour: str) -> None:
    """If a reader notices a star while reading a number, it is too strong."""
    ratio = tokens.contrast_ratio(colour, tokens.GROUND)

    assert ratio < INVISIBLE, f"{colour} is {ratio:.2f}:1 - the starfield is legible"


def test_every_colour_the_module_defines_is_classified() -> None:
    """**The guard on the guard.**

    `TEXT_TOKENS` is what the contrast test walks, so a colour added to this module and
    left out of the tuple is a colour nothing checks - and it would be added by exactly
    the kind of edit that is confident it does not need checking. Every `#RRGGBB` literal
    must appear in one of the three classifications, and the classification it lands in is
    a decision somebody has to make rather than a default.
    """
    source = Path(tokens.__file__).read_text(encoding="utf-8")
    body = source.split('"""', 2)[-1]  # the module docstring quotes ratios; skip it
    defined = set(re.findall(r'"(#[0-9A-Fa-f]{6})"', body))
    classified = {
        *tokens.TEXT_TOKENS,
        *tokens.STAR_TOKENS,
        *tokens.RULE_TOKENS,
        tokens.GROUND,
    }

    assert defined <= classified, (
        f"unclassified colours: {sorted(defined - classified)}. Every colour is text, "
        "starfield, rule or the ground, and which one it is decides what is asserted "
        "about it"
    )


def test_the_ground_is_not_itself_a_text_token() -> None:
    """It would trivially fail the contrast test against itself, and a palette that had to
    special-case its own background inside the assertion would be hiding the distinction
    the assertion depends on."""
    assert tokens.GROUND not in tokens.TEXT_TOKENS
    assert tokens.contrast_ratio(tokens.GROUND, tokens.GROUND) == 1.0


def test_the_luminance_formula_is_the_one_wcag_specifies() -> None:
    """Anchored on the two endpoints, so a transcription error in the linearisation is
    caught rather than propagated into every ratio above."""
    assert tokens.relative_luminance("#FFFFFF") == pytest.approx(1.0)
    assert tokens.relative_luminance("#000000") == pytest.approx(0.0)
    assert tokens.contrast_ratio("#FFFFFF", "#000000") == pytest.approx(21.0)


def test_gain_and_loss_are_distinguishable_from_each_other() -> None:
    """They sit side by side in a table and mean opposite things. Contrast against the
    ground says each is legible; it does not say they are legible *apart*."""
    assert tokens.contrast_ratio(tokens.GAIN, tokens.LOSS) > 1.5


def test_the_secondary_text_colour_clears_aaa_against_the_ground() -> None:
    """**Raised in the GB-63d typography pass, and this is the assertion that holds it.**

    `DIM` was `#7D8794` at 5.19:1 - AA, and set at 9px on a near-black ground for every
    label on the page. AA is measured on colour alone and knows nothing about size, so the
    palette test above passed while the thing it was protecting was unreadable in print.
    The size floor is enforced elsewhere; this is the colour half of the same repair.
    """
    ratio = tokens.contrast_ratio(tokens.DIM, tokens.GROUND)

    assert ratio >= AAA, f"DIM is {ratio:.2f}:1 against the ground, below AAA"


def test_the_secondary_text_colour_stays_apart_from_the_primary() -> None:
    """The other direction, and it is not implied by the first.

    Raising `DIM` far enough would clear any contrast floor by making it `TEXT`, at which
    point the console has one text colour and no hierarchy - and every ratio in this module
    would still be green. Both bounds are asserted because only the pair says what was
    actually wanted.
    """
    ratio = tokens.contrast_ratio(tokens.DIM, tokens.TEXT)

    assert ratio >= SECONDARY_SEPARATION, (
        f"DIM and TEXT are {ratio:.2f}:1 apart - secondary text has stopped reading as "
        "a different thing from primary"
    )


@pytest.mark.parametrize("step", tokens.TYPE_STEPS)
def test_no_step_of_the_type_scale_is_below_the_floor(step: int) -> None:
    """The floor is the point of the scale, so it is asserted on the scale itself rather
    than on the stylesheet that spends it. A step added below 12 fails here, before any
    rule uses it."""
    assert (
        step >= tokens.TYPE_FLOOR
    ), f"{step}px is below the {tokens.TYPE_FLOOR}px floor"


def test_the_type_scale_has_no_more_steps_than_a_reader_can_hold() -> None:
    """**Five is the brief's number and four is what the console needed.**

    The guard is on the count rather than on the values: a scale grows one convenient
    exception at a time, and the next size is always the one nobody decided to add.
    """
    assert len(tokens.TYPE_STEPS) <= 5
    assert len(set(tokens.TYPE_STEPS)) == len(
        tokens.TYPE_STEPS
    ), "two steps of the scale are the same size, which makes one of them a duplicate name"
    assert list(tokens.TYPE_STEPS) == sorted(tokens.TYPE_STEPS), "the scale is ordered"
