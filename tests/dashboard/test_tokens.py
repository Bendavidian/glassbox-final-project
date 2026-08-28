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
