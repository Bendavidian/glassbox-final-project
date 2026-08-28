"""The console's colour tokens, and the contrast property that chose them.

**Its own module so the contrast test can walk every token without knowing their names.**
A palette scattered through a 2,400-line view module can only be checked by a test that
lists the colours it expects to find, which is the two-places family: the list and the
palette drift, both stay internally consistent, and the check passes while the thing it
checks has moved. :data:`TEXT_TOKENS` is that list, defined beside the values, and
``test_every_colour_is_classified`` fails if a colour is added here and left out of it.

**Contrast is why this palette was chosen, so it is computed rather than quoted.** Every
entry in :data:`TEXT_TOKENS` clears WCAG AA (4.5:1) against :data:`GROUND`, and every entry
in :data:`STAR_TOKENS` sits below 2:1 - the starfield's requirement is the opposite of the
text's, and a test that only checked "high enough" would pass a starfield bright enough to
read. Both directions are asserted.

Measured against ``#0E1116``:

===========  =========  ========
token        hex        ratio
===========  =========  ========
``TEXT``     ``#F0F3F7``  16.99
``GAIN``     ``#22C55E``   8.30
``DIM``      ``#7D8794``   5.19
``ACCENT``   ``#E8542A``   5.16
``LOSS``     ``#EF4444``   5.03
``STAR_B``   ``#3A4551``   1.94
``STAR_A``   ``#2E3742``   1.57
``RULE``     ``#202832``   1.27
===========  =========  ========

**Vermillion is chrome and never a value.** Section rules, active states, the symbol
column, selection. Gain and loss carry value. A number is never vermillion and a rule is
never green - the two families answer different questions, and a reader who has to work out
which one a mark belongs to is a reader the palette has failed.
"""

from __future__ import annotations

#: The only background. Every region sits on it; there are no panels and no fills.
GROUND = "#0E1116"

TEXT = "#F0F3F7"
DIM = "#7D8794"

#: Chrome only. See the module docstring.
ACCENT = "#E8542A"

GAIN = "#22C55E"
LOSS = "#EF4444"

#: 1px separators. Not text, and deliberately below AA - a rule a reader *notices* is a
#: rule competing with the numbers it separates.
RULE = "#202832"
RULE_FAINT = "#171D25"

#: The starfield. Nearly invisible by intent: if a reader notices a star while reading a
#: number, it is too strong.
STAR_A = "#2E3742"
STAR_B = "#3A4551"

#: Everything that carries text or a mark a reader must resolve. Walked by the contrast
#: test; adding a colour above without adding it here fails `test_every_colour_is_classified`.
TEXT_TOKENS = (TEXT, DIM, ACCENT, GAIN, LOSS)

#: The opposite requirement: these must stay *under* 2:1.
STAR_TOKENS = (STAR_A, STAR_B)

#: Rules are neither - they are structure, not information, and are excluded from both
#: assertions deliberately rather than by omission.
RULE_TOKENS = (RULE, RULE_FAINT)

#: One family, terminal. IBM Plex Mono is loaded from Google Fonts by the stylesheet; the
#: rest of the stack is what a font failure degrades to rather than breaks on.
MONO = "'IBM Plex Mono', 'JetBrains Mono', ui-monospace, 'Courier New', monospace"

#: The Hebrew narrative keeps a sans stack. Monospace is for the English chrome and the
#: numerals, and a Hebrew sentence set in IBM Plex Mono would fall back anyway.
HEBREW_SANS = "'Segoe UI', 'Arial Hebrew', Arial, sans-serif"


def relative_luminance(hex_colour: str) -> float:
    """WCAG 2.x relative luminance of an ``#RRGGBB`` colour.

    Here rather than in the test, because a test that computes the property it asserts
    from its own copy of the formula is checking its own arithmetic. The view module uses
    it too, to keep a colour's contrast checkable at the point it is chosen.
    """
    channels = [int(hex_colour[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(one: str, two: str) -> float:
    """WCAG contrast between two ``#RRGGBB`` colours, lighter over darker."""
    first, second = relative_luminance(one), relative_luminance(two)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


__all__ = [
    "ACCENT",
    "DIM",
    "GAIN",
    "GROUND",
    "HEBREW_SANS",
    "LOSS",
    "MONO",
    "RULE",
    "RULE_FAINT",
    "RULE_TOKENS",
    "STAR_A",
    "STAR_B",
    "STAR_TOKENS",
    "TEXT",
    "TEXT_TOKENS",
    "contrast_ratio",
    "relative_luminance",
]
