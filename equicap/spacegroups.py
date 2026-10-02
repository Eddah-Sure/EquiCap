"""Space-group membership tables used by the reflection audit.

A parity-odd (pseudoscalar) feature channel is *allowed* to change sign under
reflection, but only for crystals that are not centrosymmetric. Splitting the
reflection drift by these two sets is what turns "the number is non-zero" into
"the number is non-zero exactly where it should be".
"""

from __future__ import annotations

__all__ = ["CENTROSYMMETRIC", "SOHNCKE", "is_centrosymmetric", "is_sohncke"]

#: Centrosymmetric space groups (the 11 Laue classes).
CENTROSYMMETRIC = frozenset(
    [2] + list(range(10, 16)) + list(range(47, 75)) + list(range(83, 89))
    + list(range(123, 143)) + [147, 148] + list(range(162, 168))
    + [175, 176] + list(range(191, 195)) + list(range(200, 207))
    + list(range(221, 231))
)

#: The 65 Sohncke groups (proper operations only; chiral crystals live here).
SOHNCKE = frozenset(
    [1] + list(range(3, 6)) + list(range(16, 25)) + list(range(75, 81))
    + list(range(89, 99)) + list(range(143, 147)) + list(range(149, 156))
    + list(range(168, 174)) + list(range(177, 183)) + list(range(195, 200))
    + list(range(207, 215))
)

# Private aliases kept for continuity with the original single-file script.
_CENTRO = CENTROSYMMETRIC
_SOHNCKE = SOHNCKE


def is_centrosymmetric(sg: int) -> bool:
    """True when space group number ``sg`` contains an inversion centre."""
    return int(sg) in CENTROSYMMETRIC


def is_sohncke(sg: int) -> bool:
    """True when space group number ``sg`` contains only proper operations."""
    return int(sg) in SOHNCKE
