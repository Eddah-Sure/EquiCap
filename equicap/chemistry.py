"""Element reference data and the per-element physical descriptor block.

These tables are fallbacks: whatever the dataset's ``element_properties`` entry
supplies takes precedence, and these values are used only for elements the
dataset does not describe.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ELEM_FEAT_DIM", "ATOMIC_RADII", "OXIDATION_STATES",
    "build_element_phys_features",
]

#: Number of physical descriptors per element appended to the one-hot block.
ELEM_FEAT_DIM = 4

#: Covalent radii in angstrom, keyed by atomic number.
ATOMIC_RADII = {
    1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 10: 0.58,
    11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 18: 1.06, 19: 2.03, 20: 1.76,
    21: 1.70, 22: 1.60, 23: 1.53, 24: 1.39, 25: 1.39, 26: 1.32, 27: 1.26, 28: 1.24, 29: 1.32, 30: 1.22,
    31: 1.22, 32: 1.20, 33: 1.19, 34: 1.20, 35: 1.20, 36: 1.16, 37: 2.20, 38: 1.95, 39: 1.90, 40: 1.75,
    41: 1.64, 42: 1.54, 43: 1.47, 44: 1.46, 45: 1.42, 46: 1.39, 47: 1.45, 48: 1.44, 49: 1.42, 50: 1.39,
    51: 1.39, 52: 1.38, 53: 1.39, 54: 1.40, 55: 2.44, 56: 2.15, 57: 2.07, 58: 2.04, 59: 2.03, 60: 2.01,
    61: 1.99, 62: 1.98, 63: 1.98, 64: 1.96, 65: 1.94, 66: 1.92, 67: 1.92, 68: 1.89, 69: 1.90, 70: 1.87,
    71: 1.87, 72: 1.75, 73: 1.70, 74: 1.62, 75: 1.51, 76: 1.44, 77: 1.41, 78: 1.36, 79: 1.36, 80: 1.32,
    81: 1.45, 82: 1.46, 83: 1.48, 84: 1.40, 85: 1.50, 86: 1.50, 87: 2.60, 88: 2.21, 89: 2.15,
    90: 2.06, 91: 2.00, 92: 1.96, 93: 1.90, 94: 1.87, 95: 1.80, 96: 1.69,
}

#: Common oxidation states, most representative first, keyed by atomic number.
OXIDATION_STATES = {
    1: [1, -1], 3: [1], 4: [2], 5: [3], 6: [4, -4, 2], 7: [-3, 3, 5], 8: [-2], 9: [-1], 11: [1], 12: [2],
    13: [3], 14: [4, -4], 15: [-3, 3, 5], 16: [-2, 4, 6], 17: [-1, 5, 7], 19: [1], 20: [2], 21: [3],
    22: [4, 3, 2], 23: [5, 4, 3, 2], 24: [3, 6, 2], 25: [2, 4, 7], 26: [3, 2], 27: [2, 3], 28: [2, 3],
    29: [2, 1], 30: [2], 31: [3], 32: [4, 2], 33: [3, 5, -3], 34: [-2, 4, 6], 35: [-1, 5], 37: [1], 38: [2],
    39: [3], 40: [4], 41: [5, 3], 42: [4, 6], 44: [4, 3], 45: [3], 46: [2, 4], 47: [1], 48: [2], 49: [3],
    50: [4, 2], 51: [3, 5], 52: [4, 6, -2], 53: [-1, 5, 7], 55: [1], 56: [2], 57: [3], 58: [3, 4], 59: [3],
    60: [3], 62: [3, 2], 63: [3, 2], 64: [3], 65: [3], 66: [3], 67: [3], 68: [3], 69: [3], 70: [3, 2],
    71: [3], 72: [4], 73: [5], 74: [6, 4], 75: [4, 7], 76: [4], 77: [3, 4], 78: [2, 4], 79: [3, 1], 80: [2, 1],
    81: [1, 3], 82: [2, 4], 83: [3], 89: [3],
    2: [0], 10: [0], 18: [0], 36: [0], 54: [0],
    43: [4, 7], 61: [3], 90: [4], 91: [5, 4],
    92: [6, 4, 5, 3], 93: [5, 4, 6, 3], 94: [4, 3, 6, 5],
    95: [3], 96: [3], 84: [4, 2, -2, 6], 85: [-1, 1, 5], 86: [0], 87: [1], 88: [2], 97: [3], 98: [3],
}


def build_element_phys_features(atomic_numbers, elem_props) -> np.ndarray:
    """``(Z, 4)`` normalised physical descriptors.

    Columns: covalent radius, mean oxidation state, max ``|oxidation state|``,
    valence-electron count — each scaled into roughly ``[0, 1]`` so they sit on
    the same footing as the one-hot block they are concatenated to.
    """
    phys = np.zeros((len(atomic_numbers), ELEM_FEAT_DIM), dtype=np.float32)
    for idx, z in enumerate(atomic_numbers):
        ep = elem_props.get(str(z), elem_props.get(int(z), {})) or {}
        r = float(ep.get("atomic_radius", ATOMIC_RADII.get(int(z), 1.5)))
        ox = ep.get("oxidation_states", None) or OXIDATION_STATES.get(int(z), [0])
        ox = [float(v) for v in ox] if len(ox) else [0.0]
        ve = float(ep.get("valence_electrons", 0))
        phys[idx] = [r / 3.0, (float(np.mean(ox)) + 4.0) / 12.0,
                     float(np.max(np.abs(ox))) / 8.0, ve / 18.0]
    return phys
