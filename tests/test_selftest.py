"""The original end-to-end check, wrapped as a (slow) pytest case."""

from __future__ import annotations

import pytest

from equicap import self_test


@pytest.mark.slow
def test_self_test_passes():
    assert self_test(verbose=False) is True
