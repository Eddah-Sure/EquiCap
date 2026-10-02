"""Neighbour-list and lattice-geometry properties."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from equicap.geometry import (build_edges_from_geometry, edge_geometry,
                              matrix_to_lattice_params, perpendicular_widths,
                              required_images, structure_edges)


def test_lattice_params_cubic():
    L = torch.eye(3) * 4.0
    p = matrix_to_lattice_params(L)
    assert torch.allclose(p[:3], torch.full((3,), 4.0), atol=1e-5)
    assert torch.allclose(p[3:], torch.full((3,), 90.0), atol=1e-4)


def test_perpendicular_widths_cubic():
    L = (torch.eye(3) * 3.0).unsqueeze(0)
    assert torch.allclose(perpendicular_widths(L), torch.full((1, 3), 3.0), atol=1e-5)


def test_required_images_thin_cell():
    thin = torch.tensor([[2.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]]).unsqueeze(0)
    assert required_images(thin, cutoff=6.0)[0] >= 3


def test_rock_salt_keeps_whole_shells():
    """6 nearest + 12 second-nearest neighbours fall inside 4.5 A for a=5.64."""
    frac = torch.tensor([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                         [.5, 0, 0], [0, .5, 0], [0, 0, .5], [.5, .5, .5]]).unsqueeze(0)
    L = (torch.eye(3) * 5.64).unsqueeze(0)
    *_, em = build_edges_from_geometry(frac, L, torch.ones(1, 8, dtype=torch.bool),
                                       cutoff=4.5, max_deg=64)
    deg = em.view(1, 8, -1).sum(-1)
    assert bool((deg == 18).all()), deg.tolist()


def test_no_edge_exceeds_cutoff():
    torch.manual_seed(0)
    frac = torch.rand(2, 7, 3)
    L = (torch.eye(3) * 4.5).repeat(2, 1, 1)
    mask = torch.ones(2, 7, dtype=torch.bool)
    src, dst, off, em = build_edges_from_geometry(frac, L, mask, cutoff=4.0, max_deg=48)
    _, d = edge_geometry(frac, L, src, dst, off)
    assert float(d[em].max()) < 4.0 + 1e-4


def test_unwrapped_coordinates_give_identical_edges():
    """Adding integer lattice vectors must not change the physical edge set."""
    torch.manual_seed(1)
    frac = torch.rand(1, 5, 3)
    L = (torch.eye(3) * 4.2).unsqueeze(0)
    mask = torch.ones(1, 5, dtype=torch.bool)
    shift = torch.randint(-2, 3, (1, 5, 3)).to(frac.dtype)

    a = build_edges_from_geometry(frac, L, mask, cutoff=4.0, max_deg=48)
    b = build_edges_from_geometry(frac + shift, L, mask, cutoff=4.0, max_deg=48)
    _, da = edge_geometry(frac, L, a[0], a[1], a[2])
    _, db = edge_geometry(frac + shift, L, b[0], b[1], b[2])
    sa = np.sort(da[a[3]].numpy())
    sb = np.sort(db[b[3]].numpy())
    assert sa.shape == sb.shape
    assert np.allclose(sa, sb, atol=1e-6)


def test_structure_edges_roundtrip():
    rng = np.random.default_rng(0)
    frac = rng.random((6, 3))
    L = np.eye(3) * 4.0
    ei, off, ovf = structure_edges(frac, L, cutoff=4.0, max_deg=48)
    assert ei.shape[0] == 2 and off.shape[1] == 3 and ei.shape[1] == off.shape[0]
    assert ovf == 0


def test_rejects_unknown_mode():
    with pytest.raises(ValueError):
        build_edges_from_geometry(torch.rand(1, 2, 3), torch.eye(3).unsqueeze(0),
                                  torch.ones(1, 2, dtype=torch.bool), mode="nope")
