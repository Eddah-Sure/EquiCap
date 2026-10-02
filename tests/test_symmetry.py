"""The symmetry contract, for every backbone x readout cell.

These are the tests that matter: an architecture change that quietly breaks an
invariance will pass every accuracy check and fail here.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from equicap.audit import symmetry_audit
from equicap.chemistry import ELEM_FEAT_DIM
from equicap.geometry import build_edges_from_geometry
from equicap.model import BACKBONES, READOUTS, CrystalPropertyModel
from equicap.spacegroups import is_centrosymmetric, is_sohncke
from equicap.utils import set_seed

SMALL = dict(hidden=32, n_layers=2, num_rbf=8, prim_dim=16, n_caps=4, sec_dim=16,
             cutoff=4.0, max_deg=48)
TOL = 1e-8


@pytest.fixture(scope="module")
def batch():
    set_seed(0)
    B, A = 4, 6
    frac = torch.rand(B, A, 3)
    L = (torch.eye(3) * 4.5 + 0.4 * torch.rand(3, 3)).repeat(B, 1, 1)
    mask = torch.ones(B, A, dtype=torch.bool)
    mask[3, 4:] = False
    src, dst, off, em = build_edges_from_geometry(frac, L, mask, cutoff=4.0, max_deg=48)
    return {"types": torch.randint(0, 10, (B, A)), "frac": frac, "lattice": L,
            "mask": mask, "src": src, "dst": dst, "offset": off, "emask": em,
            "space_group": torch.ones(B, dtype=torch.long)}


def _model(**kw):
    set_seed(0)
    m = CrystalPropertyModel(num_types=10, elem_feat_dim=ELEM_FEAT_DIM, **{**SMALL, **kw})
    return m.eval()


@pytest.mark.parametrize("backbone,readout,edge_mode",
                         list(itertools.product(BACKBONES, READOUTS, ("fixed", "rebuilt"))))
def test_invariances_hold(batch, backbone, readout, edge_mode):
    m = _model(backbone=backbone, readout=readout)
    drift = symmetry_audit(m, batch, edge_mode=edge_mode, n_rotations=4,
                           dtype=torch.float64, verbose=False)
    bad = {k: v for k, v in drift.items() if v > TOL}
    assert not bad, bad


def test_supplied_and_rebuilt_graphs_agree(batch):
    for backbone in BACKBONES:
        m = _model(backbone=backbone, readout="attention")
        with torch.no_grad():
            a = m(batch["types"], batch["frac"], batch["lattice"], batch["mask"],
                  edges=(batch["src"], batch["dst"], batch["offset"], batch["emask"]))
            b = m(batch["types"], batch["frac"], batch["lattice"], batch["mask"])
        assert float((a - b).abs().max()) < 1e-5


def test_full_descriptors_break_supercell_invariance(batch):
    """A deliberate control: if this passes silently, the audit is not sensitive."""
    m = _model(backbone="cgcnn", readout="attention", global_features="full")
    drift = symmetry_audit(m, batch, edge_mode="fixed", dtype=torch.float64, verbose=False)
    assert drift["supercell_drift"] > 1e-6


def test_chirality_breaks_reflection_invariance(batch):
    m = _model(backbone="painn", readout="capsule", chirality=True)
    assert m.chirality
    drift = symmetry_audit(m, batch, edge_mode="fixed", dtype=torch.float64, verbose=False)
    assert drift["reflection_drift"] > 1e-8


def test_chirality_only_enabled_for_painn_capsule():
    assert not _model(backbone="cgcnn", readout="capsule", chirality=True).chirality
    assert not _model(backbone="painn", readout="attention", chirality=True).chirality


def test_per_structure_output_shape(batch):
    m = _model(backbone="painn", readout="capsule")
    summary, per = symmetry_audit(m, batch, edge_mode="fixed", n_rotations=2,
                                  verbose=False, per_structure=True)
    assert set(per) == {"rotation", "reflection", "translation", "permutation",
                        "periodic_image", "supercell"}
    assert all(v.shape == (batch["types"].shape[0],) for v in per.values())
    assert set(summary) == {f"{k}_drift" for k in per}


def test_space_group_tables_are_disjoint():
    assert is_centrosymmetric(225) and not is_sohncke(225)
    assert is_sohncke(198) and not is_centrosymmetric(198)
    assert not any(is_centrosymmetric(sg) and is_sohncke(sg) for sg in range(1, 231))
