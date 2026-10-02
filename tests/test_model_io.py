"""Model construction, checkpoint round trips, metrics, collation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from equicap.checkpoint import load_property_model, save_checkpoint
from equicap.chemistry import ELEM_FEAT_DIM, build_element_phys_features
from equicap.config import default_config
from equicap.data import crystal_collate
from equicap.geometry import structure_edges
from equicap.metrics import compute_metrics
from equicap.model import GLOBAL_FEATURE_DIMS, CrystalPropertyModel
from equicap.utils import set_seed

SMALL = dict(hidden=32, n_layers=2, num_rbf=8, prim_dim=16, n_caps=4, sec_dim=16,
             cutoff=4.0, max_deg=48)


def _inputs(B=3, A=5, T=10):
    set_seed(0)
    return (torch.randint(0, T, (B, A)), torch.rand(B, A, 3),
            (torch.eye(3) * 4.5).repeat(B, 1, 1), torch.ones(B, A, dtype=torch.bool))


def test_checkpoint_roundtrip(tmp_path):
    types, frac, L, mask = _inputs()
    m = CrystalPropertyModel(num_types=10, elem_feat_dim=ELEM_FEAT_DIM,
                             backbone="painn", readout="capsule", **SMALL).eval()
    m.set_elem_feat(np.random.rand(10, ELEM_FEAT_DIM))
    m.set_y_stats(0.5, 2.0)
    p = tmp_path / "ck.pt"
    save_checkpoint(str(p), m)
    m2, ck = load_property_model(str(p))
    assert ck["config"]["backbone"] == "painn"
    assert m2.y_stats == pytest.approx((0.5, 2.0))
    with torch.no_grad():
        assert float((m(types, frac, L, mask) - m2(types, frac, L, mask)).abs().max()) < 1e-6


def test_from_config_ignores_extra_keys():
    m = CrystalPropertyModel(num_types=8, **SMALL)
    cfg = dict(m.get_config(), junk="ignore me")
    assert CrystalPropertyModel.from_config(cfg).get_config() == m.get_config()


def test_vocabulary_mismatch_raises():
    m = CrystalPropertyModel(num_types=4, **SMALL).eval()
    types, frac, L, mask = _inputs(T=4)
    types[0, 0] = 99
    with pytest.raises(IndexError):
        m(types, frac, L, mask)


def test_bad_factor_names_raise():
    for kw in ({"backbone": "nope"}, {"readout": "nope"},
               {"neighbor_mode": "nope"}, {"global_features": "nope"}):
        with pytest.raises(ValueError):
            CrystalPropertyModel(num_types=4, **{**SMALL, **kw})


def test_composition_backbone_drops_structural_descriptors():
    m = CrystalPropertyModel(num_types=4, backbone="composition", readout="mean",
                             global_features="intensive", **SMALL)
    assert m.global_features == "none"
    m = CrystalPropertyModel(num_types=4, backbone="composition", readout="mean",
                             global_features="density", **SMALL)
    assert m.global_features == "density"


def test_head_width_tracks_global_feature_block():
    for gf, dim in GLOBAL_FEATURE_DIMS.items():
        m = CrystalPropertyModel(num_types=4, backbone="cgcnn", readout="mean",
                                 global_features=gf, **SMALL)
        assert m.head[0].in_features == m.readout.out_dim + dim


def test_extensive_scales_with_atom_count():
    types, frac, L, mask = _inputs(B=1, A=5)
    m = CrystalPropertyModel(num_types=10, extensive=True, backbone="cgcnn",
                             readout="mean", **SMALL).eval()
    m.set_y_stats(1.0, 1.0)
    with torch.no_grad():
        total = m(types, frac, L, mask)
        per_atom = m(types, frac, L, mask, denorm=False) * m.y_std + m.y_mean
    assert float((total - per_atom * 5).abs().max()) < 1e-5


def test_metrics_perfect_prediction():
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = compute_metrics(y, y)
    assert out["mae"] == pytest.approx(0.0)
    assert out["r2"] == pytest.approx(1.0)
    assert out["r"] == pytest.approx(1.0, abs=1e-6)
    assert out["n"] == 4


def test_metrics_mean_baseline_has_unit_mad_ratio():
    y = torch.tensor([1.0, 2.0, 3.0, 10.0])
    out = compute_metrics(torch.full_like(y, float(y.mean())), y)
    assert out["mad_over_mae"] == pytest.approx(1.0)
    assert out["r2"] == pytest.approx(0.0, abs=1e-9)


def test_collate_pads_and_masks():
    items = []
    for i in range(3):
        n = 4 + i
        fr = np.random.default_rng(i).random((n, 3))
        ei, off, _ = structure_edges(fr, np.eye(3) * 4.0, 4.0, 48)
        items.append({"types": torch.zeros(n, dtype=torch.long),
                      "edge_index": torch.tensor(ei), "edge_offsets": torch.tensor(off),
                      "y": torch.randn(1), "material_id": f"m{i}",
                      "lattice_matrix": torch.eye(3) * 4.0,
                      "frac_coords": torch.tensor(fr, dtype=torch.float32),
                      "space_group": torch.tensor([1]), "n_atoms": n})
    b = crystal_collate(items)
    assert b["types"].shape == (3, 6)
    assert int(b["mask"].sum()) == 15
    assert b["material_ids"] == ["m0", "m1", "m2"]


def test_default_config_covers_model_keys():
    from equicap.config import MODEL_KEYS
    cfg = default_config()
    assert all(k in cfg for k in MODEL_KEYS)


def test_element_features_normalised():
    phys = build_element_phys_features([1, 6, 26, 92], {})
    assert phys.shape == (4, ELEM_FEAT_DIM)
    assert np.isfinite(phys).all()
    assert (phys >= -1).all() and (phys <= 2).all()
