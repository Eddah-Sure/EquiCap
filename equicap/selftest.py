"""Synthetic end-to-end check that needs no dataset.

Run this first after installing, and again after any change to the geometry or
model code::

    python -m equicap --self-test

It asserts, for every backbone x readout combination, that: supplied and rebuilt
graphs agree; gradients are finite; every symmetry in the contract holds to
``1e-8`` in float64 with both fixed and rebuilt edges; and a round trip through a
checkpoint reproduces the predictions. It then asserts that the two deliberate
symmetry-breaking controls *do* break their symmetry, which is what makes the
passing numbers meaningful.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from .audit import symmetry_audit
from .checkpoint import load_property_model, save_checkpoint
from .chemistry import ELEM_FEAT_DIM
from .data import crystal_collate
from .geometry import build_edges_from_geometry, edge_geometry, required_images, structure_edges
from .model import BACKBONES, READOUTS, CrystalPropertyModel
from .utils import set_seed

__all__ = ["self_test"]


def self_test(device: str = "cpu", verbose: bool = True) -> bool:
    """Synthetic end-to-end check; no dataset required. Returns True or raises."""
    set_seed(0)
    dev = torch.device(device)
    B, A, T = 4, 6, 10
    types = torch.randint(0, T, (B, A))
    frac = torch.rand(B, A, 3)
    L = (torch.eye(3) * 4.5 + 0.4 * torch.rand(3, 3)).repeat(B, 1, 1)
    mask = torch.ones(B, A, dtype=torch.bool)
    mask[3, 4:] = False
    y = torch.randn(B)
    small = dict(hidden=32, n_layers=2, num_rbf=8, prim_dim=16, n_caps=4, sec_dim=16,
                 cutoff=4.0, max_deg=48)

    src, dst, off, em, ovf = build_edges_from_geometry(
        frac, L, mask, cutoff=4.0, max_deg=48, return_overflow=True)
    assert ovf == 0, "radius cap hit in self-test"
    batch = {"types": types, "frac": frac, "lattice": L, "mask": mask,
             "src": src, "dst": dst, "offset": off, "emask": em,
             "space_group": torch.ones(B, dtype=torch.long)}
    _, d = edge_geometry(frac, L, src, dst, off)
    assert float(d[em].max()) < 4.0 + 1e-4

    for backbone in BACKBONES:
        for readout in READOUTS:
            model = CrystalPropertyModel(num_types=T, elem_feat_dim=ELEM_FEAT_DIM,
                                         backbone=backbone, readout=readout, **small).to(dev)
            model.set_elem_feat(np.random.rand(T, ELEM_FEAT_DIM))
            model.set_y_stats(float(y.mean()), float(y.std()))
            model.eval()
            out = model(types, frac, L, mask, edges=(src, dst, off, em))
            out_rb = model(types, frac, L, mask)
            assert out.shape == (B,) and torch.isfinite(out).all()
            assert float((out - out_rb).abs().max()) < 1e-5, "supplied != rebuilt graph"
            _, lat = model(types, frac, L, mask, return_latent=True)
            assert "pooled" in lat
            if readout == "capsule":
                assert len(lat["coupling_history"]) == model.routing_iters

            model.train()
            loss = F.smooth_l1_loss(model(types, frac, L, mask, denorm=False),
                                    (y - model.y_mean) / model.y_std)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters()
                       if p.grad is not None)
            model.eval()

            for mode in ("fixed", "rebuilt"):
                aud = symmetry_audit(model, batch, device=device, edge_mode=mode,
                                     n_rotations=4, dtype=torch.float64, verbose=False)
                bad = {k: v for k, v in aud.items() if v > 1e-8}
                assert not bad, f"{backbone}/{readout}/{mode}: {bad}"

            with tempfile.TemporaryDirectory() as td:
                p = os.path.join(td, "ck.pt")
                save_checkpoint(p, model)
                m2, _ = load_property_model(p, map_location=str(dev))
                with torch.no_grad():
                    a = model(types, frac, L, mask)
                    c = m2.eval()(types, frac, L, mask)
                assert float((a - c).abs().max()) < 1e-5
            if verbose:
                print(f"  ok  {backbone:11s}/{readout:9s}  params={model.n_params():,}")

    # controls that must break a symmetry
    m = CrystalPropertyModel(num_types=T, elem_feat_dim=ELEM_FEAT_DIM, backbone="cgcnn",
                             readout="attention", global_features="full", **small).eval()
    dr = symmetry_audit(m, batch, edge_mode="fixed", dtype=torch.float64,
                        verbose=False)["supercell_drift"]
    assert dr > 1e-6, "legacy 'full' descriptors should break supercell invariance"
    m = CrystalPropertyModel(num_types=T, elem_feat_dim=ELEM_FEAT_DIM, backbone="painn",
                             readout="capsule", chirality=True, **small).eval()
    dr = symmetry_audit(m, batch, edge_mode="fixed", dtype=torch.float64,
                        verbose=False)["reflection_drift"]
    if verbose:
        print(f"  ok  'full' descriptors break supercell invariance; chiral reflection "
              f"drift on random (non-centrosymmetric) cells = {dr:.2e}")

    # radius graphs keep whole shells: rock salt, 6 + 12 neighbours inside 4.5 A
    fr_nacl = torch.tensor([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                            [.5, 0, 0], [0, .5, 0], [0, 0, .5], [.5, .5, .5]]).unsqueeze(0)
    Ln = (torch.eye(3) * 5.64).unsqueeze(0)
    *_, emn = build_edges_from_geometry(fr_nacl, Ln, torch.ones(1, 8, dtype=torch.bool),
                                        cutoff=4.5, max_deg=64)
    deg = emn.view(1, 8, -1).sum(-1)
    assert bool((deg == 18).all()), f"rock-salt radius degree {deg.tolist()} != 18"

    thin = torch.tensor([[2.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]]).unsqueeze(0)
    assert required_images(thin, cutoff=6.0)[0] >= 3

    items = []
    for i in range(3):
        n_at = 4 + i
        fr = np.random.rand(n_at, 3)
        ei, of, _ = structure_edges(fr, np.eye(3) * 4.0, 4.0, 48)
        items.append({"types": torch.randint(0, T, (n_at,)),
                      "edge_index": torch.tensor(ei), "edge_offsets": torch.tensor(of),
                      "y": torch.randn(1), "material_id": f"m{i}",
                      "lattice_matrix": torch.eye(3) * 4.0,
                      "frac_coords": torch.tensor(fr, dtype=torch.float32),
                      "space_group": torch.tensor([1]), "n_atoms": n_at})
    bt = crystal_collate(items)
    assert bt["types"].shape == (3, 6) and int(bt["mask"].sum()) == 15
    if verbose:
        print("  ok  graph builder, collate, rock-salt shells, adaptive images")
        print("self_test: all checks passed")
    return True
