"""Complete-pipeline invariance audits.

Two independent checks:

* :func:`symmetry_audit` — does the trained predictor return the same number for
  the same physical crystal under rotation, reflection, translation,
  permutation, choice of periodic image, and supercell replication?
* :func:`audit_edges` — do the *stored* periodic offsets actually reproduce
  distances inside the cutoff? Mass beyond the cutoff means the mined graph is
  wrong and ``edge_source='stored'`` must not be used.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import torch

from .geometry import build_edges_from_geometry, edge_geometry
from .transforms import (permute_fixed_edges, random_rotations, shift_atoms_fixed_edges,
                         supercell_fixed_edges, translate_fixed_edges)
from .utils import _warn_once

if TYPE_CHECKING:                                            # pragma: no cover
    from .model import CrystalPropertyModel

__all__ = ["symmetry_audit", "audit_edges"]

#: Drift above this is flagged as a violation in the printed report.
VIOLATION_THRESHOLD = 1e-3


def _cast_batch(b: dict, device, dtype) -> dict:
    out = {}
    for k, v in b.items():
        if torch.is_tensor(v):
            v = v.to(device)
            if v.is_floating_point() and dtype is not None:
                v = v.to(dtype)
        out[k] = v
    return out


@torch.no_grad()
def symmetry_audit(model: "CrystalPropertyModel", batch: dict, device="cpu",
                   edge_mode: str = "fixed", n_rotations: int = 8, seed: int = 0,
                   dtype: Optional[torch.dtype] = None, supercell=(2, 1, 1),
                   verbose: bool = True, per_structure: bool = False):
    """Complete-pipeline invariance audit.  [R3-6]

    ``edge_mode='fixed'``
        The batch's own edge list is transformed EXACTLY with the structure
        (offsets / indices remapped), so drift measures the network and its
        numerics only.
    ``edge_mode='rebuilt'``
        Every transformed structure gets a fresh graph from the model's own
        builder, so drift also includes neighbour selection and image
        enumeration.

    ``dtype=torch.float64`` runs the audit on a float64 copy of the model;
    comparing float32 and float64 separates arithmetic invariance from rounding
    effects.

    Returns ``{transform: max |drift|}``; with ``per_structure=True`` also returns
    a dict of per-structure drift arrays (max over rotations), for subgroup
    analysis (e.g. centrosymmetric vs Sohncke space groups for the reflection
    test — see :mod:`equicap.spacegroups`).
    """
    if edge_mode not in ("fixed", "rebuilt"):
        raise ValueError("edge_mode must be 'fixed' or 'rebuilt'")
    m = model
    if dtype is not None and dtype != next(model.parameters()).dtype:
        m = copy.deepcopy(model).to(dtype)
    m = m.to(device).eval()
    pdt = next(m.parameters()).dtype
    b = _cast_batch(batch, device, pdt)
    types, frac, L, mask = b["types"], b["frac"], b["lattice"], b["mask"]
    B, A = types.shape
    has_edges = all(k in b for k in ("src", "dst", "offset", "emask"))
    if edge_mode == "fixed" and not has_edges:
        e = build_edges_from_geometry(frac, L, mask.bool(), cutoff=m.cutoff,
                                      max_deg=m.max_deg, mode=m.neighbor_mode)
    elif has_edges:
        e = (b["src"], b["dst"], b["offset"].to(pdt), b["emask"])
    else:
        e = None
    fixed = edge_mode == "fixed"

    def f(t, fr, lat, mk, ed):
        return m(t, fr, lat, mk, edges=(ed if fixed else None))

    base = f(types, frac, L, mask, e)
    per: Dict[str, torch.Tensor] = {}

    Rs = random_rotations(n_rotations, seed=seed, dtype=pdt, device=device)
    rot = torch.zeros(B, device=device, dtype=pdt)
    for R in Rs:
        rot = torch.maximum(rot, (base - f(types, frac, L @ R.T, mask, e)).abs())
    per["rotation"] = rot

    per["reflection"] = (base - f(types, frac, -L, mask, e)).abs()

    g = torch.Generator().manual_seed(int(seed) + 1)
    shift = torch.rand(1, 1, 3, generator=g).to(device=device, dtype=pdt)
    fr_t, e_t = translate_fixed_edges(frac, e, shift) if fixed else ((frac + shift) % 1.0, None)
    per["translation"] = (base - f(types, fr_t, L, mask, e_t)).abs()

    perm = torch.stack([torch.randperm(A, generator=g) for _ in range(B)]).to(device)
    if fixed:
        tp, fp, mp, ep = permute_fixed_edges(types, frac, mask, e, perm)
    else:
        tp = torch.gather(types, 1, perm)
        fp = torch.gather(frac, 1, perm.unsqueeze(-1).expand(B, A, 3))
        mp, ep = torch.gather(mask, 1, perm), None
    per["permutation"] = (base - f(tp, fp, L, mp, ep)).abs()

    n_int = torch.randint(-2, 3, (B, A, 3), generator=g).to(device=device, dtype=pdt)
    if fixed:
        fr_i, e_i = shift_atoms_fixed_edges(frac, e, n_int)
    else:
        fr_i, e_i = frac + n_int, None
    per["periodic_image"] = (base - f(types, fr_i, L, mask, e_i)).abs()

    ts, fs, Ls, ms, es = supercell_fixed_edges(types, frac, L, mask,
                                               e if fixed else None, reps=supercell)
    sup = f(ts, fs, Ls, ms, es)
    if m.extensive:
        sup = sup / float(np.prod(supercell))
    per["supercell"] = (base - sup).abs()

    out = {f"{k}_drift": float(v.max()) for k, v in per.items()}
    if verbose:
        print(f"  symmetry audit ({edge_mode} edges, {str(pdt).replace('torch.', '')})")
        for k, v in out.items():
            flag = ""
            if k == "reflection_drift" and m.chirality:
                flag = ("  (pseudoscalar channel on; non-zero allowed for "
                        "non-centrosymmetric cells)")
            elif v > VIOLATION_THRESHOLD:
                flag = "   <-- VIOLATION"
            print(f"    {k:24s} {v:.3e}{flag}")
    if per_structure:
        return out, {k: v.detach().cpu().double().numpy() for k, v in per.items()}
    return out


@torch.no_grad()
def audit_edges(loader, device="cpu", cutoff: float = 5.0,
                max_batches: int = 50) -> Dict[str, object]:
    """Recompute edge lengths from coordinates, lattice and offsets.

    Mass beyond the cutoff means the stored offsets are wrong.
    """
    ds, n_e, n_zero = [], 0, 0
    for k, batch in enumerate(loader):
        if k >= max_batches:
            break
        b = {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in batch.items()}
        _, d = edge_geometry(b["frac"], b["lattice"], b["src"], b["dst"], b["offset"])
        em = b["emask"]
        ds.append(d[em].cpu())
        n_e += int(em.sum())
        n_zero += int(((b["offset"].abs().sum(-1) == 0) & em).sum())
    d = torch.cat(ds).numpy() if ds else np.zeros(1, np.float32)
    res = {
        "n_edges": n_e,
        "frac_zero_offset": n_zero / max(n_e, 1),
        "frac_beyond_cutoff": float((d > cutoff + 1e-4).mean()),
        "frac_beyond_2x_cutoff": float((d > 2 * cutoff).mean()),
        "d_mean": float(d.mean()), "d_median": float(np.median(d)),
        "d_p99": float(np.percentile(d, 99)), "d_max": float(d.max()),
        "histogram": np.histogram(d, bins=120, range=(0.0, max(12.0, float(d.max())))),
    }
    if res["frac_beyond_cutoff"] > 0.01:
        _warn_once(f"{res['frac_beyond_cutoff']:.1%} of edges exceed the {cutoff} A "
                   f"cutoff -- stored periodic offsets are probably wrong")
    return res
