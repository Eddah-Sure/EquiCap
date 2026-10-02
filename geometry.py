"""Lattice geometry and periodic neighbour-list construction.

Everything here is pure geometry: no learned parameters, no model state. The
neighbour list is the one place where a symmetry can be silently broken by an
implementation detail, so the two failure modes are made explicit:

* the radius graph keeps *whole* coordination shells, so the predictor stays a
  continuous function of the coordinates (with the cosine envelope);
* when the per-atom safety cap binds, the graph is no longer a radius graph —
  the number of affected atoms is returned and must be reported.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch

from .utils import _warn_once

__all__ = [
    "safe_norm", "smooth_norm", "matrix_to_lattice_params", "perpendicular_widths",
    "required_images", "edge_geometry", "build_edges_from_geometry",
    "structure_edges", "NEIGHBOR_MODES",
]


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------


def safe_norm(x, dim=-1, eps: float = 1e-12, keepdim: bool = False):
    """``||x||`` with the singularity at the origin removed from the gradient."""
    return (x.pow(2).sum(dim=dim, keepdim=keepdim) + eps).sqrt()


def smooth_norm(x, delta: float, dim=-1, keepdim: bool = False):
    """``sqrt(|x|^2 + delta) - sqrt(delta)``.

    Zero at the origin, Lipschitz, and insensitive to rounding residuals much
    smaller than ``sqrt(delta)``.  [R3-2]
    """
    return (x.pow(2).sum(dim=dim, keepdim=keepdim) + delta).sqrt() - math.sqrt(delta)


def neg_inf(dtype) -> float:
    """Dtype-aware masking sentinel; ``-1e9`` overflows to ``-inf`` in fp16."""
    return torch.finfo(dtype).min / 4


_neg_inf = neg_inf


# ---------------------------------------------------------------------------
# Lattice descriptors
# ---------------------------------------------------------------------------


def matrix_to_lattice_params(L: torch.Tensor) -> torch.Tensor:
    """3x3 lattice matrix -> ``(a, b, c, alpha, beta, gamma)`` with angles in degrees."""
    _sq = L.dim() == 2
    if _sq:
        L = L.unsqueeze(0)
    v1, v2, v3 = L[:, 0], L[:, 1], L[:, 2]
    a, b, c = safe_norm(v1), safe_norm(v2), safe_norm(v3)

    def _ang(u, v, nu, nv):
        return torch.acos(((u * v).sum(-1) / (nu * nv + 1e-8)).clamp(-1, 1)) * 180 / math.pi

    out = torch.stack([a, b, c, _ang(v2, v3, b, c), _ang(v1, v3, a, c), _ang(v1, v2, a, b)], -1)
    return out.squeeze(0) if _sq else out


def perpendicular_widths(L: torch.Tensor) -> torch.Tensor:
    """Perpendicular width of the cell along each lattice direction, ``(B, 3)``.

    ``w_k = V / ||a_i x a_j||``.
    """
    _sq = L.dim() == 2
    if _sq:
        L = L.unsqueeze(0)
    V = torch.linalg.det(L).abs().clamp(min=1e-8)
    a1, a2, a3 = L[:, 0], L[:, 1], L[:, 2]
    w = torch.stack([
        V / safe_norm(torch.cross(a2, a3, dim=-1)).clamp(min=1e-8),
        V / safe_norm(torch.cross(a3, a1, dim=-1)).clamp(min=1e-8),
        V / safe_norm(torch.cross(a1, a2, dim=-1)).clamp(min=1e-8),
    ], -1)
    return w.squeeze(0) if _sq else w


def required_images(L: torch.Tensor, cutoff: float, cap: int = 4) -> Tuple[int, int, int]:
    """Periodic images needed per axis to guarantee ``cutoff`` coverage.

    The cap bounds the cost for pathological cells; when it binds, coverage is
    incomplete and a warning is issued (the manuscript must report how often).
    """
    w = perpendicular_widths(L)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    need = torch.ceil(cutoff / w.clamp(min=1e-3)).max(dim=0).values
    if bool((need > cap).any()):
        _warn_once(f"a cell needs more than {cap} periodic images per axis for a "
                   f"{cutoff} A cutoff; neighbour search is truncated for it")
    n = need.clamp(min=1, max=cap).long().tolist()
    return int(n[0]), int(n[1]), int(n[2])


def _image_shifts(n: Tuple[int, int, int], device, dtype) -> torch.Tensor:
    rngs = [torch.arange(-k, k + 1, device=device, dtype=dtype) for k in n]
    return torch.stack(torch.meshgrid(*rngs, indexing="ij"), -1).reshape(-1, 3)


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


def edge_geometry(frac: torch.Tensor, L: torch.Tensor,
                  src: torch.Tensor, dst: torch.Tensor, offset: torch.Tensor):
    """``(displacement, distance)`` for a padded edge list.

    ``d_ij = (f_j + s_ij - f_i) L``.
    """
    C3 = src.unsqueeze(-1).expand(-1, -1, 3)
    fi = torch.gather(frac, 1, C3)
    fj = torch.gather(frac, 1, dst.unsqueeze(-1).expand(-1, -1, 3))
    dvec = torch.einsum("bek,bkl->bel", fj + offset.to(frac.dtype) - fi, L)
    return dvec, safe_norm(dvec)


NEIGHBOR_MODES = ("radius", "knn")


@torch.no_grad()
def build_edges_from_geometry(frac, L, mask, cutoff: float = 5.0, max_deg: int = 64,
                              mode: str = "radius",
                              n_images: Optional[Tuple[int, int, int]] = None,
                              shift_chunk: int = 64, return_overflow: bool = False):
    """Padded periodic neighbour list ``(src, dst, offset, emask)``.  [R3-4][R3-5]

    ``mode='radius'``
        Every ``(atom, image)`` pair with ``d < cutoff``, except the trivial self
        pair, up to a safety cap of ``max_deg`` per source atom. Equidistant
        shells are kept or dropped together, so with the cosine envelope the
        predictor is a continuous function of the coordinates. If the cap binds
        the graph is no longer a radius graph; the number of overflowing atoms is
        returned when ``return_overflow=True`` and must be reported.
    ``mode='knn'``
        The ``max_deg`` nearest pairs (classic CGCNN-style top-K) *that also fall
        inside the cutoff*, kept as the graph-construction ablation. Top-K
        truncation is discontinuous and breaks ties arbitrarily inside a shell.

    Distances are computed in float64 regardless of input dtype so that the
    selection does not depend on float32 rounding.
    """
    if mode not in NEIGHBOR_MODES:
        raise ValueError(f"mode must be one of {NEIGHBOR_MODES}, got {mode!r}")
    B, A, _ = frac.shape
    dev = frac.device
    f_raw, L64 = frac.double(), L.double()
    # Work with coordinates wrapped into [0,1) so the image range is sufficient,
    # then express the offsets relative to the caller's (possibly unwrapped)
    # coordinates: s' = s + k_i - k_j with k = floor(f).
    kf = torch.floor(f_raw)
    f64 = f_raw - kf
    if n_images is None:
        n_images = required_images(L64, cutoff)
    shifts = _image_shifts(n_images, dev, torch.float64)
    S = shifts.size(0)
    K = int(max(1, max_deg))

    df = f64.unsqueeze(2) - f64.unsqueeze(1)                       # f_i - f_j
    mb = mask.bool()
    pair = (mb.unsqueeze(2) & mb.unsqueeze(1)).unsqueeze(-1)
    eye = torch.eye(A, device=dev, dtype=torch.bool).view(1, A, A, 1)
    BIG = 1.0e4

    cand_d, cand_j, cand_s = [], [], []
    n_inside = torch.zeros(B, A, device=dev, dtype=torch.long)
    for lo in range(0, S, shift_chunk):
        sh = shifts[lo:lo + shift_chunk]
        sc = sh.size(0)
        disp = -df.unsqueeze(3) + sh.view(1, 1, 1, sc, 3)             # f_j + s - f_i
        d = safe_norm(torch.einsum("bijsk,bkl->bijsl", disp, L64))
        zero_s = (sh.abs().sum(-1) == 0).view(1, 1, 1, sc)
        invalid = (eye & zero_s) | (~pair)
        d = torch.where(invalid, torch.full_like(d, BIG), d)
        n_inside += (d < cutoff).reshape(B, A, -1).sum(-1)
        flat = d.reshape(B, A, A * sc)
        k = min(K, flat.size(-1))
        dk, idx = flat.topk(k, dim=-1, largest=False)
        cand_d.append(dk)
        cand_j.append(torch.div(idx, sc, rounding_mode="floor"))
        cand_s.append(sh[(idx % sc)])

    dk = torch.cat(cand_d, -1)
    jk = torch.cat(cand_j, -1)
    sk = torch.cat(cand_s, -2)
    k = min(K, dk.size(-1))
    dk, sel = dk.topk(k, dim=-1, largest=False)
    jk = torch.gather(jk, -1, sel)
    sk = torch.gather(sk, 2, sel.unsqueeze(-1).expand(B, A, k, 3))

    ok = (dk < cutoff) & (dk < 0.5 * BIG) & mb.unsqueeze(-1)
    src = torch.arange(A, device=dev).view(1, A, 1).expand(B, A, k)
    k_i = kf.unsqueeze(2).expand(B, A, k, 3)
    k_j = torch.gather(kf, 1, jk.reshape(B, A * k, 1).expand(B, A * k, 3)).reshape(B, A, k, 3)
    sk = sk + k_i - k_j
    out = (src.reshape(B, A * k), jk.reshape(B, A * k),
           sk.reshape(B, A * k, 3).to(frac.dtype), ok.reshape(B, A * k))
    if return_overflow:
        overflow = int(((n_inside > K) & mb).sum()) if mode == "radius" else 0
        return out + (overflow,)
    return out


def structure_edges(frac: np.ndarray, L: np.ndarray, cutoff: float, max_deg: int,
                    mode: str = "radius") -> Tuple[np.ndarray, np.ndarray, int]:
    """Edges for ONE structure as compact arrays.

    Returns ``edge_index (2, E)``, ``offsets (E, 3)`` and the number of atoms
    whose radius neighbourhood exceeded the cap.
    """
    f = torch.as_tensor(np.asarray(frac, dtype=np.float64)).unsqueeze(0)
    Lt = torch.as_tensor(np.asarray(L, dtype=np.float64)).unsqueeze(0)
    m = torch.ones(1, f.shape[1], dtype=torch.bool)
    src, dst, off, em, ovf = build_edges_from_geometry(
        f, Lt, m, cutoff=cutoff, max_deg=max_deg, mode=mode, return_overflow=True)
    em = em[0]
    ei = torch.stack([src[0][em], dst[0][em]]).numpy().astype(np.int64)
    of = off[0][em].numpy().astype(np.float32)
    return ei, of, ovf
