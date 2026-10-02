"""Symmetry transforms that carry the edge list with them, exactly.  [R3-6]

A transform that only moves the coordinates and then rebuilds the graph cannot
tell you whether a drift came from the *network* or from the *neighbour search*.
Each function here rewrites atom indices and periodic offsets so that every edge
still points at the same physical neighbour image. Combined with
``edge_mode='fixed'`` in the audit, that isolates the network and its numerics;
``edge_mode='rebuilt'`` then adds the neighbour search back in on purpose.
"""

from __future__ import annotations

import torch

__all__ = [
    "random_rotations", "shift_atoms_fixed_edges", "translate_fixed_edges",
    "permute_fixed_edges", "supercell_fixed_edges",
]


def random_rotations(n: int, seed: int = 0, dtype=torch.float32, device="cpu",
                     proper: bool = True) -> torch.Tensor:
    """``n`` rotation matrices drawn uniformly (Haar) from SO(3), or O(3) if not proper."""
    g = torch.Generator().manual_seed(int(seed))
    out = []
    for _ in range(int(n)):
        q, r = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        if proper and torch.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        out.append(q)
    return torch.stack(out).to(device=device, dtype=dtype)


def shift_atoms_fixed_edges(frac, edges, n_int):
    """Add an integer lattice vector to every atom and compensate the offsets.

    ``f'_i = f_i + n_i``, ``s'_ij = s_ij + n_i - n_j``, so every edge still points
    at the same image. The physical structure is unchanged.
    """
    src, dst, off, em = edges
    n_int = n_int.to(frac.dtype)
    ni = torch.gather(n_int, 1, src.unsqueeze(-1).expand(-1, -1, 3))
    nj = torch.gather(n_int, 1, dst.unsqueeze(-1).expand(-1, -1, 3))
    return frac + n_int, (src, dst, off.to(frac.dtype) + ni - nj, em)


def translate_fixed_edges(frac, edges, shift, wrap: bool = True):
    """Rigid fractional translation, optionally wrapped back into ``[0, 1)``.

    The offsets of every edge that crosses the wrap are corrected.
    """
    f1 = frac + shift
    if not wrap:
        return f1, edges
    return shift_atoms_fixed_edges(f1, edges, -torch.floor(f1))


def permute_fixed_edges(types, frac, mask, edges, perm):
    """Reorder atoms and map edge endpoints through the inverse permutation.

    New atom ``k`` = old atom ``perm[k]``; old index ``i`` -> new index
    ``argsort(perm)[i]``.
    """
    src, dst, off, em = edges
    B, A = types.shape
    inv = torch.argsort(perm, dim=1)
    types_p = torch.gather(types, 1, perm)
    frac_p = torch.gather(frac, 1, perm.unsqueeze(-1).expand(B, A, 3))
    mask_p = torch.gather(mask, 1, perm)
    return types_p, frac_p, mask_p, (torch.gather(inv, 1, src),
                                     torch.gather(inv, 1, dst), off, em)


def supercell_fixed_edges(types, frac, L, mask, edges, reps=(2, 1, 1)):
    """Exact ``(n1, n2, n3)`` supercell of a padded batch together with its edge list.

    New atom index = ``i * R + r`` for replica ``r`` with cell vector ``c_r``
    (meshgrid order). An edge ``(i -> j, s)`` from replica ``r`` points to cell
    ``c = c_r + s``, i.e. to replica ``c mod n`` with supercell offset
    ``floor(c / n)``. Every edge vector is preserved, so an intensive predictor
    must return the same value.
    """
    dev, dt = frac.device, frac.dtype
    n1, n2, n3 = (int(x) for x in reps)
    R = n1 * n2 * n3
    nvec = torch.tensor([n1, n2, n3], device=dev, dtype=dt)
    sh = torch.stack(torch.meshgrid(
        torch.arange(n1, device=dev, dtype=dt), torch.arange(n2, device=dev, dtype=dt),
        torch.arange(n3, device=dev, dtype=dt), indexing="ij"), -1).reshape(R, 3)
    B, A = types.shape
    frac_s = ((frac.unsqueeze(2) + sh.view(1, 1, R, 3)) / nvec).reshape(B, A * R, 3)
    types_s = types.unsqueeze(2).expand(B, A, R).reshape(B, A * R)
    mask_s = mask.unsqueeze(2).expand(B, A, R).reshape(B, A * R)
    L_s = L * nvec.view(1, 3, 1)
    if edges is None:
        return types_s, frac_s, L_s, mask_s, None
    src, dst, off, em = edges
    E = src.size(1)
    c = sh.view(1, 1, R, 3) + off.to(dt).unsqueeze(2)                # (B,E,R,3)
    new_off = torch.floor(c / nvec)
    cm = c - new_off * nvec
    r_dst = (cm[..., 0] * (n2 * n3) + cm[..., 1] * n3 + cm[..., 2]).round().long()
    r_src = torch.arange(R, device=dev).view(1, 1, R)
    src_s = (src.unsqueeze(2) * R + r_src).reshape(B, E * R)
    dst_s = (dst.unsqueeze(2) * R + r_dst).reshape(B, E * R)
    em_s = em.unsqueeze(2).expand(B, E, R).reshape(B, E * R)
    return types_s, frac_s, L_s, mask_s, (src_s, dst_s, new_off.reshape(B, E * R, 3), em_s)
