"""Encoder building blocks: radial basis, message-passing convolutions, capsules.

Three ideas live here and nothing else:

1. :class:`RBF` — the distance expansion plus the cosine cutoff envelope, which
   is returned *separately* because it multiplies the whole message, not just
   the basis.
2. :class:`EquivariantConv` / :class:`CGConv` — the two structural backbones.
3. :class:`PrimaryCapsules` / :class:`MotifCapsules` — the capsule readout stack.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import safe_norm, smooth_norm

__all__ = [
    "RBF", "EquivariantConv", "CGConv", "PrimaryCapsules", "MotifCapsules", "PROBES",
]


# ---------------------------------------------------------------------------
# Radial basis
# ---------------------------------------------------------------------------


class RBF(nn.Module):
    """Gaussian radial basis plus a cosine cutoff envelope.

    ``forward`` returns ``(basis, envelope)``; the envelope is applied to the
    whole message by the caller so that messages go smoothly to zero at the
    cutoff.
    """

    def __init__(self, num_rbf: int = 16, cutoff: float = 5.0, trainable: bool = False):
        super().__init__()
        self.cutoff = float(cutoff)
        self.num_rbf = int(num_rbf)
        centers = torch.linspace(0.0, cutoff, num_rbf)
        width = torch.tensor(cutoff / max(num_rbf - 1, 1))
        if trainable:
            self.centers = nn.Parameter(centers)
            self.log_width = nn.Parameter(width.log())
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("log_width", width.log())

    def forward(self, dist: torch.Tensor):
        d = dist.clamp(max=self.cutoff).unsqueeze(-1)
        w = self.log_width.exp().clamp(min=1e-4)
        basis = torch.exp(-((d - self.centers) / w) ** 2)
        env = 0.5 * (torch.cos(math.pi * dist.clamp(max=self.cutoff) / self.cutoff) + 1.0)
        return basis, env


# ---------------------------------------------------------------------------
# Backbone convolutions
# ---------------------------------------------------------------------------

#: Vector-invariant probes. ``smooth`` is the default; ``legacy`` reproduces the
#: numerically fragile formulation and exists only as an ablation.
PROBES = ("smooth", "legacy")


class EquivariantConv(nn.Module):
    """PaiNN-inspired scalar-vector message passing with invariant read-back.

    Scalars are O(3)-invariant and vectors O(3)-equivariant: every quantity that
    enters the message MLP is an inner product, a norm or a radial function.
    Vector channels are not mixed inside the layer (no PaiNN U/V update block);
    this is why the paper calls it PaiNN-*inspired* rather than PaiNN.

    [R3-2] ``probe='smooth'`` (default) uses::

        c_ij = <v_i,v_j> / (sqrt(|v_i|^2+d) sqrt(|v_j|^2+d))
        n    = sqrt(|v|^2+d) - sqrt(d)

    which is bounded, Lipschitz and insensitive to rounding residuals smaller
    than ``sqrt(d)``. ``probe='legacy'`` reproduces
    ``<v_i,v_j>/(|v_i||v_j|+1e-6)``, whose value at a symmetry-cancelled site is
    the direction of a float32 rounding residual.
    """

    def __init__(self, channels: int, num_rbf: int = 16, cutoff: float = 5.0,
                 vector_norm: bool = False, probe: str = "smooth", vec_eps: float = 1e-2):
        super().__init__()
        if probe not in PROBES:
            raise ValueError(f"probe must be one of {PROBES}")
        self.C = channels
        self.probe = probe
        self.vec_eps = float(vec_eps)
        ln_eps = 1e-3 if probe == "smooth" else 1e-5
        self.rbf = RBF(num_rbf, cutoff)
        self.inv_norm = nn.LayerNorm(3 * channels, eps=ln_eps)
        self.msg = nn.Sequential(
            nn.Linear(5 * channels + num_rbf, channels), nn.SiLU(),
            nn.Linear(channels, 3 * channels),
        )
        self.vn_norm = nn.LayerNorm(channels, eps=ln_eps)
        self.upd_s = nn.Sequential(nn.Linear(2 * channels, channels), nn.SiLU())
        self.norm = nn.LayerNorm(channels)
        self.vector_norm = vector_norm
        if vector_norm:
            self.v_gain = nn.Parameter(torch.zeros(channels))

    def _vnorm(self, v, mask):
        n = safe_norm(v)
        rms = n.pow(2).mean(-1, keepdim=True).clamp(min=1e-8).sqrt()
        gain = F.softplus(self.v_gain).view(1, 1, -1, 1)
        return (v * gain / rms.unsqueeze(-1)) * mask.unsqueeze(-1).unsqueeze(-1)

    def _invariants(self, v_i, v_j):
        if self.probe == "legacy":
            n_i, n_j = safe_norm(v_i), safe_norm(v_j)
            cos_ij = (v_i * v_j).sum(-1) / (n_i * n_j + 1e-6)
            return cos_ij, n_i, n_j
        d = self.vec_eps
        r_i = (v_i.pow(2).sum(-1) + d).sqrt()
        r_j = (v_j.pow(2).sum(-1) + d).sqrt()
        cos_ij = (v_i * v_j).sum(-1) / (r_i * r_j)
        return cos_ij, r_i - math.sqrt(d), r_j - math.sqrt(d)

    def _vec_scalar(self, v):
        if self.probe == "legacy":
            return safe_norm(v)
        return smooth_norm(v, self.vec_eps)

    def forward(self, s, v, edges, mask):
        B, A, C = s.shape
        src, dst, dvec, dist, em = edges
        E = src.size(1)
        idx_s = src.unsqueeze(-1).expand(-1, -1, C)
        idx_d = dst.unsqueeze(-1).expand(-1, -1, C)
        s_i = torch.gather(s, 1, idx_s)
        s_j = torch.gather(s, 1, idx_d)
        vsel = src.view(B, E, 1, 1).expand(-1, -1, C, 3)
        dsel = dst.view(B, E, 1, 1).expand(-1, -1, C, 3)
        v_i = torch.gather(v, 1, vsel)
        v_j = torch.gather(v, 1, dsel)

        basis, env = self.rbf(dist)
        cos_ij, n_i, n_j = self._invariants(v_i, v_j)
        inv = self.inv_norm(torch.cat([cos_ij, n_i, n_j], -1))

        m = self.msg(torch.cat([s_i, s_j, basis, inv], -1))
        ms, mv1, mv2 = m.split(C, -1)

        w = (em.to(s.dtype) * env).unsqueeze(-1)
        unit = dvec / dist.clamp(min=1e-8).unsqueeze(-1)

        s_agg = torch.zeros_like(s).scatter_add_(1, idx_s, ms * w)
        v_msg = (mv1 * w).unsqueeze(-1) * v_j + (mv2 * w).unsqueeze(-1) * unit.unsqueeze(2)
        v_agg = torch.zeros_like(v).scatter_add_(1, vsel, v_msg)

        v = (v + v_agg) * mask.unsqueeze(-1).unsqueeze(-1)
        if self.vector_norm:
            v = self._vnorm(v, mask)
        s_from_v = self.vn_norm(self._vec_scalar(v))
        s = self.norm(s + self.upd_s(torch.cat([s_agg, s_from_v], -1))) * mask.unsqueeze(-1)
        return s, v


class CGConv(nn.Module):
    """CGCNN-style gated convolution (scalars only).

    LayerNorm instead of the original BatchNorm because tensors are padded;
    Gaussian basis as in :class:`RBF`.
    """

    def __init__(self, channels: int, edge_dim: int):
        super().__init__()
        self.lin = nn.Linear(2 * channels + edge_dim, 2 * channels)
        self.ln_msg = nn.LayerNorm(2 * channels)
        self.ln_out = nn.LayerNorm(channels)
        self.C = channels

    def forward(self, s, src, dst, efeat, emask, env=None):
        B, A, C = s.shape
        idx_s = src.unsqueeze(-1).expand(-1, -1, C)
        idx_d = dst.unsqueeze(-1).expand(-1, -1, C)
        s_i = torch.gather(s, 1, idx_s)
        s_j = torch.gather(s, 1, idx_d)
        z = self.ln_msg(self.lin(torch.cat([s_i, s_j, efeat], -1)))
        gate, core = z.split(C, -1)
        w = emask.to(s.dtype)
        if env is not None:
            w = w * env
        m = torch.sigmoid(gate) * F.softplus(core) * w.unsqueeze(-1)
        agg = self.ln_out(torch.zeros_like(s).scatter_add_(1, idx_s, m))
        return F.softplus(s + agg)


#: Alias kept so older checkpoints / notebooks referring to ``_CGConv`` still work.
_CGConv = CGConv


# ---------------------------------------------------------------------------
# Capsules
# ---------------------------------------------------------------------------


class PrimaryCapsules(nn.Module):
    """Per-atom primary capsules.  [R3-1][R3-3]

    Scalar-only backbones: ``u_i = W s_i``.

    Scalar-vector backbone: ``u_i = [W [s_i, chi_i], LN(n(v_i M))]``, where ``n``
    is the smooth norm and ``chi_i`` are optional pseudoscalars (normalised
    triple products of learned vector mixtures). ``chirality=False`` (default)
    drops ``chi``, making the capsule parity-even and the whole predictor
    O(3)-invariant.
    """

    def __init__(self, scalar_c: int, vector_c: int, prim_dim: int,
                 chirality: bool = False, n_chirality: int = 2, vec_eps: float = 1e-2):
        super().__init__()
        self.use_vec = vector_c > 0
        self.chirality = bool(chirality) and self.use_vec
        self.vec_eps = float(vec_eps)
        nc = n_chirality if self.chirality else 0
        self.nc = nc
        if self.use_vec:
            half = prim_dim // 2
            self.scalar_proj = nn.Linear(scalar_c + nc, prim_dim - half)
            self.vec_mix = nn.Parameter(torch.randn(vector_c, half) / math.sqrt(vector_c))
            self.norm = nn.LayerNorm(half, eps=1e-3)
            if self.chirality:
                self.chiral = nn.Parameter(torch.randn(3, nc, vector_c) / math.sqrt(vector_c))
        else:
            self.scalar_proj = nn.Linear(scalar_c, prim_dim)

    def forward(self, s, v=None):
        if not self.use_vec:
            return self.scalar_proj(s)
        feats = [s]
        if self.chirality:
            uvw = torch.einsum("knc,bacd->baknd", self.chiral, v)
            u, w_, x = uvw[:, :, 0], uvw[:, :, 1], uvw[:, :, 2]
            d = self.vec_eps
            trip = (torch.cross(u, w_, dim=-1) * x).sum(-1)
            den = ((u.pow(2).sum(-1) + d).sqrt() * (w_.pow(2).sum(-1) + d).sqrt()
                   * (x.pow(2).sum(-1) + d).sqrt())
            feats.append(trip / den)
        sp = self.scalar_proj(torch.cat(feats, -1))
        mixed = torch.einsum("bacd,ch->bahd", v, self.vec_mix)
        inv = self.norm(smooth_norm(mixed, self.vec_eps))
        return torch.cat([sp, inv], -1)


class MotifCapsules(nn.Module):
    """Dynamic routing by agreement (Sabour et al. 2017) over the atoms of a cell.  [R3-9]

    With zero initial logits the couplings are uniform, ``c_ij = 1/M``. Then::

        routing_norm='mass'  : s_j = sum_i c_ij u_j|i / sum_i c_ij = mean_i u_j|i
        routing_norm='atoms' : s_j = sum_i c_ij u_j|i / N          = mean_i u_j|i / M

    so at the FIRST iteration both schemes give capsules in the same directions
    and differ only by the scale factor ``1/M``. Any difference between them is a
    scale effect: it sets where the squash non-linearity operates and hence how
    large the agreement update to the logits is. A scale-matched control is
    ``routing_norm='mass'`` with ``temperature=M``. Later iterations differ
    genuinely: ``'mass'`` is a coupling-weighted mean per capsule, ``'atoms'``
    keeps the summed coupling mass inside ``s_j``. Both are invariant to
    supercell replication.
    """

    def __init__(self, prim_dim: int, n_caps: int, sec_dim: int, iters: int = 3,
                 routing_norm: str = "atoms", temperature: float = 1.0):
        super().__init__()
        if routing_norm not in ("atoms", "mass"):
            raise ValueError("routing_norm must be 'atoms' or 'mass'")
        self.n_caps, self.sec_dim, self.iters = int(n_caps), int(sec_dim), int(iters)
        self.routing_norm = routing_norm
        self.temperature = float(temperature)
        self.W = nn.Parameter(torch.randn(n_caps, sec_dim, prim_dim) / math.sqrt(prim_dim))
        self.bias = nn.Parameter(torch.zeros(n_caps, sec_dim))

    @staticmethod
    def squash(s, eps: float = 1e-8):
        n2 = (s ** 2).sum(-1, keepdim=True)
        n = (n2 + eps).sqrt()
        return (n2 / (1 + n2)) * s / n

    def forward(self, u, mask, return_assign: bool = False):
        B, A, P = u.shape
        u_hat = torch.einsum("cod,bad->baco", self.W, u)
        b = torch.zeros(B, A, self.n_caps, device=u.device, dtype=u.dtype)
        m = mask.to(u.dtype).unsqueeze(-1)
        n_atoms = mask.to(u.dtype).sum(1).clamp(min=1).view(B, 1, 1)
        scale = 1.0 / math.sqrt(self.sec_dim)
        c, vj, hist = None, None, []
        for _ in range(self.iters):
            c = F.softmax(b / self.temperature, dim=2) * m
            hist.append(c)
            num = (c.unsqueeze(-1) * u_hat).sum(1)
            if self.routing_norm == "mass":
                den = c.sum(1).clamp(min=1e-6).unsqueeze(-1)
            else:
                den = n_atoms
            vj = self.squash(num / den + self.bias)
            b = b + scale * (u_hat * vj.unsqueeze(1)).sum(-1)
        routing = c.sum(1)
        return (vj, routing, c, hist) if return_assign else (vj, routing)
