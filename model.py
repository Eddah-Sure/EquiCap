"""The crystal property predictor.

The point of this class is that **backbone and readout are independent factors**:
every readout is available for every backbone, so a factorial design can
attribute an effect to one or the other instead of to a bundled architecture.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .geometry import (NEIGHBOR_MODES, build_edges_from_geometry, edge_geometry,
                       matrix_to_lattice_params)
from .layers import PROBES, CGConv, EquivariantConv, MotifCapsules, PrimaryCapsules, RBF
from .readouts import AttentionPool, MeanMaxPool, RoutingWeightedPool
from .utils import _warn_once

__all__ = [
    "CrystalPropertyModel", "BACKBONES", "READOUTS", "GLOBAL_FEATURE_DIMS", "PROBES",
]

#: Dimensionality of each cell-level descriptor block.
GLOBAL_FEATURE_DIMS = {"none": 0, "density": 2, "intensive": 5, "full": 8}
BACKBONES = ("painn", "cgcnn", "composition")
READOUTS = ("attention", "capsule", "mean")


class CrystalPropertyModel(nn.Module):
    """Crystal property predictor with INDEPENDENT backbone and readout factors.

    Parameters
    ----------
    backbone : {'painn', 'cgcnn', 'composition'}
        ``painn`` scalar-vector message passing, ``cgcnn`` invariant gated
        convolution, ``composition`` structure-blind per-atom MLP.
    readout : {'attention', 'mean', 'capsule'}
        ``attention`` / ``mean`` pool the N atom states; ``capsule`` runs primary
        capsules + dynamic routing to ``n_caps`` motif capsules, then a
        routing-mass-weighted mean+max over capsules. Every readout is available
        for every backbone.  [R3-1]
    global_features : {'none', 'density', 'intensive', 'full'}
        ``none`` pooled representation only; ``density`` ``log(1+V/N)`` and
        ``N/V`` (lattice + atom count only — the only mode allowed for the
        composition control); ``intensive`` density + mean/std edge length + mean
        degree; ``full`` legacy, NOT supercell-invariant (ablation only).
    neighbor_mode : {'radius', 'knn'}  [R3-4]
    probe : {'smooth', 'legacy'}
        Vector invariants.  [R3-2]
    chirality : bool
        Include pseudoscalar capsule features (breaks O(3) -> SO(3)).

    Notes
    -----
    Symmetry contract, verified by :func:`equicap.audit.symmetry_audit` with
    fixed AND rebuilt edges: invariant to rotation, reflection (if
    ``chirality=False``), translation, permutation, periodic-image choice, and
    supercell replication (if ``global_features != 'full'`` and the radius cap
    does not bind).
    """

    def __init__(
        self,
        num_types: int,
        elem_feat_dim: int = 0,
        backbone: str = "painn",
        readout: str = "attention",
        hidden: int = 128,
        n_layers: int = 3,
        num_rbf: int = 16,
        cutoff: float = 5.0,
        max_deg: int = 64,
        neighbor_mode: str = "radius",
        prim_dim: int = 32,
        n_caps: int = 8,
        sec_dim: int = 64,
        routing_iters: int = 3,
        routing_norm: str = "atoms",
        routing_temperature: float = 1.0,
        chirality: bool = False,
        probe: str = "smooth",
        vec_eps: float = 1e-2,
        extensive: bool = False,
        dropout: float = 0.10,
        global_features: Union[str, bool] = "intensive",
    ):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone!r}")
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
        if neighbor_mode not in NEIGHBOR_MODES:
            raise ValueError(f"neighbor_mode must be one of {NEIGHBOR_MODES}")
        if isinstance(global_features, bool):
            global_features = "full" if global_features else "none"
        if global_features not in GLOBAL_FEATURE_DIMS:
            raise ValueError(f"global_features must be one of {list(GLOBAL_FEATURE_DIMS)}")
        if backbone == "composition" and global_features in ("intensive", "full"):
            _warn_once("backbone='composition' cannot use edge/lattice-shape "
                       "descriptors; global_features set to 'none' "
                       "(use 'density' for the descriptor-matched control)")
            global_features = "none"

        self.num_types = int(num_types)
        self.elem_feat_dim = int(elem_feat_dim)
        self.backbone = backbone
        self.readout_type = readout
        self.hidden = int(hidden)
        self.n_layers = int(n_layers)
        self.num_rbf = int(num_rbf)
        self.cutoff = float(cutoff)
        self.max_deg = int(max_deg)
        self.neighbor_mode = neighbor_mode
        self.prim_dim = int(prim_dim)
        self.n_caps = int(n_caps)
        self.sec_dim = int(sec_dim)
        self.routing_iters = int(routing_iters)
        self.routing_norm = routing_norm
        self.routing_temperature = float(routing_temperature)
        self.chirality = bool(chirality) and backbone == "painn" and readout == "capsule"
        self.probe = probe
        self.vec_eps = float(vec_eps)
        self.extensive = bool(extensive)
        self.dropout = float(dropout)
        self.global_features = global_features

        # --- embedding ---
        self.embed = nn.Embedding(num_types, hidden)
        if elem_feat_dim > 0:
            self.elem_proj = nn.Linear(elem_feat_dim, hidden)
            self.register_buffer("elem_feat_table", torch.zeros(num_types, elem_feat_dim))

        # --- backbone ---
        if backbone == "painn":
            self.convs = nn.ModuleList([
                EquivariantConv(hidden, num_rbf, cutoff, probe=probe, vec_eps=vec_eps)
                for _ in range(n_layers)])
        elif backbone == "cgcnn":
            self.rbf = RBF(num_rbf, cutoff)
            self.convs = nn.ModuleList([CGConv(hidden, num_rbf) for _ in range(n_layers)])
        else:
            self.comp_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                          nn.Linear(hidden, hidden))

        # --- readout ---  [R3-1]
        if readout == "capsule":
            self.primary = PrimaryCapsules(
                hidden, hidden if backbone == "painn" else 0, prim_dim,
                chirality=self.chirality, vec_eps=vec_eps)
            self.secondary = MotifCapsules(prim_dim, n_caps, sec_dim, iters=routing_iters,
                                           routing_norm=routing_norm,
                                           temperature=routing_temperature)
            self.readout = RoutingWeightedPool(sec_dim)
        elif readout == "attention":
            self.readout = AttentionPool(hidden)
        else:
            self.readout = MeanMaxPool(hidden)

        gf_dim = GLOBAL_FEATURE_DIMS[global_features]
        self.head = nn.Sequential(
            nn.Linear(self.readout.out_dim + gf_dim, hidden), nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden // 2, 1),
        )

        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))
        self.register_buffer("fitted", torch.tensor(0, dtype=torch.long))

    # ------------------------------------------------------------------
    # Target standardisation / element table
    # ------------------------------------------------------------------

    @torch.no_grad()
    def set_y_stats(self, mean: float, std: float):
        self.y_mean.fill_(float(mean))
        self.y_std.fill_(max(float(std), 1e-6))

    @property
    def y_stats(self) -> Tuple[float, float]:
        return float(self.y_mean.item()), float(self.y_std.item())

    @torch.no_grad()
    def set_elem_feat(self, table):
        if self.elem_feat_dim <= 0:
            return
        t = torch.as_tensor(np.asarray(table, dtype=np.float32))
        if t.shape != self.elem_feat_table.shape:
            raise ValueError(f"elem_feat table is {tuple(t.shape)}, expected "
                             f"{tuple(self.elem_feat_table.shape)}")
        self.elem_feat_table.copy_(t.to(self.elem_feat_table.device,
                                        self.elem_feat_table.dtype))

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _global_feats(self, frac, L, mask, dist=None, emask=None):
        """Cell-level descriptors appended to the pooled representation."""
        dt = L.dtype
        n_atoms = mask.to(dt).sum(1)
        vol = torch.linalg.det(L).abs().clamp(min=1e-6)
        vpa = vol / n_atoms.clamp(min=1)
        dens = [vpa.log1p(), 1.0 / vpa.clamp(min=1e-3)]

        if self.global_features == "density":
            return torch.stack(dens, -1)

        if self.global_features == "full":
            lp = matrix_to_lattice_params(L)
            abc = lp[:, :3].sort(-1).values
            ang = lp[:, 3:].sort(-1).values
            return torch.stack([
                n_atoms.log1p(), vpa.log1p(),
                abc[:, 0] / 10.0, abc[:, 1] / 10.0, abc[:, 2] / 10.0,
                ang[:, 0] / 180.0, ang[:, 1] / 180.0, ang[:, 2] / 180.0], -1)

        if dist is None or emask is None:
            z = torch.zeros_like(vpa)
            d_mean = d_std = deg = z
        else:
            w = emask.to(dt)
            n_e = w.sum(1).clamp(min=1)
            d_mean = (dist * w).sum(1) / n_e
            d_var = ((dist - d_mean.unsqueeze(1)) ** 2 * w).sum(1) / n_e
            d_std = d_var.clamp(min=0).sqrt()
            deg = w.sum(1) / n_atoms.clamp(min=1)
        return torch.stack(dens + [d_mean / 5.0, d_std / 2.0, deg / float(self.max_deg)], -1)

    def _embed_atoms(self, types, mask, elem_feat=None, type_probs=None):
        ef = elem_feat if elem_feat is not None else getattr(self, "elem_feat_table", None)
        if type_probs is not None:
            s = type_probs @ self.embed.weight
            if self.elem_feat_dim > 0 and ef is not None:
                s = s + self.elem_proj(type_probs @ ef.to(s.dtype))
        else:
            s = self.embed(types)
            if self.elem_feat_dim > 0 and ef is not None:
                s = s + self.elem_proj(ef.to(s.dtype)[types])
        return s * mask.unsqueeze(-1).to(s.dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        types: torch.Tensor,
        frac: torch.Tensor,
        L: torch.Tensor,
        mask: torch.Tensor,
        edges: Optional[Tuple[torch.Tensor, ...]] = None,
        elem_feat: Optional[torch.Tensor] = None,
        type_probs: Optional[torch.Tensor] = None,
        denorm: bool = True,
        return_latent: bool = False,
    ):
        """Predict the property for a padded batch of crystals.

        ``edges``: ``(src, dst, offset, emask)``. If None, the graph is built from
        geometry with THIS model's ``cutoff`` / ``max_deg`` / ``neighbor_mode``,
        which is the same builder the dataset uses, so supplied and rebuilt
        graphs coincide.
        """
        B, A = types.shape[:2]
        if types.numel() and int(types.max()) >= self.num_types:
            raise IndexError(f"atom type index {int(types.max())} >= num_types "
                             f"{self.num_types}; vocabulary mismatch")
        mb = mask.bool()
        mf = mb.to(L.dtype)
        s = self._embed_atoms(types, mb, elem_feat, type_probs)
        latent: Dict[str, object] = {}
        v, dist, emask = None, None, None

        # --- backbone -------------------------------------------------------
        if self.backbone == "composition":
            s = self.comp_mlp(s) * mf.unsqueeze(-1)
        else:
            if edges is None:
                src, dst, offset, emask = build_edges_from_geometry(
                    frac, L, mb, cutoff=self.cutoff, max_deg=self.max_deg,
                    mode=self.neighbor_mode)
            else:
                src, dst, offset, emask = edges
            dvec, dist = edge_geometry(frac, L, src, dst, offset)
            if self.backbone == "painn":
                v = torch.zeros(B, A, self.hidden, 3, device=s.device, dtype=s.dtype)
                et = (src, dst, dvec, dist, emask)
                for conv in self.convs:
                    s, v = conv(s, v, et, mf)
            else:
                basis, env = self.rbf(dist)
                for conv in self.convs:
                    s = conv(s, src, dst, basis, emask, env) * mf.unsqueeze(-1)

        # --- readout --------------------------------------------------------
        if self.readout_type == "capsule":
            u = self.primary(s, v) * mf.unsqueeze(-1)
            caps, rho, c, hist = self.secondary(u, mb, return_assign=True)
            pooled_repr, pw = self.readout(caps, None, rho, return_weights=True)
            if return_latent:
                latent.update(primary=u, capsules=caps, routing_mass=rho,
                              coupling=c, coupling_history=hist)
        else:
            pooled_repr, pw = self.readout(s, mb, None, return_weights=True)

        pooled = pooled_repr
        if self.global_features != "none":
            pooled = torch.cat([pooled_repr,
                                self._global_feats(frac, L, mb, dist, emask)], -1)
        y = self.head(pooled).squeeze(-1)

        if return_latent:
            latent.update(nodes=s, vectors=v, pooled=pooled, pooled_repr=pooled_repr,
                          pool_weights=pw, edge_dist=dist, edge_mask=emask)

        # de-standardise first, then per-atom -> total
        if denorm:
            y = y * self.y_std + self.y_mean
            if self.extensive:
                y = y * mf.sum(1)
        return (y, latent) if return_latent else y

    # ------------------------------------------------------------------
    # Config / introspection
    # ------------------------------------------------------------------

    _CONFIG_KEYS = ("num_types", "elem_feat_dim", "backbone", "readout", "hidden",
                    "n_layers", "num_rbf", "cutoff", "max_deg", "neighbor_mode",
                    "prim_dim", "n_caps", "sec_dim", "routing_iters", "routing_norm",
                    "routing_temperature", "chirality", "probe", "vec_eps",
                    "extensive", "dropout", "global_features")

    def get_config(self) -> dict:
        cfg = {k: getattr(self, k) for k in self._CONFIG_KEYS if k != "readout"}
        cfg["readout"] = self.readout_type
        return cfg

    @classmethod
    def from_config(cls, cfg: dict) -> "CrystalPropertyModel":
        cfg = {k: v for k, v in dict(cfg).items() if k in cls._CONFIG_KEYS}
        return cls(**cfg)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def summary(self):
        tot = self.n_params()
        print(f"CrystalPropertyModel  backbone={self.backbone}  readout={self.readout_type}")
        print(f"  graph: {self.neighbor_mode}, cutoff={self.cutoff} A, cap={self.max_deg}")
        print(f"  hidden={self.hidden}  n_layers={self.n_layers}  n_caps={self.n_caps}"
              f"  routing_norm={self.routing_norm}  T={self.routing_temperature}")
        print(f"  probe={self.probe}  chirality={self.chirality}  "
              f"global_features={self.global_features}  extensive={self.extensive}")
        print(f"  params: {tot:,}")
