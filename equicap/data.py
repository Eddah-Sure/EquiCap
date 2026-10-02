"""Dataset, collation and the shared element vocabulary.

Expected layout under ``data_root``::

    data_root/
      train/  train_config.json  train.npz  train.csv
      val/    val_config.json    val.npz    val.csv
      test/   test_config.json   test.npz   test.csv

``*.npz`` holds a pickled ``graph_dict`` mapping ``material_id -> structure``;
``*.csv`` holds the targets and MUST have a ``material_id`` column, because
positional alignment of targets to structures is not safe.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .chemistry import build_element_phys_features
from .geometry import structure_edges
from .utils import _warn_once

__all__ = [
    "CrystalGraph", "CrystalPropertyDataset", "crystal_collate",
    "build_global_vocabulary", "EDGE_SOURCES", "CORRUPT_MODES",
]

EDGE_SOURCES = ("rebuild", "stored")
CORRUPT_MODES = ("none", "coords", "edges", "lattice")


class CrystalGraph:
    """Single-crystal record parsed from the mined NPZ."""

    def __init__(self, gd: dict, name: str = "?"):
        self.name = name
        self.nodes = np.array(gd["node_features"], dtype=np.int32)
        self.cart_coords = np.array(gd["cart_coords"], dtype=np.float64)

        lm = np.array(gd.get("lattice_matrix", np.eye(3)), dtype=np.float64)
        if lm.ndim == 3:
            lm = lm[0]
        if lm.size == 9:
            lm = lm.reshape(3, 3)
        elif lm.size == 3:
            lm = np.diag(lm.ravel())
        else:
            raise ValueError("unparseable lattice_matrix")
        self.lattice_matrix = lm

        if "frac_coords" in gd:
            self.frac_coords = np.array(gd["frac_coords"], dtype=np.float64)
        else:
            self.frac_coords = self.cart_coords @ np.linalg.inv(lm)

        self.space_group = int(np.asarray(gd.get("space_group", 1)).ravel()[0])

        # stored graph (only used when edge_source='stored')
        ei = gd.get("edge_index", None)
        ei_arr = np.asarray(ei) if ei is not None else None
        self.stored_ok = ei_arr is not None and ei_arr.ndim == 2 and ei_arr.shape[0] == 2
        off_raw = gd.get("edge_offsets", None)
        self.offsets_present = off_raw is not None and np.asarray(off_raw).size > 0
        if self.stored_ok:
            ei_arr = ei_arr.astype(np.int64)
            off = (np.asarray(off_raw, dtype=np.float32).reshape(-1, 3)
                   if self.offsets_present else np.zeros((ei_arr.shape[1], 3), np.float32))
            if ei_arr.shape[1] > 0:
                keep = ~((ei_arr[0] == ei_arr[1]) & (np.abs(off).sum(-1) == 0))
                ei_arr, off = ei_arr[:, keep], off[keep]
            self.stored_edge_index, self.stored_edge_offsets = ei_arr, off
        # active graph, filled by the dataset
        self.edge_index = np.zeros((2, 0), np.int64)
        self.edge_offsets = np.zeros((0, 3), np.float32)

    @property
    def n_atoms(self) -> int:
        return len(self.nodes)


class CrystalPropertyDataset(Dataset):
    """NPZ + CSV crystal property dataset.

    Parameters
    ----------
    edge_source : {'rebuild', 'stored'}
        ``rebuild`` (default): the neighbour graph is built ONCE per structure
        with :func:`~equicap.geometry.build_edges_from_geometry` (float64,
        independent of batch composition) and reused for every epoch, evaluation
        and audit.  [R3-5]
        ``stored``: use the mined edge list — only after
        :func:`~equicap.audit.audit_edges` passes.
    corrupt : {'none', 'coords', 'edges', 'lattice'}
        Retrained structure-ablation control, drawn ONCE per structure from a
        seeded RNG and applied to every split.  [R3-7]
        ``coords`` uniform random fractional coordinates (graph rebuilt);
        ``lattice`` isotropic cubic cell at the same volume (graph rebuilt);
        ``edges`` degree-preserving rewiring of edge targets with minimum-image
        offsets recomputed for the new pairs.
    cache_dir : str, optional
        If given, the per-structure graphs are cached there, keyed by a hash of
        every setting that can change them.
    """

    def __init__(
        self,
        base_path: str,
        target_name: str,
        split: str,
        global_atomic_to_idx: Optional[Dict[int, int]] = None,
        global_node_vectors: Optional[np.ndarray] = None,
        max_atoms: Optional[int] = None,
        edge_source: str = "rebuild",
        cutoff: float = 5.0,
        max_deg: int = 64,
        neighbor_mode: str = "radius",
        corrupt: str = "none",
        seed: int = 0,
        cache_dir: Optional[str] = None,
        rebuild_edges: Optional[bool] = None,
        verbose: bool = True,
    ):
        super().__init__()
        if rebuild_edges is not None:                 # backward compatibility
            edge_source = "rebuild" if rebuild_edges else "stored"
        if edge_source not in EDGE_SOURCES:
            raise ValueError(f"edge_source must be one of {EDGE_SOURCES}")
        if corrupt not in CORRUPT_MODES:
            raise ValueError(f"unknown corrupt mode {corrupt!r}")
        self.split, self.path = split, os.path.join(base_path, split)
        self.target_name = target_name
        self.global_atomic_to_idx = global_atomic_to_idx
        self.global_node_vectors = global_node_vectors
        self._max_atoms = max_atoms
        self.edge_source = edge_source
        self.cutoff, self.max_deg = float(cutoff), int(max_deg)
        self.neighbor_mode = neighbor_mode
        self.corrupt = corrupt
        self.seed = int(seed)
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.n_overflow_atoms = 0

        self._load_config()
        self._load_graph_data()
        self._apply_max_atoms_filter()
        self._load_targets()
        self._apply_corruption_and_edges()
        if len(self.graph_data) != len(self.targets):
            raise ValueError("graph/target count mismatch after alignment")
        if verbose:
            self._report()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_config(self):
        with open(os.path.join(self.path, f"{self.split}_config.json")) as f:
            cfg = json.load(f)
        if self.global_atomic_to_idx is not None:
            self.atomic_numbers = sorted(self.global_atomic_to_idx.keys())
            self.atomic_to_idx = self.global_atomic_to_idx
        else:
            self.atomic_numbers = cfg["atomic_numbers"]
            self.atomic_to_idx = {n: i for i, n in enumerate(self.atomic_numbers)}
        n_types = len(self.atomic_numbers)
        if self.global_node_vectors is not None:
            gv = np.asarray(self.global_node_vectors, dtype=np.float32)
            if gv.shape[0] != n_types:
                raise ValueError("global_node_vectors has the wrong number of rows")
            self.node_vectors = gv
            return
        elem_props = cfg.get("element_properties", {})
        self.node_vectors = np.concatenate(
            [np.eye(n_types, dtype=np.float32),
             build_element_phys_features(self.atomic_numbers, elem_props)], axis=1)

    def _load_graph_data(self):
        with np.load(os.path.join(self.path, f"{self.split}.npz"), allow_pickle=True) as data:
            gd = data["graph_dict"].item()
        self.graph_data, self.graph_names = [], []
        skipped = 0
        for name, g in gd.items():
            try:
                if "cart_coords" not in g:
                    raise ValueError("missing cart_coords")
                cg = CrystalGraph(g, name=name)
                if self.edge_source == "stored" and not cg.stored_ok:
                    raise ValueError("no stored edge_index")
                self.graph_data.append(cg)
                self.graph_names.append(name)
            except ValueError:
                skipped += 1
        if not self.graph_data:
            raise RuntimeError(f"no valid graphs loaded from {self.path}")
        self.n_skipped = skipped

    def _apply_max_atoms_filter(self):
        if self._max_atoms is None:
            return
        keep = [(g, n) for g, n in zip(self.graph_data, self.graph_names)
                if g.n_atoms <= self._max_atoms]
        if not keep:
            raise RuntimeError("max_atoms filtered out every crystal in this split")
        self.graph_data, self.graph_names = [k[0] for k in keep], [k[1] for k in keep]

    def _load_targets(self):
        df = pd.read_csv(os.path.join(self.path, f"{self.split}.csv"))
        if self.target_name not in df.columns:
            raise ValueError(f"target {self.target_name!r} not in {self.split}.csv; "
                             f"available: {list(df.columns)}")
        if "material_id" not in df.columns:
            raise ValueError(f"{self.split}.csv has no 'material_id' column; "
                             f"positional alignment of targets is unsafe")
        idx = df.drop_duplicates("material_id").set_index("material_id")
        missing = [n for n in self.graph_names if n not in idx.index]
        if missing:
            raise ValueError(f"{len(missing)} graphs absent from CSV; first: {missing[0]}")
        aligned = idx.loc[self.graph_names]
        self.targets = aligned[self.target_name].values.astype(np.float32)
        self.material_ids = np.asarray(aligned.index.values)
        self.space_groups = np.asarray([g.space_group for g in self.graph_data])
        if "space_group" in aligned.columns:
            self.space_groups = aligned["space_group"].values.astype(np.int64)

    # ------------------------------------------------------------------
    # Graph construction / corruption
    # ------------------------------------------------------------------

    def _cache_path(self) -> Optional[str]:
        if not self.cache_dir:
            return None
        key = json.dumps([self.split, self.edge_source, self.cutoff, self.max_deg,
                          self.neighbor_mode, self.corrupt, self.seed,
                          self._max_atoms], sort_keys=True)
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return os.path.join(self.cache_dir, f"graphs_{self.split}_{h}.pkl")

    def _apply_corruption_and_edges(self):
        cp = self._cache_path()
        if cp and os.path.exists(cp):
            with open(cp, "rb") as fh:
                cached = pickle.load(fh)
            if cached.get("names") == list(self.graph_names):
                for g, rec in zip(self.graph_data, cached["records"]):
                    g.frac_coords, g.lattice_matrix, g.edge_index, g.edge_offsets = rec
                self.n_overflow_atoms = cached["n_overflow_atoms"]
                return
        rng = np.random.default_rng(self.seed)
        n_ovf = 0
        if self.verbose:
            print(f"  [{self.split}] building {self.neighbor_mode} graphs "
                  f"(cutoff {self.cutoff} A, cap {self.max_deg}) ...", flush=True)
        for g in self.graph_data:
            if self.corrupt == "coords":
                g.frac_coords = rng.random(g.frac_coords.shape)
            elif self.corrupt == "lattice":
                V = abs(float(np.linalg.det(g.lattice_matrix)))
                g.lattice_matrix = np.eye(3) * V ** (1.0 / 3.0)
            if self.edge_source == "stored":
                g.edge_index = g.stored_edge_index
                g.edge_offsets = g.stored_edge_offsets
            else:
                ei, of, ovf = structure_edges(g.frac_coords, g.lattice_matrix,
                                              self.cutoff, self.max_deg, self.neighbor_mode)
                g.edge_index, g.edge_offsets = ei, of
                n_ovf += ovf
            if self.corrupt == "edges" and g.edge_index.shape[1] > 0:
                ei = g.edge_index.copy()
                ei[1] = rng.permutation(ei[1])
                df = g.frac_coords[ei[1]] - g.frac_coords[ei[0]]
                g.edge_index = ei
                g.edge_offsets = (-np.round(df)).astype(np.float32)
        self.n_overflow_atoms = n_ovf
        if n_ovf:
            _warn_once(f"[{self.split}] {n_ovf} atoms have more than {self.max_deg} "
                       f"neighbours inside {self.cutoff} A; their neighbourhoods are "
                       f"truncated (raise max_deg or lower the cutoff)")
        if cp:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cp, "wb") as fh:
                pickle.dump({"names": list(self.graph_names), "n_overflow_atoms": n_ovf,
                             "records": [(g.frac_coords, g.lattice_matrix, g.edge_index,
                                          g.edge_offsets) for g in self.graph_data]}, fh)

    def _report(self):
        n = len(self.graph_data)
        na = np.array([g.n_atoms for g in self.graph_data])
        ne = np.array([g.edge_index.shape[1] for g in self.graph_data])
        y = self.targets
        print(f"  [{self.split}] {n} structures | atoms {na.mean():.1f}+-{na.std():.1f} "
              f"(max {na.max()}) | degree {ne.sum() / max(na.sum(), 1):.1f} "
              f"| {self.target_name} {y.mean():.4f}+-{y.std():.4f} "
              f"| MAD {np.abs(y - y.mean()).mean():.4f}"
              + (f" | corrupt={self.corrupt}" if self.corrupt != "none" else "")
              + (f" | cap hit on {self.n_overflow_atoms} atoms" if self.n_overflow_atoms else ""))

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def composition_stats(self) -> Dict[str, float]:
        """Structures vs distinct reduced compositions (polymorph multiplicity).

        A ratio near 1.0 means a composition-only baseline can in principle match
        a structural model, which is the number to report before claiming that
        structure helps.
        """
        keys = set()
        for g in self.graph_data:
            z, c = np.unique(g.nodes, return_counts=True)
            gcd = int(np.gcd.reduce(c)) or 1
            keys.add(tuple(sorted(zip(z.tolist(), (c // gcd).tolist()))))
        return {"n_structures": len(self.graph_data), "n_compositions": len(keys),
                "structures_per_composition": len(self.graph_data) / max(len(keys), 1)}

    def __len__(self):
        return len(self.graph_data)

    def __getitem__(self, i):
        g = self.graph_data[i]
        ai = np.array([self.atomic_to_idx[int(n)] for n in g.nodes])
        return {
            "types": torch.tensor(ai, dtype=torch.long),
            "edge_index": torch.tensor(g.edge_index, dtype=torch.long),
            "edge_offsets": torch.tensor(g.edge_offsets, dtype=torch.float32),
            "y": torch.tensor([self.targets[i]], dtype=torch.float32),
            "material_id": str(self.material_ids[i]),
            "lattice_matrix": torch.tensor(g.lattice_matrix, dtype=torch.float32),
            "frac_coords": torch.tensor(g.frac_coords, dtype=torch.float32),
            "space_group": torch.tensor([int(self.space_groups[i])], dtype=torch.long),
            "n_atoms": g.n_atoms,
        }


def crystal_collate(batch):
    """Pad a list of crystal dicts into dense ``(B, A)`` / ``(B, E)`` tensors.

    Edges are taken as given — the dataset already built them once per structure,
    so batching can never change the graph.
    """
    B = len(batch)
    A_max = max(len(b["types"]) for b in batch)
    E_max = max(1, max(b["edge_index"].size(1) for b in batch))

    types = torch.zeros(B, A_max, dtype=torch.long)
    mask = torch.zeros(B, A_max, dtype=torch.bool)
    frac = torch.zeros(B, A_max, 3)
    L = torch.stack([b["lattice_matrix"] for b in batch])
    y = torch.stack([b["y"] for b in batch])
    sg = torch.cat([b["space_group"] for b in batch])
    n_atoms = torch.tensor([len(b["types"]) for b in batch], dtype=torch.long)
    src = torch.zeros(B, E_max, dtype=torch.long)
    dst = torch.zeros(B, E_max, dtype=torch.long)
    offset = torch.zeros(B, E_max, 3)
    emask = torch.zeros(B, E_max, dtype=torch.bool)

    for i, b in enumerate(batch):
        a = len(b["types"])
        types[i, :a] = b["types"]
        mask[i, :a] = True
        frac[i, :a] = b["frac_coords"]
        e = b["edge_index"].size(1)
        if e > 0:
            src[i, :e] = b["edge_index"][0]
            dst[i, :e] = b["edge_index"][1]
            offset[i, :e] = b["edge_offsets"]
            emask[i, :e] = True

    return {"types": types, "frac": frac, "lattice": L, "mask": mask, "y": y,
            "material_ids": [b["material_id"] for b in batch], "n_atoms": n_atoms,
            "space_group": sg, "src": src, "dst": dst, "offset": offset, "emask": emask}


def build_global_vocabulary(data_root: str,
                            splits: Tuple[str, ...] = ("train", "val", "test")):
    """Union of atom types across splits + shared node feature matrix.

    Only the element vocabulary is shared; no target information crosses splits.
    """
    all_z, merged = set(), {}
    for split in splits:
        p = os.path.join(data_root, split, f"{split}_config.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            cfg = json.load(f)
        all_z.update(cfg["atomic_numbers"])
        for k, v in (cfg.get("element_properties", {}) or {}).items():
            merged.setdefault(str(k), v)
    zs = sorted(all_z)
    a2i = {n: i for i, n in enumerate(zs)}
    nv = np.concatenate([np.eye(len(zs), dtype=np.float32),
                         build_element_phys_features(zs, merged)], axis=1)
    return zs, a2i, nv
