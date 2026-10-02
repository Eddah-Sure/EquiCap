"""Experiment orchestration: one cell, and the full factorial grid.

:func:`main` runs exactly one cell of the design end to end — data, audits,
training, evaluation, checkpoint. :func:`run_factorial` iterates cells x seeds
and writes a tidy results table after every cell, so an interrupted sweep still
leaves usable output.
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .audit import audit_edges, symmetry_audit
from .checkpoint import save_checkpoint
from .config import (DEFAULT_CKPT_DIRNAME, DEFAULT_DATA_ROOT, MODEL_KEYS, SPLITS,
                     check_dataset_layout, default_config, run_tag)
from .data import CrystalPropertyDataset, build_global_vocabulary, crystal_collate
from .inference import collect_predictions
from .metrics import compute_metrics
from .model import BACKBONES, READOUTS, CrystalPropertyModel
from .training import train_model, validate
from .utils import maybe_mount_drive, set_seed, worker_init_fn

__all__ = ["make_datasets", "main", "run", "run_factorial", "DEFAULT_CONTROLS"]


#: Controls run in addition to the 3 x 3 backbone x readout grid. Each entry is a
#: set of overrides and a variant name (written into the checkpoint tag).
DEFAULT_CONTROLS = [
    # descriptor-matched structure-blind control and descriptor-free structural cells
    {"variant": "density", "backbone": "composition", "readout": "mean",
     "global_features": "density"},
    {"variant": "nodesc", "backbone": "cgcnn", "readout": "attention",
     "global_features": "none"},
    {"variant": "nodesc", "backbone": "painn", "readout": "attention",
     "global_features": "none"},
    # routing normalisation (scale-matched control uses T = n_caps)
    {"variant": "mass", "backbone": "painn", "readout": "capsule", "routing_norm": "mass"},
    {"variant": "massT", "backbone": "painn", "readout": "capsule", "routing_norm": "mass",
     "routing_temperature": "n_caps"},
    # numerics and parity
    {"variant": "legacyprobe", "backbone": "painn", "readout": "attention",
     "probe": "legacy"},
    {"variant": "chiral", "backbone": "painn", "readout": "capsule", "chirality": True},
    # graph construction
    {"variant": "knn12", "backbone": "cgcnn", "readout": "attention",
     "neighbor_mode": "knn", "max_deg": 12},
    {"variant": "knn12", "backbone": "painn", "readout": "attention",
     "neighbor_mode": "knn", "max_deg": 12},
]


def make_datasets(cfg: dict, a2i, nv, verbose: bool = True):
    """Build train / val / test datasets with identical graph settings."""
    kw = dict(global_atomic_to_idx=a2i, global_node_vectors=nv,
              max_atoms=cfg["max_atoms"], edge_source=cfg["edge_source"],
              cutoff=float(cfg["cutoff"]), max_deg=int(cfg["max_deg"]),
              neighbor_mode=cfg["neighbor_mode"], corrupt=cfg["corrupt"],
              seed=int(cfg["seed"]) if cfg["corrupt"] != "none" else 0,
              cache_dir=cfg.get("graph_cache_dir"), verbose=verbose)
    return tuple(CrystalPropertyDataset(cfg["data_root"], cfg["target"], sp, **kw)
                 for sp in SPLITS)


def main(config: dict):
    """Train and evaluate one cell. Returns ``(model, results)``."""
    cfg = default_config()
    cfg.update({k: v for k, v in (config or {}).items() if v is not None})
    if not cfg["data_root"]:
        raise ValueError("config['data_root'] is required")

    cfg["data_root"] = os.path.abspath(os.path.expanduser(str(cfg["data_root"])))
    check_dataset_layout(cfg["data_root"])
    ckpt_dir = cfg.get("ckpt_dir") or os.path.join(cfg["data_root"], DEFAULT_CKPT_DIRNAME)
    ckpt_dir = os.path.abspath(os.path.expanduser(str(ckpt_dir)))
    os.makedirs(ckpt_dir, exist_ok=True)
    cfg["ckpt_dir"] = ckpt_dir
    if not cfg.get("graph_cache_dir"):
        cfg["graph_cache_dir"] = os.path.join(ckpt_dir, "graph_cache")
    tag = run_tag(cfg)
    for key, suffix in (("ckpt", ".pt"), ("metrics_csv", "_metrics.csv"),
                        ("predictions_csv", "_test_predictions.csv")):
        p = cfg.get(key)
        if p and not os.path.isabs(p):
            cfg[key] = os.path.join(ckpt_dir, f"{tag}{suffix}")

    print(f"cell: {tag}\ndata root:   {cfg['data_root']}\ncheckpoints: {ckpt_dir}")
    set_seed(int(cfg["seed"]), deterministic=bool(cfg["deterministic"]))
    device = torch.device(cfg["device"])

    # --- data ---------------------------------------------------------------
    zs, a2i, nv = build_global_vocabulary(cfg["data_root"])
    num_types = len(zs)
    if num_types == 0:
        raise RuntimeError(f"no *_config.json under {cfg['data_root']}")
    elem_feat_dim = int(nv.shape[1] - num_types)

    tr_ds, val_ds, te_ds = make_datasets(cfg, a2i, nv)
    comp = tr_ds.composition_stats()
    print(f"train: {comp['n_structures']} structures / {comp['n_compositions']} "
          f"compositions = {comp['structures_per_composition']:.2f} per formula")
    graph_stats = {sp: int(d.n_overflow_atoms) for sp, d in zip(SPLITS, (tr_ds, val_ds, te_ds))}

    ys = torch.tensor(np.asarray(tr_ds.targets, dtype=np.float32))
    if cfg["extensive"]:
        na = np.array([g.n_atoms for g in tr_ds.graph_data], dtype=np.float32)
        ys = ys / torch.tensor(na).clamp(min=1)
    y_mean, y_std = float(ys.mean()), float(ys.std())

    nw, bs = int(cfg["num_workers"]), int(cfg["batch_size"])
    pin = device.type == "cuda"
    gen = torch.Generator().manual_seed(int(cfg["seed"]))

    def ldr(d, sh):
        return DataLoader(d, bs, shuffle=sh, collate_fn=crystal_collate,
                          num_workers=nw, pin_memory=pin,
                          worker_init_fn=worker_init_fn,
                          generator=gen if sh else None, drop_last=False)

    tr_l, v_l, te_l = ldr(tr_ds, True), ldr(val_ds, False), ldr(te_ds, False)

    # --- pre-training audits ------------------------------------------------
    audits = {"overflow_atoms": graph_stats}
    if cfg["run_audits"]:
        print("\n--- graph audit ---")
        ea = audit_edges(v_l, device="cpu", cutoff=float(cfg["cutoff"]), max_batches=25)
        hist = ea.pop("histogram")
        np.save(os.path.join(ckpt_dir, f"{tag}_distance_hist.npy"),
                np.vstack([hist[0], hist[1][:-1]]))
        for k, v in ea.items():
            print(f"  {k:24s} {v:.6g}")
        audits["edges"] = ea

    # --- model --------------------------------------------------------------
    model = CrystalPropertyModel(num_types=num_types, elem_feat_dim=elem_feat_dim,
                                 **{k: cfg[k] for k in MODEL_KEYS})
    if elem_feat_dim > 0:
        model.set_elem_feat(nv[:, num_types:])
    model.set_y_stats(y_mean, y_std)
    model.fitted.fill_(1)
    model.summary()

    if cfg["run_audits"]:
        b0 = next(iter(v_l))
        audits["symmetry_init_fixed"] = symmetry_audit(
            model, b0, device=str(device), edge_mode="fixed",
            n_rotations=int(cfg["audit_rotations"]))

    # --- train --------------------------------------------------------------
    print("\ntraining ...")
    model, history = train_model(
        model, tr_l, v_l, epochs=int(cfg["epochs"]), lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]), clip=float(cfg["clip"]),
        device=device, patience=int(cfg["patience"]),
        log_every=int(cfg["log_every"]), ckpt_path=cfg["ckpt"],
        use_ema=bool(cfg["use_ema"]), ema_decay=float(cfg["ema_decay"]),
        warmup_frac=float(cfg["warmup_frac"]), extensive=bool(cfg["extensive"]))

    if cfg["metrics_csv"] and history:
        pd.DataFrame(history).to_csv(cfg["metrics_csv"], index=False)

    # --- evaluate -----------------------------------------------------------
    val_m = validate(model, v_l, device, extensive=bool(cfg["extensive"]))
    print(f"\nVAL   MAE={val_m['mae']:.4f} RMSE={val_m['rmse']:.4f} R2={val_m['r2']:.3f}")
    df = collect_predictions(model, te_l, device, extensive=bool(cfg["extensive"]))
    test_m = compute_metrics(torch.tensor(df["prediction"].values),
                             torch.tensor(df["target"].values)) if len(df) else {}
    print(f"TEST  MAE={test_m.get('mae', float('nan')):.4f} "
          f"RMSE={test_m.get('rmse', float('nan')):.4f} R2={test_m.get('r2', 0):.3f} "
          f"MAD/MAE={test_m.get('mad_over_mae', 0):.1f}")
    if cfg["predictions_csv"] and len(df):
        df.to_csv(cfg["predictions_csv"], index=False)

    # --- post-training audits ----------------------------------------------
    if cfg["run_audits"]:
        bt = next(iter(te_l))
        for mode in ("fixed", "rebuilt"):
            audits[f"symmetry_trained_{mode}"] = symmetry_audit(
                model, bt, device=str(device), edge_mode=mode,
                n_rotations=int(cfg["audit_rotations"]), verbose=True)
        audits["symmetry_trained_fixed_f64"] = symmetry_audit(
            model, bt, device=str(device), edge_mode="fixed", dtype=torch.float64,
            n_rotations=int(cfg["audit_rotations"]), verbose=True)

    if cfg["ckpt"]:
        save_checkpoint(cfg["ckpt"], model, extra={
            "val_metrics": val_m, "test_metrics": test_m, "history": history,
            "atomic_numbers": list(map(int, zs)), "target_name": cfg["target"],
            "run_config": cfg, "audits": audits, "composition_stats": comp})
        print(f"\nsaved -> {cfg['ckpt']}")

    return model, {"val": val_m, "test": test_m, "history": history,
                   "audits": audits, "predictions": df, "config": cfg}


def run(data_root=None, target=None, mount=False, **overrides):
    """Notebook entry point. Any key in :func:`~equicap.config.default_config` may be passed."""
    data_root = data_root or DEFAULT_DATA_ROOT
    maybe_mount_drive(data_root, force=mount)
    cfg = default_config()
    cfg["data_root"] = data_root
    if target:
        cfg["target"] = target
    unknown = set(overrides) - set(cfg)
    if unknown:
        print(f"[warn] ignoring unknown settings: {sorted(unknown)}")
    cfg.update({k: v for k, v in overrides.items() if k in cfg})
    return main(cfg)


def run_factorial(data_root=None, target="formation_energy_per_atom",
                  backbones=BACKBONES, readouts=READOUTS, seeds=(0, 1, 2),
                  include_controls: bool = True, controls=None,
                  out_csv: str = "factorial_results.csv", **overrides):
    """Full backbone x readout x seed grid plus the control cells."""
    data_root = data_root or DEFAULT_DATA_ROOT
    cells = [{"variant": "base", "backbone": b, "readout": r}
             for b in backbones for r in readouts]
    if include_controls:
        cells += [dict(c) for c in (controls or DEFAULT_CONTROLS)]

    rows = []
    for cell, s in itertools.product(cells, seeds):
        cfg = default_config()
        cfg.update({"data_root": data_root, "target": target, "seed": s})
        cfg.update({k: v for k, v in overrides.items() if k in cfg})
        cfg.update(cell)
        if cfg.get("routing_temperature") == "n_caps":
            cfg["routing_temperature"] = float(cfg["n_caps"])
        tag = run_tag(cfg)
        print("\n" + "=" * 72 + f"\nCELL  {tag}\n" + "=" * 72)
        try:
            model, res = main(cfg)
            au = res["audits"]
            rows.append({
                "variant": cfg["variant"], "backbone": cfg["backbone"],
                "readout": cfg["readout"], "seed": s, "n_params": model.n_params(),
                "val_mae": res["val"]["mae"], "test_mae": res["test"].get("mae"),
                "test_rmse": res["test"].get("rmse"), "test_r2": res["test"].get("r2"),
                "mad_over_mae": res["test"].get("mad_over_mae"),
                "epochs_run": len(res["history"]),
                "rot_drift_fixed": au.get("symmetry_trained_fixed", {}).get("rotation_drift"),
                "rot_drift_rebuilt": au.get("symmetry_trained_rebuilt", {}).get("rotation_drift"),
                "supercell_drift_fixed": au.get("symmetry_trained_fixed", {}).get("supercell_drift"),
                "ckpt": res["config"]["ckpt"],
            })
        except Exception as e:
            print(f"[cell failed] {tag}: {type(e).__name__}: {e}")
            rows.append({"variant": cfg["variant"], "backbone": cfg["backbone"],
                         "readout": cfg["readout"], "seed": s, "error": str(e)})
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    if "test_mae" in df:
        g = (df.dropna(subset=["test_mae"])
               .groupby(["variant", "backbone", "readout"])["test_mae"])
        print("\nFACTORIAL SUMMARY (test MAE, mean +- s.d. over seeds)")
        print(g.agg(["mean", "std", "count"]).round(5).to_string())
    print(f"\n-> {out_csv}")
    return df
