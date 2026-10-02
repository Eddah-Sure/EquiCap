"""Single source of truth for run configuration.  [R3-10]

:func:`default_config` is the only place defaults are written down; the model
class defaults match it, and the CLI is generated from it, so a new setting only
has to be added once.
"""

from __future__ import annotations

import os

import torch

__all__ = [
    "default_config", "check_dataset_layout", "run_tag",
    "DEFAULT_DATA_ROOT", "DEFAULT_CKPT_DIRNAME", "SPLITS", "MODEL_KEYS",
]

DEFAULT_DATA_ROOT = ("/content/gdrive/MyDrive/2024SUMMER/MaterialsProjectData/"
                     "MaterialsProjectData/MP20/")
DEFAULT_CKPT_DIRNAME = "propertycheckpoints"
SPLITS = ("train", "val", "test")

#: Config keys forwarded verbatim to :class:`~equicap.model.CrystalPropertyModel`.
MODEL_KEYS = ("backbone", "readout", "hidden", "n_layers", "num_rbf", "cutoff",
              "max_deg", "neighbor_mode", "prim_dim", "n_caps", "sec_dim",
              "routing_iters", "routing_norm", "routing_temperature", "chirality",
              "probe", "vec_eps", "extensive", "dropout", "global_features")


def default_config() -> dict:
    """Default run configuration. Every key is overridable from CLI or ``run()``."""
    return {
        # ---- data ----
        "data_root": None,
        "target": "formation_energy_per_atom",
        "max_atoms": None,
        "num_workers": 0,
        "batch_size": 32,
        "edge_source": "rebuild",       # rebuild | stored
        "graph_cache_dir": None,        # default: <ckpt_dir>/graph_cache
        "corrupt": "none",              # none | coords | edges | lattice
        # ---- model ----
        "backbone": "cgcnn",            # painn | cgcnn | composition
        "readout": "capsule",           # attention | capsule | mean
        "hidden": 128,
        "n_layers": 3,
        "num_rbf": 16,
        "cutoff": 5.0,
        "max_deg": 64,                  # radius-graph safety cap / K for knn
        "neighbor_mode": "radius",      # radius | knn
        "prim_dim": 32,
        "n_caps": 8,
        "sec_dim": 64,
        "routing_iters": 3,
        "routing_norm": "atoms",        # atoms | mass
        "routing_temperature": 1.0,
        "chirality": False,
        "probe": "smooth",              # smooth | legacy
        "vec_eps": 1e-2,
        "extensive": False,
        "dropout": 0.10,
        "global_features": "intensive",  # none | density | intensive | full
        # ---- optimisation ----
        "epochs": 200,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "clip": 1.0,
        "patience": 25,
        "warmup_frac": 0.03,
        "use_ema": True,
        "ema_decay": 0.999,
        "log_every": 1,
        # ---- io / reproducibility ----
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "ckpt_dir": None,
        "ckpt": "property_model.pt",
        "metrics_csv": "property_metrics.csv",
        "predictions_csv": "property_test_predictions.csv",
        "variant": "base",
        "seed": 0,
        "deterministic": False,
        "run_audits": True,
        "audit_rotations": 8,
    }


def check_dataset_layout(data_root: str, splits=SPLITS) -> None:
    """Fail early and loudly when the expected split files are missing."""
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root does not exist: {data_root}\n"
                                "In Colab, mount Drive first: run(..., mount=True)")
    missing = []
    for split in splits:
        d = os.path.join(data_root, split)
        for fn in (f"{split}_config.json", f"{split}.npz", f"{split}.csv"):
            if not os.path.exists(os.path.join(d, fn)):
                missing.append(os.path.join(d, fn))
    if missing:
        raise FileNotFoundError("Dataset under\n  " + data_root +
                                "\nis missing:\n  " + "\n  ".join(missing))


def run_tag(cfg: dict) -> str:
    """Filename-safe identifier for one cell of the design."""
    return f"{cfg['backbone']}_{cfg['readout']}_{cfg.get('variant', 'base')}_s{cfg['seed']}"


_tag = run_tag
