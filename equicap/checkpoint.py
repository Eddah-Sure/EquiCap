"""Checkpoint writing and loading.

A checkpoint carries the full model config, so :func:`load_property_model` can
rebuild the architecture without the caller knowing which cell of the factorial
design produced it.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from .model import CrystalPropertyModel

__all__ = ["save_checkpoint", "load_checkpoint", "load_property_model"]


def save_checkpoint(path: str, model: CrystalPropertyModel, extra: Optional[dict] = None):
    """Write weights + config + target standardisation, plus anything in ``extra``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "config": model.get_config(),
                "y_mean": model.y_mean.item(), "y_std": model.y_std.item(),
                "fitted": int(model.fitted.item()),
                "n_params": model.n_params(),
                "torch_version": torch.__version__,
                **(extra or {})}, path)


def load_checkpoint(path: str, model: CrystalPropertyModel, map_location="cpu"):
    """Load weights into an ALREADY constructed model; returns the checkpoint dict."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ck["model"])
    if "y_mean" in ck:
        model.set_y_stats(ck["y_mean"], ck["y_std"])
    if "fitted" in ck:
        model.fitted.fill_(int(ck["fitted"]))
    return ck


def load_property_model(path: str, map_location="cpu"):
    """Rebuild a model straight from a checkpoint; returns ``(model, ckpt_dict)``."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    if "config" not in ck:
        raise KeyError(f"{path} has no 'config' entry")
    model = CrystalPropertyModel.from_config(ck["config"])
    model.load_state_dict(ck["model"])
    if "y_mean" in ck:
        model.set_y_stats(ck["y_mean"], ck["y_std"])
    if "fitted" in ck:
        model.fitted.fill_(int(ck["fitted"]))
    model.eval()
    return model, ck
