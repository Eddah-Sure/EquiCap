"""Inference: per-structure prediction tables and ad-hoc structure scoring.

:func:`collect_predictions` returns per-structure rows rather than summary
statistics on purpose — paired error analysis between two cells of the factorial
design is only reproducible if the predictions themselves are deposited.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch

from .metrics import to_device

__all__ = ["collect_predictions", "predict"]


@torch.no_grad()
def collect_predictions(model, loader, device, extensive=False,
                        return_latent: bool = False):
    """Run over a loader; returns a tidy DataFrame (and optionally pooled latents)."""
    model.eval()
    ids, preds, tgts, nat, sgs = [], [], [], [], []
    pooled_all = []
    for batch in loader:
        b = to_device(batch, device)
        edges = (b["src"], b["dst"], b["offset"], b["emask"])
        if return_latent:
            p, lat = model(b["types"], b["frac"], b["lattice"], b["mask"],
                           edges=edges, return_latent=True)
            pooled_all.append(lat["pooled"].cpu())
        else:
            p = model(b["types"], b["frac"], b["lattice"], b["mask"], edges=edges)
        preds.append(p.cpu())
        tgts.append(b["y"].reshape(-1).cpu())
        nat.append(b["n_atoms"].cpu())
        sgs.append(b["space_group"].cpu())
        ids.extend([str(m) for m in batch["material_ids"]])
    if not preds:
        return (pd.DataFrame(), None) if return_latent else pd.DataFrame()
    df = pd.DataFrame({
        "material_id": ids,
        "target": torch.cat(tgts).numpy(),
        "prediction": torch.cat(preds).numpy(),
        "n_atoms": torch.cat(nat).numpy(),
        "space_group": torch.cat(sgs).numpy(),
    })
    df["error"] = df["prediction"] - df["target"]
    df["abs_error"] = df["error"].abs()
    if return_latent:
        return df, torch.cat(pooled_all).numpy()
    return df


@torch.no_grad()
def predict(model, structures: List[dict], device="cpu") -> np.ndarray:
    """Predict for raw structure dicts.

    Each dict needs ``types``, ``frac``, ``lattice``, ``mask`` and may supply
    ``src`` / ``dst`` / ``offset`` / ``emask``; without them the graph is rebuilt
    from geometry with the model's own settings.
    """
    model.to(device).eval()
    B = len(structures)
    A_max = max(len(s["types"]) for s in structures)
    has_edges = all("src" in s for s in structures)
    E_max = max(1, max(len(s["src"]) if has_edges else 0 for s in structures))

    types = torch.zeros(B, A_max, dtype=torch.long, device=device)
    frac = torch.zeros(B, A_max, 3, device=device)
    L = torch.stack([torch.as_tensor(np.asarray(s["lattice"]), dtype=torch.float32)
                     for s in structures]).to(device)
    mask = torch.zeros(B, A_max, dtype=torch.bool, device=device)
    src = torch.zeros(B, E_max, dtype=torch.long, device=device)
    dst = torch.zeros(B, E_max, dtype=torch.long, device=device)
    offset = torch.zeros(B, E_max, 3, device=device)
    emask = torch.zeros(B, E_max, dtype=torch.bool, device=device)

    for i, s in enumerate(structures):
        a = len(s["types"])
        types[i, :a] = torch.as_tensor(s["types"]).to(device)
        frac[i, :a] = torch.as_tensor(np.asarray(s["frac"]), dtype=torch.float32).to(device)
        mask[i, :a] = torch.as_tensor(s.get("mask", torch.ones(a, dtype=torch.bool))).to(device)
        if has_edges:
            e = len(s["src"])
            src[i, :e] = torch.as_tensor(s["src"]).to(device)
            dst[i, :e] = torch.as_tensor(s["dst"]).to(device)
            offset[i, :e] = torch.as_tensor(s.get("offset", torch.zeros(e, 3))).to(device)
            emask[i, :e] = torch.as_tensor(
                s.get("emask", torch.ones(e, dtype=torch.bool))).to(device)

    edges = (src, dst, offset, emask) if has_edges else None
    return model(types, frac, L, mask, edges=edges).cpu().numpy()
