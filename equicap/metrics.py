"""Regression metrics and small tensor helpers.

``mad_over_mae`` is the headline sanity number: it is the ratio of the
mean-absolute-deviation baseline (predict the training mean) to the model's MAE.
A value near 1 means the model has learned nothing useful, whatever the R^2 says.
"""

from __future__ import annotations

from typing import Dict

import torch

__all__ = ["compute_metrics", "to_device"]


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """MAE, RMSE, R^2, Pearson r, and the MAD baseline ratio."""
    pred, target = pred.reshape(-1).double(), target.reshape(-1).double()
    n = pred.numel()
    mae = (pred - target).abs().mean()
    rmse = ((pred - target) ** 2).mean().sqrt()
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot.clamp(min=1e-12)
    # [FIX-10] compute r from centred sums; the old form mixed a biased mean with
    # torch's unbiased std and was off by (n-1)/n.
    if n > 1:
        pc, tc = pred - pred.mean(), target - target.mean()
        denom = (pc.pow(2).sum().sqrt() * tc.pow(2).sum().sqrt()).clamp(min=1e-12)
        r = (pc * tc).sum() / denom
    else:
        r = torch.tensor(0.0, dtype=torch.float64)
    mad = (target - target.mean()).abs().mean()
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), "r": float(r),
            "mad": float(mad), "mad_over_mae": float(mad / mae.clamp(min=1e-12)),
            "n": int(n)}


def to_device(batch, device):
    """Move every tensor in a collated batch to ``device``, leaving the rest alone."""
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}
