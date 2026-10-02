"""Pooling heads.

Every readout returns a *pooled representation*, never a prediction: the
regression head lives in :mod:`equicap.model` so that the readout factor and the
head are not confounded in the factorial design. Each exposes ``out_dim`` and
the same call signature ``(x, mask, weights, return_weights)``, so they are
drop-in substitutes for one another.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import neg_inf

__all__ = ["AttentionPool", "RoutingWeightedPool", "MeanMaxPool"]


class AttentionPool(nn.Module):
    """Additive attention concatenated with masked mean."""

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)
        self.out_dim = 2 * dim

    def forward(self, x, mask=None, weights=None, return_weights: bool = False):
        a = self.attn(x)
        if mask is not None:
            a = a.masked_fill(~mask.unsqueeze(-1), neg_inf(a.dtype))
        w = F.softmax(a, dim=1)
        pooled = (w * x).sum(1)
        if mask is not None:
            m = mask.to(x.dtype).unsqueeze(-1)
            mean = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        else:
            mean = x.mean(1)
        out = torch.cat([pooled, mean], -1)
        return (out, w.squeeze(-1)) if return_weights else out


class RoutingWeightedPool(nn.Module):
    """Routing-mass-weighted mean concatenated with max over the capsule set."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.out_dim = 2 * dim

    def forward(self, x, mask=None, weights=None, return_weights: bool = False):
        x = self.norm(x)
        w = (torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
             if weights is None else weights.clamp(min=0).to(x.dtype))
        if mask is not None:
            w = w * mask.to(x.dtype)
        wu = w.unsqueeze(-1)
        mean = (wu * x).sum(1) / wu.sum(1).clamp(min=1e-6)
        if mask is not None:
            xm = x.masked_fill(~mask.unsqueeze(-1), neg_inf(x.dtype))
            mx = xm.max(1).values
            mx = torch.where(mask.any(1, keepdim=True), mx, torch.zeros_like(mx))
        else:
            mx = x.max(1).values
        out = torch.cat([mean, mx], -1)
        return (out, w) if return_weights else out


class MeanMaxPool(nn.Module):
    """Null readout: masked mean concatenated with masked max."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.out_dim = 2 * dim

    def forward(self, x, mask=None, weights=None, return_weights: bool = False):
        x = self.norm(x)
        if mask is not None:
            m = mask.to(x.dtype).unsqueeze(-1)
            mean = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
            mx = x.masked_fill(~mask.unsqueeze(-1), neg_inf(x.dtype)).max(1).values
            mx = torch.where(mask.any(1, keepdim=True), mx, torch.zeros_like(mx))
        else:
            mean, mx = x.mean(1), x.max(1).values
        out = torch.cat([mean, mx], -1)
        if return_weights:
            w = (mask.to(x.dtype) if mask is not None
                 else torch.ones(x.shape[:2], device=x.device, dtype=x.dtype))
            return out, w
        return out
