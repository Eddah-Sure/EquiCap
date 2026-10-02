"""Training loop, EMA, and the per-step LR schedule.

Two details matter for the Methods section:

* [FIX-8] the EMA is updated per optimiser **step**, not per epoch, so the
  reported decay means what it says;
* [FIX-9] warmup + cosine are stepped per optimiser step for the same reason,
  which also decouples the LR trajectory from the dataset size.

Checkpoint selection and early stopping both use the raw validation MAE computed
on the EMA weights, and the checkpoint written is exactly the weight set that
produced the reported score.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .checkpoint import save_checkpoint
from .metrics import compute_metrics, to_device
from .utils import progress

__all__ = ["ModelEMA", "train_epoch", "validate", "train_model"]


class ModelEMA:
    """Per-STEP exponential moving average of the weights.

    ``apply_to()`` swaps the averaged weights in and keeps a backup of the live
    ones; ``restore()`` puts them back, so evaluating on the average never costs
    the optimiser its trajectory.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.n = 0
        self.shadow = {k: p.detach().clone()
                       for k, p in model.named_parameters() if p.requires_grad}
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for k, p in model.named_parameters():
            if p.requires_grad and k in self.shadow:
                self.shadow[k].mul_(d).add_(p.detach(), alpha=1 - d)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        self.backup = {}
        for k, p in model.named_parameters():
            if p.requires_grad and k in self.shadow:
                self.backup[k] = p.detach().clone()
                p.copy_(self.shadow[k])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for k, p in model.named_parameters():
            if k in self.backup:
                p.copy_(self.backup[k])
        self.backup = {}


def lr_at(step: int, total: int, base: float, warmup: int, min_frac: float = 0.05) -> float:
    """Per-step linear warmup followed by cosine decay to ``min_frac * base``."""
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    prog = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return base * (min_frac + 0.5 * (1 - min_frac) * (1 + math.cos(math.pi * prog)))


_lr_at = lr_at


def train_epoch(model, loader, optimizer, device, clip=1.0, extensive=False,
                ema=None, sched=None, desc="") -> Dict[str, float]:
    """One training epoch. Returns normalized-space loss and physical-unit MAE."""
    model.train()
    losses, err_sum, n_seen = [], 0.0, 0
    bar = progress(loader, desc)
    for batch in bar:
        b = to_device(batch, device)
        edges = (b["src"], b["dst"], b["offset"], b["emask"])
        optimizer.zero_grad(set_to_none=True)
        if sched is not None:
            lr = sched()
            for g in optimizer.param_groups:
                g["lr"] = lr
        # denorm=False: compare the raw (standardized) output against a
        # standardized target. Using the default would compare a physical-unit
        # prediction against a z-scored target.
        pred = model(b["types"], b["frac"], b["lattice"], b["mask"], edges=edges,
                     denorm=False)
        tgt_raw = b["y"].reshape(-1)
        tgt = tgt_raw / b["n_atoms"].to(tgt_raw.dtype) if extensive else tgt_raw
        tgt_norm = (tgt - model.y_mean) / model.y_std
        loss = F.smooth_l1_loss(pred, tgt_norm)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if not torch.isfinite(gn):
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()
        if ema is not None:
            ema.update(model)                       # [FIX-8] per step
        losses.append(float(loss))
        with torch.no_grad():
            ph = pred.detach() * model.y_std + model.y_mean
            if extensive:
                ph = ph * b["n_atoms"].to(ph.dtype)
            err_sum += float((ph - tgt_raw).abs().sum())
            n_seen += int(tgt_raw.numel())
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(loss=f"{np.mean(losses):.4f}",
                            mae=f"{err_sum / max(n_seen, 1):.4f}")
    return {"loss": float(np.mean(losses)) if losses else float("inf"),
            "mae": err_sum / n_seen if n_seen else float("inf")}


@torch.no_grad()
def validate(model, loader, device, extensive=False) -> Dict[str, float]:
    """Physical-unit metrics plus the same smooth-L1 objective used in training."""
    model.eval()
    preds, tgts, losses = [], [], []
    for batch in loader:
        b = to_device(batch, device)
        edges = (b["src"], b["dst"], b["offset"], b["emask"])
        pn = model(b["types"], b["frac"], b["lattice"], b["mask"], edges=edges,
                   denorm=False)
        tgt_raw = b["y"].reshape(-1)
        tgt = tgt_raw / b["n_atoms"].to(tgt_raw.dtype) if extensive else tgt_raw
        vl = F.smooth_l1_loss(pn, (tgt - model.y_mean) / model.y_std)
        if torch.isfinite(vl):
            losses.append(float(vl))
        p = pn * model.y_std + model.y_mean
        if extensive:
            p = p * b["n_atoms"].to(p.dtype)
        preds.append(p.cpu())
        tgts.append(tgt_raw.cpu())
    if not preds:
        return {"mae": float("inf"), "rmse": float("inf"), "r2": 0.0, "r": 0.0,
                "loss": float("inf"), "mad": 0.0, "mad_over_mae": 0.0, "n": 0}
    out = compute_metrics(torch.cat(preds), torch.cat(tgts))
    out["loss"] = float(np.mean(losses)) if losses else float("inf")
    return out


def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3,
                weight_decay=1e-5, clip=1.0, device="cpu", patience=15,
                log_every=1, ckpt_path="property_model.pt", use_ema=True,
                ema_decay=0.999, warmup_frac=0.03, min_lr_frac=0.05,
                extensive=False, verbose=True):
    """Train with AdamW, per-step cosine LR, EMA, and early stopping on val MAE."""
    model.to(device)
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.dim() <= 1 or n_.endswith(".bias")) else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lr)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * int(epochs)
    warmup_steps = int(max(1, warmup_frac * total_steps))
    state = {"step": 0}

    def sched():
        lr_now = lr_at(state["step"], total_steps, lr, warmup_steps, min_lr_frac)
        state["step"] += 1
        return lr_now

    ema = ModelEMA(model, ema_decay) if use_ema else None
    best, best_state, best_ep, stale = float("inf"), None, 0, 0
    history: List[dict] = []
    t0 = time.time()

    for ep in range(int(epochs)):
        te = time.time()
        tr = train_epoch(model, train_loader, opt, device, clip, extensive,
                         ema=ema, sched=sched, desc=f"ep {ep+1}/{epochs}")
        if ema is not None:
            ema.apply_to(model)
        va = validate(model, val_loader, device, extensive)
        improved = va["mae"] < best - 1e-6
        if improved:
            best, best_ep = va["mae"], ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ckpt_path:
                save_checkpoint(ckpt_path, model, extra={"val_mae": best, "epoch": best_ep})
        if ema is not None:
            ema.restore(model)

        row = {"epoch": ep + 1, "lr": opt.param_groups[0]["lr"],
               "train_loss": tr["loss"], "train_mae": tr["mae"],
               "val_loss": va["loss"], "val_mae": va["mae"], "val_rmse": va["rmse"],
               "val_r2": va["r2"], "val_r": va["r"], "sec": time.time() - te}
        history.append(row)
        if verbose and (ep + 1) % max(log_every, 1) == 0:
            print(f"  ep {ep+1:3d}/{epochs} lr={row['lr']:.2e} "
                  f"train {tr['loss']:.4f}/{tr['mae']:.4f}  "
                  f"val {va['loss']:.4f}/{va['mae']:.4f}  "
                  f"RMSE {va['rmse']:.4f}  R2 {va['r2']:.3f}"
                  + ("  *" if improved else ""))
        stale = 0 if improved else stale + 1
        if patience and stale >= patience:
            if verbose:
                print(f"  early stop @ epoch {ep+1} (best {best:.4f} @ {best_ep})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"  restored best weights: val MAE {best:.4f} @ epoch {best_ep} "
                  f"({time.time()-t0:.0f}s total)")
    model.eval()
    return model, history
