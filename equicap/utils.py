"""Process-wide helpers: one-shot warnings, RNG seeding, environment probes.

This module must not import any other ``equicap`` module, so that every other
module is free to import it.
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch

__all__ = [
    "set_seed",
    "worker_init_fn",
    "in_notebook",
]

_WARNED: set[str] = set()


def _warn_once(msg: str) -> None:
    """Print ``msg`` to stderr at most once per process."""
    if msg in _WARNED:
        return
    _WARNED.add(msg)
    print(f"[equicap][WARN] {msg}", file=sys.stderr, flush=True)


def _reset_warnings() -> None:
    """Clear the one-shot warning cache (used by the test suite)."""
    _WARNED.clear()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG that can affect a run.

    With ``deterministic=True`` cuDNN autotuning is disabled and torch is asked
    for deterministic kernels, which is slower but makes a run bit-reproducible
    on the same hardware.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding: distinct, reproducible streams per worker."""
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


# Backwards-compatible private alias (the original single-file script used it).
_worker_init_fn = worker_init_fn


# ---------------------------------------------------------------------------
# Environment probes
# ---------------------------------------------------------------------------


def in_notebook() -> bool:
    """True inside IPython / Jupyter / Colab."""
    try:
        shell = get_ipython().__class__.__name__            # noqa: F821
        return shell in ("ZMQInteractiveShell", "Shell", "google.colab._shell")
    except NameError:
        pass
    return "ipykernel" in sys.modules or "google.colab" in sys.modules


_in_notebook = in_notebook


def maybe_mount_drive(data_root: str, force: bool = False) -> None:
    """Mount Google Drive when the data root points into it and is absent."""
    if not (force or (str(data_root).startswith("/content/gdrive")
                      and not os.path.isdir(data_root))):
        return
    try:
        from google.colab import drive
        drive.mount("/content/gdrive")
    except Exception as e:                                   # pragma: no cover
        print(f"(not in Colab or mount failed: {e})")


def progress(it, desc: str = ""):
    """Wrap an iterable in tqdm when it is installed, otherwise pass it through."""
    try:
        from tqdm.auto import tqdm
        return tqdm(it, desc=desc, leave=False)
    except Exception:
        return it
