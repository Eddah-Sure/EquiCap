"""Command-line interface.

The parser is generated from :func:`~equicap.config.default_config`, so every
config key is automatically a flag (``hidden`` -> ``--hidden``) and a new setting
never has to be registered twice.
"""

from __future__ import annotations

import argparse

from .config import DEFAULT_DATA_ROOT, default_config
from .experiment import main, run_factorial
from .selftest import self_test
from .utils import in_notebook, maybe_mount_drive

__all__ = ["build_arg_parser", "cli", "console_main"]


def _str2bool(x) -> bool:
    return str(x).lower() in ("1", "true", "yes", "y")


def build_arg_parser() -> argparse.ArgumentParser:
    d = default_config()
    ap = argparse.ArgumentParser(description="EquiCap crystal property model")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--mount", action="store_true",
                    help="mount Google Drive before reading data (Colab)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the synthetic end-to-end check and exit")
    ap.add_argument("--factorial", action="store_true",
                    help="run the backbone x readout x seed grid")
    ap.add_argument("--no-controls", action="store_true",
                    help="with --factorial, skip the control cells")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--no-audits", action="store_true")
    for k, v in d.items():
        if k in ("data_root", "use_ema", "run_audits", "device"):
            continue
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(flag, type=_str2bool, default=v)
        elif isinstance(v, int):
            ap.add_argument(flag, type=int, default=v)
        elif isinstance(v, float):
            ap.add_argument(flag, type=float, default=v)
        else:
            ap.add_argument(flag, default=v)
    ap.add_argument("--device", default=None)
    return ap


def cli(argv=None):
    args, unknown = build_arg_parser().parse_known_args(argv)
    if unknown:
        print(f"[note] ignoring unrecognised arguments: {unknown}")
    if args.self_test:
        return self_test()
    maybe_mount_drive(args.data_root, force=args.mount)
    d = default_config()
    over = {k: getattr(args, k) for k in d if hasattr(args, k)}
    over["use_ema"] = not args.no_ema
    over["run_audits"] = not args.no_audits
    if args.device:
        over["device"] = args.device
    for k in ("max_atoms", "ckpt_dir", "graph_cache_dir"):
        if over.get(k) in ("None", ""):
            over[k] = None
    if over.get("max_atoms") is not None:
        over["max_atoms"] = int(over["max_atoms"])
    if args.factorial:
        over = {k: v for k, v in over.items()
                if k not in ("backbone", "readout", "variant", "seed", "target")}
        return run_factorial(data_root=args.data_root, target=args.target,
                             seeds=tuple(args.seeds),
                             include_controls=not args.no_controls, **over)
    cfg = default_config()
    cfg["data_root"] = args.data_root
    cfg.update(over)
    return main(cfg)


def console_main():
    """Entry point for ``python -m equicap`` and the ``equicap`` console script."""
    if in_notebook():
        print(f"[notebook] default data_root={DEFAULT_DATA_ROOT!r}\n"
              f"           use run(...) or run_factorial(...) for keyword control\n")
    return cli()
