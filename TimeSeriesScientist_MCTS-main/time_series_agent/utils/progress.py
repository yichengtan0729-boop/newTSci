"""
Verbose progress printing utility.

Usage:
    from utils.progress import vprint, set_verbose
    set_verbose(True)           # enable (typically from config["verbose"])
    vprint("MCTS", "Rollout %d/%d: reward=%.4f", 3, 10, -0.12)
    vprint("FUNNEL", "Phase 1 complete")
"""

from __future__ import annotations

import time
from typing import Any

_verbose = False
_t0: float = 0.0  # start timestamp for elapsed display


def set_verbose(v: bool) -> None:
    """Turn verbose printing on/off globally."""
    global _verbose, _t0
    _verbose = bool(v)
    if _verbose:
        _t0 = time.time()


def is_verbose() -> bool:
    return _verbose


def vprint(tag: str, msg: str, *args: Any) -> None:
    """Print a tagged progress message if verbose mode is on.

    Parameters
    ----------
    tag : str
        Short label, e.g. ``"MCTS"``, ``"FUNNEL"``, ``"TUNING"``, ``"LLM"``.
    msg : str
        Format-string message (printf-style ``%`` placeholders).
    *args :
        Values interpolated into *msg*.
    """
    if not _verbose:
        return
    try:
        text = msg % args if args else msg
    except Exception:
        text = msg
    elapsed = time.time() - _t0 if _t0 else 0.0
    print(f"[{elapsed:7.1f}s] [{tag:>8s}] {text}", flush=True)
