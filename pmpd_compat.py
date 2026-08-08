"""
pmpd_compat.py — Compatibility shim for the PMPD package.

The top-level pmpd.__init__ eagerly imports pmpd.modules.model which pulls in
auto_gptq, and auto_gptq is broken against transformers >= 5.x
(missing 'no_init_weights').  This shim loads only the submodules that this
project actually needs (the scheduler and its implementations) by bypassing
the broken __init__ chain entirely.

Usage (drop-in replacement for the broken top-level import):
    from pmpd_compat import Scheduler          # base class + registry
    from pmpd_compat import NaiveScheduler     # concrete NaiveScheduler

The shim:
  1. Locates the installed pmpd package directory.
  2. Uses importlib to load scheduler.py and naive_scheduler.py directly.
  3. Registers all scheduler submodules so that Scheduler.get_scheduler()
     works correctly.
"""
from __future__ import annotations

import importlib.util
import os
import sys


def _load_module(name: str, path: str):
    """Load a Python source file into sys.modules under *name*."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bootstrap():
    """Locate pmpd on disk and load the scheduler submodules directly."""
    # Find the installed pmpd package directory without importing it.
    import importlib.util as _ilu
    pmpd_spec = _ilu.find_spec("pmpd")
    if pmpd_spec is None:
        raise ImportError(
            "The 'pmpd' package is not installed.\n"
            "Install it with: pip install git+https://github.com/SamsungLabs/PMPD.git"
        )

    pmpd_dir = os.path.dirname(pmpd_spec.origin)   # e.g. .../site-packages/pmpd
    sched_dir = os.path.join(pmpd_dir, "modules", "scheduler")

    # 1) Load scheduler.py (base Scheduler class + registry)
    sched_mod = _load_module(
        "pmpd.modules.scheduler.scheduler",
        os.path.join(sched_dir, "scheduler.py"),
    )

    # 2) Load every concrete scheduler implementation so they register
    #    themselves via __init_subclass__ / _scheduler_list.
    scheduler_files = [
        "naive_scheduler.py",
        "act_scheduler.py",
        "confidence_scheduler.py",
        "kv_cache_scheduler.py",
        "random_scheduler.py",
    ]
    for fname in scheduler_files:
        fpath = os.path.join(sched_dir, fname)
        if os.path.exists(fpath):
            mod_name = f"pmpd.modules.scheduler.{fname[:-3]}"
            try:
                _load_module(mod_name, fpath)
            except Exception:
                pass  # Skip any that have missing transitive deps

    return sched_mod


_sched_module = _bootstrap()

# Re-export the public symbols callers need
Scheduler = _sched_module.Scheduler

try:
    from pmpd.modules.scheduler.naive_scheduler import NaiveScheduler
except Exception:
    NaiveScheduler = None  # type: ignore
