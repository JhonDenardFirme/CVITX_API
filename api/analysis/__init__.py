# -*- coding: utf-8 -*-
"""
Lightweight package init for api.analysis.

Goals
-----
• Zero heavy side-effects on import (no GPU/torch/engine import at package import time)
• Lazy re-export of internal submodules to keep workers and API startup fast
• Optional "eager" mode for debugging via ENV toggle
• Type-checker friendly (no runtime cost)

Usage
-----
from api.analysis import utils_runtime, utils_color, utils_plate, bbox_utils, engine

Notes
-----
• This module never imports torch/engine unless you actually access .engine
• Set CVITX_ANALYSIS_EAGER=1 to eagerly import all exposed submodules at startup
"""
from __future__ import annotations
import importlib, os, sys
from typing import TYPE_CHECKING, Dict, Set

_EXPOSED: Set[str] = {
    "utils_runtime",
    "utils_color",
    "utils_plate",
    "bbox_utils",
    "engine",
}
__all__ = tuple(sorted(_EXPOSED))
_cache: Dict[str, object] = {}

def _lazy_import(name: str):
    if name in _cache:
        return _cache[name]
    full = f"{__name__}.{name}"
    try:
        mod = importlib.import_module(full)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Cannot import '{name}' from package '{__name__}'. "
            f"Expected module path: '{full}'. Original: {e}"
        ) from e
    _cache[name] = mod
    return mod

def __getattr__(name: str):
    if name in _EXPOSED:
        return _lazy_import(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + list(_EXPOSED))

if os.getenv("CVITX_ANALYSIS_EAGER", "").strip() in {"1","true","True","YES","yes"}:
    for _n in list(_EXPOSED):
        try:
            _lazy_import(_n)
        except Exception as _e:
            print(f"[api.analysis] EAGER import failed for '{_n}': {_e}", file=sys.stderr)

if TYPE_CHECKING:
    from . import utils_runtime, utils_color, utils_plate, bbox_utils, engine
