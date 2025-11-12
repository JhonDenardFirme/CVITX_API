# ── utils_runtime.py 
# File: api/analysis/utils_runtime.py
# -*- coding: utf-8 -*-
"""
Runtime utilities for inference metrics (warmup, GFLOPs, memory).
Safe on CPU-only hosts; CUDA/NVML/THOP are optional via try/except.

PUBLIC CONTRACT (unchanged: do not rename without coordinating workers/DB):
  - warmup_model(model, device: str, img_size: int, steps: int = 2) -> int
      Runs N forward passes to warm kernels/caches. Returns warmup_ms (int).
  - estimate_gflops(model, img_size: int, channels: int = 3) -> Optional[float]
      One-time estimate per (model, img_size). Cached. Returns GFLOPs or None.
  - total_gpu_gb() -> Optional[float]
      Total GPU memory (GB) via NVML, fallback to torch.cuda.mem_get_info. Cached.
  - per_infer_memory_metrics(device: str) -> Dict[str, Optional[float]]
      Returns {
        "mem_gb": Optional[float],          # allocated GB (decimal, 1e9)
        "memory_usage": Optional[float],    # ratio in [0,1] = mem_gb / total_gb
        "reserved_gb": Optional[float],     # CUDA reserved GB (optional ops aid)
        "total_gb": Optional[float],        # total device GB (decimal, 1e9)
      }

ENGINE USAGE (unchanged):
  warm_ms = warmup_model(model, dev, IMG_SIZE, steps=WARMUP_STEPS)
  # ... measure forward latency for your request ...
  gfl = estimate_gflops(model, IMG_SIZE)
  mem = per_infer_memory_metrics(dev)
  # emit fields:
  #   "latency_ms": total_ms - warm_ms
  #   "gflops": gfl
  #   "mem_gb": mem["mem_gb"]
  #   "memory_usage": mem["memory_usage"]   # ratio; FE/DB contract preserved

Notes:
- AMP autocast is used for warmup on CUDA if available.
- GFLOPs uses THOP (optional). If not installed or profiling fails, returns None.
- NVML (optional) provides total GPU memory; falls back to torch.cuda.mem_get_info().
- All paths are CPU-safe; on CPU-only hosts these APIs return None where appropriate.
"""

from __future__ import annotations
import time
import logging
from typing import Optional, Tuple, Dict

import torch
from contextlib import nullcontext

LOG = logging.getLogger("utils_runtime")

# --------- Simple caches (process-lifetime) ---------
_GFLOPS_CACHE: Dict[Tuple[int, int, int], Optional[float]] = {}  # (model_id, H, C) -> GFLOPs|None
_TOTAL_GPU_GB: Optional[float] = None


# --------- Helpers ---------
def _is_cuda_device(device: str) -> bool:
    return isinstance(device, str) and device.startswith("cuda")


def _sync_if_cuda(device: str) -> None:
    if _is_cuda_device(device) and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception as e:
            LOG.debug("CUDA synchronize failed (non-fatal): %s", e)


def _resolve_device(model: torch.nn.Module, device: str) -> torch.device:
    """
    Choose a safe torch.device to run a dummy forward:
    1) If user asked for CUDA and it's available, use that device ('cuda' or 'cuda:N').
    2) Else, use the model's current parameter device if any.
    3) Else, fallback to CPU.
    """
    try:
        if _is_cuda_device(device) and torch.cuda.is_available():
            return torch.device(device)
    except Exception:
        pass
    try:
        return next(model.parameters()).device  # type: ignore[arg-type]
    except Exception:
        return torch.device("cpu")


# --------- Public API ---------
def warmup_model(model: torch.nn.Module, device: str, img_size: int, steps: int = 2) -> int:
    """
    Run a few forward passes to warm up kernels/caches before measuring latency.
    Returns the warmup time in milliseconds (rounded int). CPU-safe.

    Implementation notes:
      - Uses inference_mode() for lower overhead.
      - Resolves a safe device; if requested device is invalid, falls back to model's device.
      - Preserves original training/eval mode (restores it after warmup).
      - Uses AMP autocast on CUDA if available.
    """
    if steps <= 0:
        return 0

    was_training = model.training
    try:
        model.eval()
        run_dev = _resolve_device(model, device)

        h = int(img_size)
        try:
            x = torch.zeros(1, 3, h, h, device=run_dev)
        except Exception as e:
            LOG.warning("warmup_model: failed to allocate dummy input on '%s': %s; using CPU.", run_dev, e)
            x = torch.zeros(1, 3, h, h)

        use_amp = (run_dev.type == "cuda") and torch.cuda.is_available()
        amp_ctx = torch.amp.autocast(device_type="cuda") if use_amp else nullcontext()

        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                with amp_ctx:
                    for _ in range(int(steps)):
                        _ = model(x)
                    _sync_if_cuda(run_dev.type)
        except Exception as e:
            LOG.warning("warmup_model: forward failed during warmup (non-fatal): %s", e)
        dt_ms = int(round((time.perf_counter() - t0) * 1000.0))
        return dt_ms
    finally:
        if was_training:
            try:
                model.train()
            except Exception:
                pass


def _total_gpu_gb_via_nvml() -> Optional[float]:
    """Try NVML for total GPU memory (GB, decimal 1e9)."""
    try:
        import pynvml  # package 'nvidia-ml-py3'
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        return float(info.total) / 1e9
    except Exception as e:
        LOG.debug("NVML total mem unavailable: %s", e)
        return None


def _total_gpu_gb_via_torch() -> Optional[float]:
    """Fallback: torch.cuda.mem_get_info() total in GB (decimal 1e9), if CUDA is available."""
    try:
        if torch.cuda.is_available():
            # mem_get_info returns (free_bytes, total_bytes)
            _, total_b = torch.cuda.mem_get_info()
            return float(total_b) / 1e9
    except Exception as e:
        LOG.debug("torch total mem unavailable: %s", e)
    return None


def total_gpu_gb() -> Optional[float]:
    """
    Total GPU memory in GB (cached). Uses NVML first, then torch fallback.
    Returns None on CPU or if both methods are unavailable.
    """
    global _TOTAL_GPU_GB
    if _TOTAL_GPU_GB is not None:
        return _TOTAL_GPU_GB

    total = _total_gpu_gb_via_nvml()
    if total is None:
        total = _total_gpu_gb_via_torch()
    _TOTAL_GPU_GB = total
    return _TOTAL_GPU_GB


def per_infer_memory_metrics(device: str) -> Dict[str, Optional[float]]:
    """
    Returns a dict with:
      - mem_gb: torch-reported currently allocated memory in GB (decimal, 1e9). May be None.
      - memory_usage: mem_gb / total_gb (ratio in [0,1]) if total is known; else None.
      - reserved_gb: torch-reported reserved CUDA memory in GB (decimal, 1e9). May be None.
      - total_gb: total GPU memory in GB (decimal, 1e9) if known; else None.

    Notes:
    - Reports *allocated* memory for the current device; keeps it cheap without NVML.
    - On CPU-only, values are None.
    - Retains the historical contract where 'memory_usage' is a ratio.
    """
    used_gb: Optional[float] = None
    reserved_gb: Optional[float] = None
    total_gb: Optional[float] = None

    try:
        if _is_cuda_device(device) and torch.cuda.is_available():
            try:
                idx = int(device.split(":")[1]) if ":" in device else torch.cuda.current_device()
            except Exception:
                idx = torch.cuda.current_device()

            used_gb = float(torch.cuda.memory_allocated(idx)) / 1e9
            # reserved may not be meaningful on all backends, guard with try
            try:
                reserved_gb = float(torch.cuda.memory_reserved(idx)) / 1e9
            except Exception:
                reserved_gb = None

            # Prefer cached total; if None, try properties as a best-effort
            total_gb = total_gpu_gb()
            if total_gb is None:
                try:
                    props = torch.cuda.get_device_properties(idx)
                    total_gb = float(getattr(props, "total_memory", 0.0)) / 1e9 or None
                except Exception:
                    total_gb = None
    except Exception as e:
        LOG.debug("per_infer_memory_metrics: CUDA memory query failed: %s", e)

    ratio = (used_gb / total_gb) if (used_gb is not None and total_gb) else None
    return {
        "mem_gb": used_gb,
        "memory_usage": ratio,      # keep as ratio for contract stability
        "reserved_gb": reserved_gb,
        "total_gb": total_gb,
    }


def estimate_gflops(model: torch.nn.Module, img_size: int, channels: int = 3) -> Optional[float]:
    """
    Estimate GFLOPs using THOP on a single (1, C, H, H) dummy input.
    Cached per (model_id, H, C). Returns None if THOP is missing or profiling fails.

    NOTE: We report MACs/1e9 from THOP as 'gflops' to keep behavior consistent
    with existing logs (no *2 factor applied).
    """
    key = (id(model), int(img_size), int(channels))
    if key in _GFLOPS_CACHE:
        return _GFLOPS_CACHE[key]

    try:
        from thop import profile  # optional dep
        dev = _resolve_device(model, str(next(model.parameters()).device) if torch.cuda.is_available() else "cpu")
        x = torch.zeros(1, int(channels), int(img_size), int(img_size), device=dev)
        was_training = model.training
        try:
            model.eval()
            with torch.inference_mode():
                macs, _ = profile(model, inputs=(x,), verbose=False)
        finally:
            if was_training:
                try:
                    model.train()
                except Exception:
                    pass
        gflops = float(macs) / 1e9
        _GFLOPS_CACHE[key] = gflops
        return gflops
    except Exception as e:
        LOG.debug("estimate_gflops: THOP/profile failed or unavailable: %s", e)
        _GFLOPS_CACHE[key] = None
        return None


__all__ = [
    "warmup_model",
    "estimate_gflops",
    "total_gpu_gb",
    "per_infer_memory_metrics",
]
