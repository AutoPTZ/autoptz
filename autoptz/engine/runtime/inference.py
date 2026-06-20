"""ONNX Runtime session factory: pick the best available EP per platform/hardware.

Selection order:
  macOS (Apple Silicon / Intel+AMD): CoreML → CPU
  Windows:                           TensorRT → CUDA → DirectML → OpenVINO → CPU
  Linux:                             TensorRT → CUDA → OpenVINO → CPU

Beyond *picking* the EP, :func:`make_session` also *tunes* it: full graph
optimization, sensible threading, and per-EP acceleration options (CoreML
MLProgram on the ANE/GPU, TensorRT FP16 + a persistent engine cache, OpenVINO
FP16, …).  Every step degrades safely — a provider that rejects its options is
retried bare, and any GPU failure falls back to CPU — so a session is always
returned when the model is valid.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import onnxruntime as ort

logger = logging.getLogger(__name__)

Precision = Literal["auto", "fp32", "fp16"]


class EP(str, Enum):
    COREML = "CoreMLExecutionProvider"
    TENSORRT = "TensorrtExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    DIRECTML = "DmlExecutionProvider"
    OPENVINO = "OpenVINOExecutionProvider"
    CPU = "CPUExecutionProvider"


#: EPs that run on a GPU/accelerator (used to decide FP16 + fallback messaging).
_ACCELERATED_EPS = frozenset({EP.COREML, EP.TENSORRT, EP.CUDA, EP.DIRECTML, EP.OPENVINO})


@dataclass
class HardwarePrefs:
    """Per-camera or global hardware preferences forwarded to ``make_session``."""

    force_ep: EP | None = None
    #: "auto" lets each EP choose (TRT/OpenVINO use FP16); "fp32"/"fp16" force it.
    precision: Precision = "auto"
    #: Cap intra-op threads (None = ORT default ≈ physical cores). Lower this when
    #: many camera workers share the machine to avoid CPU oversubscription.
    intra_op_threads: int | None = None


# Global fallback order regardless of platform
_GLOBAL_ORDER: list[EP] = [
    EP.COREML,
    EP.TENSORRT,
    EP.CUDA,
    EP.DIRECTML,
    EP.OPENVINO,
    EP.CPU,
]

# Platform-preferred order (prepended to _GLOBAL_ORDER, deduped)
_PLATFORM_ORDER: dict[str, list[EP]] = {
    "darwin": [EP.COREML, EP.CPU],
    "win32": [EP.TENSORRT, EP.CUDA, EP.DIRECTML, EP.OPENVINO, EP.CPU],
    "linux": [EP.TENSORRT, EP.CUDA, EP.OPENVINO, EP.CPU],
}


def prefs_from_env() -> HardwarePrefs | None:
    """Build HardwarePrefs from AUTOPTZ_* env vars, or None if none are set.

    The supervisor sets these before spawning camera workers (env is inherited by
    spawned processes), so global hardware prefs reach every worker's sessions
    without threading them through the command schema.
    """
    force = os.environ.get("AUTOPTZ_FORCE_EP")
    precision_raw = os.environ.get("AUTOPTZ_PRECISION", "auto")
    threads_raw = os.environ.get("AUTOPTZ_ORT_INTRA_THREADS")

    ep: EP | None = None
    if force:
        try:
            ep = EP(force)
        except ValueError:
            logger.warning("Ignoring unknown AUTOPTZ_FORCE_EP=%r", force)
    precision: Precision = "auto"
    if precision_raw == "fp32":
        precision = "fp32"
    elif precision_raw == "fp16":
        precision = "fp16"
    threads: int | None = None
    if threads_raw:
        try:
            threads = max(1, int(threads_raw))
        except ValueError:
            threads = None

    if ep is None and precision == "auto" and threads is None:
        return None
    return HardwarePrefs(force_ep=ep, precision=precision, intra_op_threads=threads)


def _resolve_prefs(prefs: HardwarePrefs | None) -> HardwarePrefs | None:
    """Fall back to env-derived prefs when the caller passes none."""
    return prefs if prefs is not None else prefs_from_env()


def _available_providers() -> frozenset[str]:
    return frozenset(ort.get_available_providers())


def _candidate_order() -> list[EP]:
    """Return deduplicated EP candidates, platform-preferred first, CPU always last."""
    platform_preferred = _PLATFORM_ORDER.get(sys.platform, _GLOBAL_ORDER)
    seen: set[EP] = set()
    result: list[EP] = []
    for ep in platform_preferred + _GLOBAL_ORDER:
        if ep not in seen and ep != EP.CPU:
            seen.add(ep)
            result.append(ep)
    result.append(EP.CPU)  # always last; guaranteed to be present
    return result


def get_best_ep(prefs: HardwarePrefs | None = None) -> EP:
    """Return the best EP available on this machine without creating a session."""
    prefs = _resolve_prefs(prefs)
    if prefs and prefs.force_ep:
        forced = prefs.force_ep
        available = _available_providers()
        if forced.value in available:
            logger.info("Forced EP %s is available", forced.value)
            return forced
        logger.warning(
            "Forced EP %s not available (available: %s); falling back to auto-select",
            forced.value,
            sorted(available),
        )

    available = _available_providers()
    for ep in _candidate_order():
        if ep.value in available:
            logger.debug("Selected EP: %s (available: %s)", ep.value, sorted(available))
            return ep

    return EP.CPU  # always available


def _trt_engine_cache_dir() -> str:
    """Persistent TensorRT engine-cache dir (so the multi-minute build is one-time)."""
    try:
        from autoptz.config.store import default_config_dir

        cache = default_config_dir() / "trt_cache"
    except Exception:  # noqa: BLE001 — config import must never break inference
        cache = Path.home() / ".cache" / "AutoPTZ" / "trt_cache"
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        logger.debug("Could not create TRT cache dir %s", cache, exc_info=True)
    return str(cache)


def _wants_fp16(prefs: HardwarePrefs | None) -> bool:
    """FP16 is on unless the user explicitly forced fp32."""
    return prefs is None or prefs.precision != "fp32"


def effective_precision(ep: EP | str, prefs: HardwarePrefs | None = None) -> str:
    """Return "fp16"/"fp32" actually used for *ep* under (env-resolved) prefs.

    Accelerator EPs (CoreML/TensorRT/CUDA/DirectML/OpenVINO) run FP16 unless the
    user forced fp32; the CPU EP is always FP32. Lets the UI show configured →
    effective precision without querying the live session.
    """
    prefs = _resolve_prefs(prefs)
    if isinstance(ep, str):
        try:
            ep = EP(ep)
        except ValueError:
            return "fp32"
    return "fp16" if (ep in _ACCELERATED_EPS and _wants_fp16(prefs)) else "fp32"


def _provider_options(ep: EP, prefs: HardwarePrefs | None) -> dict[str, object]:
    """Per-EP acceleration options. Empty dict = provider defaults."""
    if ep is EP.COREML:
        # MLProgram routes to the Apple Neural Engine / GPU (incl. AMD on Intel
        # Macs via Metal); ALL lets CoreML pick the fastest unit per op.
        return {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}
    if ep is EP.TENSORRT:
        opts: dict[str, object] = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": _trt_engine_cache_dir(),
            "trt_timing_cache_enable": True,
        }
        if _wants_fp16(prefs):
            opts["trt_fp16_enable"] = True
        return opts
    if ep is EP.CUDA:
        return {"cudnn_conv_algo_search": "HEURISTIC"}
    if ep is EP.DIRECTML:
        return {"device_id": 0}
    if ep is EP.OPENVINO:
        return {
            "device_type": "AUTO",
            "precision": "FP16" if _wants_fp16(prefs) else "FP32",
        }
    return {}


def _build_session_options(
    prefs: HardwarePrefs | None,
    base: ort.SessionOptions | None,
) -> ort.SessionOptions:
    """Return tuned SessionOptions (full graph opt + optional thread cap)."""
    so = base if base is not None else ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if prefs and prefs.intra_op_threads:
        so.intra_op_num_threads = max(1, int(prefs.intra_op_threads))
    return so


def _provider_list(
    chosen: EP, prefs: HardwarePrefs | None
) -> list[tuple[str, dict[str, object]] | str]:
    """Chosen EP (with options) followed by a bare-CPU fallback."""
    providers: list[tuple[str, dict[str, object]] | str] = [
        (chosen.value, _provider_options(chosen, prefs))
    ]
    if chosen is not EP.CPU:
        providers.append(EP.CPU.value)
    return providers


def make_session(
    model_path: str | Path,
    prefs: HardwarePrefs | None = None,
    session_options: ort.SessionOptions | None = None,
) -> ort.InferenceSession:
    """Return a tuned ORT InferenceSession using the best available EP.

    Tries the chosen EP with acceleration options first; if those options are
    rejected (older driver / EP build), retries with bare providers; finally
    falls back CPU-ward.  Every downgrade is logged so the UI can surface what is
    actually running.
    """
    prefs = _resolve_prefs(prefs)
    chosen = get_best_ep(prefs)
    so = _build_session_options(prefs, session_options)
    name = Path(model_path).name

    precision = "fp16" if (chosen in _ACCELERATED_EPS and _wants_fp16(prefs)) else "fp32"
    logger.info("Creating ORT session | model=%s ep=%s precision=%s", name, chosen.value, precision)

    attempts: list[tuple[str, list[tuple[str, dict[str, object]] | str]]] = [
        ("tuned", _provider_list(chosen, prefs)),
    ]
    # Bare chosen EP (no options), then CPU-only — progressively safer fallbacks.
    if chosen is not EP.CPU:
        attempts.append(("bare", [chosen.value, EP.CPU.value]))
    attempts.append(("cpu", [EP.CPU.value]))

    last_exc: Exception | None = None
    for label, providers in attempts:
        try:
            session = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
        except Exception as exc:  # noqa: BLE001 — try the next, safer provider set
            last_exc = exc
            logger.warning("ORT session attempt '%s' failed (%s); trying fallback", label, exc)
            continue
        actual_ep = session.get_providers()[0]
        if actual_ep != chosen.value:
            logger.warning(
                "ORT EP downgraded: requested=%s actual=%s (model=%s)",
                chosen.value,
                actual_ep,
                name,
            )
        else:
            logger.info(
                "ORT session active | ep=%s precision=%s model=%s", actual_ep, precision, name
            )
        return session

    # Exhausted every fallback — surface the original failure.
    raise RuntimeError(f"Could not create ORT session for {name}") from last_exc
