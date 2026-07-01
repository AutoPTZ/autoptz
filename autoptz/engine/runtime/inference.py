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
import platform
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import onnxruntime as ort

logger = logging.getLogger(__name__)

Precision = Literal["auto", "fp32", "fp16", "int8"]


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
    if precision_raw in ("fp32", "fp16", "int8"):
        precision = precision_raw  # type: ignore[assignment]
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


def _cache_dir(subdir: str) -> str:
    """Persistent per-machine cache dir for an EP's compiled artifacts."""
    try:
        from autoptz.config.store import default_config_dir

        cache = default_config_dir() / subdir
    except Exception:  # noqa: BLE001 — config import must never break inference
        cache = Path.home() / ".cache" / "AutoPTZ" / subdir
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        logger.debug("Could not create cache dir %s", cache, exc_info=True)
    return str(cache)


def _trt_engine_cache_dir() -> str:
    """Persistent TensorRT engine-cache dir (so the multi-minute build is one-time)."""
    return _cache_dir("trt_cache")


def _coreml_cache_dir() -> str:
    """Persistent CoreML compiled-model cache dir (skips per-session MLProgram compile)."""
    return _cache_dir("coreml_cache")


def _wants_fp16(prefs: HardwarePrefs | None) -> bool:
    """FP16 is on for auto/fp16 (not fp32, and not int8 where the model is quantized)."""
    return prefs is None or prefs.precision in ("auto", "fp16")


def _ep_can_run_fp16(ep: EP) -> bool:
    """Return True only when *ep* genuinely executes the model in FP16 on this host.

    Rules (conservative — when in doubt we report FP32 so the UI never lies):

    - **CoreML**: FP16 via MLProgram only on Apple Silicon (arm64/aarch64).
      On Intel Macs the ANE is absent; CoreML effectively runs on the CPU in
      FP32 even with ``MLComputeUnits=ALL``.
    - **TensorRT / CUDA**: always FP16-capable (engine built with FP16 flags).
    - **DirectML**: passes the FP32 ONNX model through without converting it,
      so the compute stays in FP32.
    - **OpenVINO**: reports FP32 conservatively.  The EP can run FP16 on a
      dedicated GPU/NPU, but the device resolved at runtime may be "CPU"
      (common on laptops).  Without querying the live session we cannot know
      which device was actually chosen, so we err on the side of honesty.
    - **CPU**: FP32 always.
    """
    if ep is EP.COREML:
        machine = platform.machine().lower()
        return machine in ("arm64", "aarch64")
    if ep in (EP.TENSORRT, EP.CUDA):
        return True
    # DirectML, OpenVINO, CPU — FP32
    return False


def effective_precision(ep: EP | str, prefs: HardwarePrefs | None = None) -> str:
    """Return "fp16"/"fp32"/"int8" actually used for *ep* under (env-resolved) prefs.

    Precision is determined by both *what the user requested* and *what the EP
    can genuinely deliver on this host*:

    - If the user forced ``int8`` that is always returned verbatim (quantised
      model path).
    - If the user forced ``fp32`` that overrides any EP capability.
    - Otherwise the EP's true hardware capability is consulted via
      :func:`_ep_can_run_fp16`; only EPs that genuinely run FP16 on the
      current host report ``fp16``.

    This means the Camera Info / About panels never claim FP16 when the
    hardware is quietly running FP32 (e.g. CoreML on an Intel Mac, DirectML,
    OpenVINO with a CPU device).
    """
    prefs = _resolve_prefs(prefs)
    if prefs is not None and prefs.precision == "int8":
        return "int8"
    if isinstance(ep, str):
        try:
            ep = EP(ep)
        except ValueError:
            return "fp32"
    if not _wants_fp16(prefs):
        return "fp32"
    return "fp16" if _ep_can_run_fp16(ep) else "fp32"


_VALID_COREML_UNITS = ("ALL", "CPUAndGPU", "CPUOnly", "CPUAndNeuralEngine")


def _coreml_compute_units() -> str:
    """CoreML compute-unit target. Defaults to ``ALL`` (CoreML picks the fastest
    unit per op; on Intel Macs ``ALL`` includes the discrete AMD GPU via Metal).

    Override with ``AUTOPTZ_COREML_UNITS`` to diagnose or force the path on
    Intel + AMD (e.g. iMac Pro Xeon + Vega) Macs — ``CPUOnly`` to *measure* whether
    the GPU is helping at all (compare two ``--bench`` runs), or ``CPUAndGPU`` to
    pin the discrete GPU if ``ALL`` is silently choosing the CPU. Invalid values
    fall back to ``ALL``.
    """
    import os  # noqa: PLC0415

    val = os.environ.get("AUTOPTZ_COREML_UNITS", "").strip().lower()
    for v in _VALID_COREML_UNITS:
        if val == v.lower():
            return v
    return "ALL"


def _provider_options(ep: EP, prefs: HardwarePrefs | None) -> dict[str, object]:
    """Per-EP acceleration options. Empty dict = provider defaults."""
    if ep is EP.COREML:
        # MLProgram routes to the Apple Neural Engine / GPU (incl. AMD on Intel
        # Macs via Metal); the compute-unit target is tunable (AUTOPTZ_COREML_UNITS)
        # so an Intel+AMD Mac can verify/force the GPU path.
        coreml_opts: dict[str, object] = {
            "ModelFormat": "MLProgram",
            "MLComputeUnits": _coreml_compute_units(),
        }
        # ModelCacheDirectory persists the compiled MLProgram so the slow first
        # compile is one-time per machine — BUT spawned model-server camera
        # children can fail the CoreML EP with it set ("Failed to create model URL
        # from path"). Omit the on-disk cache in that process mode: the child
        # still runs on the ANE/GPU, it just recompiles its MLProgram each start.
        # The shared in-process path keeps the cache.
        from autoptz.engine.runtime.flags import env_process_per_camera

        if not env_process_per_camera():
            coreml_opts["ModelCacheDirectory"] = _coreml_cache_dir()
        return coreml_opts
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


def _apply_low_idle_threading(so: ort.SessionOptions) -> None:
    """Stop ORT worker threads from busy-spinning while idle.

    AutoPTZ runs each model only intermittently (detect every Nth frame, pose/face
    a few Hz), so an ORT session spends most of its life *idle between runs*.  By
    default ORT's intra-op worker threads **busy-spin** (~200 ms) before parking,
    which on a multi-camera box turns several mostly-idle sessions into a wall of
    CPU — profiling showed ``ThreadPoolTempl::WorkerLoop`` as the single largest
    consumer.  Disabling spinning makes workers block immediately on a condition
    variable instead, collapsing idle/intermittent CPU at the cost of a few hundred
    microseconds of wake-up latency per run (negligible next to a 10-40 ms model).

    We also force a single, sequential inter-op pool: AutoPTZ runs one graph per
    call, so a parallel inter-op pool only adds another idle spinner.
    """
    # Keys are stable ORT session-config entries (>=1.6); unknown keys are ignored
    # by older runtimes, so this stays safe across the EP matrix.
    try:
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    except Exception:  # noqa: BLE001 — a tuning hint must never block session build
        pass
    try:
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
    except Exception:  # noqa: BLE001
        pass


def _build_session_options(
    prefs: HardwarePrefs | None,
    base: ort.SessionOptions | None,
) -> ort.SessionOptions:
    """Return tuned SessionOptions (full graph opt + optional thread cap)."""
    so = base if base is not None else ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if prefs and prefs.intra_op_threads:
        so.intra_op_num_threads = max(1, int(prefs.intra_op_threads))
    _apply_low_idle_threading(so)
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

    precision = effective_precision(chosen, prefs)
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
