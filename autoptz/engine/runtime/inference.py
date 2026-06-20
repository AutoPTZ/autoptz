"""ONNX Runtime session factory: pick the best available EP per platform/hardware.

Selection order:
  macOS (Apple Silicon): CoreML → CPU
  Windows:               TensorRT → CUDA → DirectML → OpenVINO → CPU
  Linux:                 TensorRT → CUDA → OpenVINO → CPU
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import onnxruntime as ort

logger = logging.getLogger(__name__)


class EP(str, Enum):
    COREML = "CoreMLExecutionProvider"
    TENSORRT = "TensorrtExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    DIRECTML = "DmlExecutionProvider"
    OPENVINO = "OpenVINOExecutionProvider"
    CPU = "CPUExecutionProvider"


@dataclass
class HardwarePrefs:
    """Per-camera or global hardware preferences."""

    force_ep: EP | None = None


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


def make_session(
    model_path: str | Path,
    prefs: HardwarePrefs | None = None,
    session_options: ort.SessionOptions | None = None,
) -> ort.InferenceSession:
    """Return an ORT InferenceSession using the best available EP.

    Falls back CPU-ward on failure and logs the downgrade so the UI can
    surface "running on CPU."
    """
    chosen = get_best_ep(prefs)
    providers: list[str] = [chosen.value]
    if chosen != EP.CPU:
        providers.append(EP.CPU.value)  # always include CPU as final fallback

    logger.info(
        "Creating ORT session | model=%s ep=%s",
        Path(model_path).name,
        chosen.value,
    )

    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=providers,
        )
    except Exception as exc:
        if chosen != EP.CPU:
            logger.warning("EP %s failed (%s); downgrading to CPU", chosen.value, exc)
            session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=[EP.CPU.value],
            )
        else:
            raise

    actual_ep = session.get_providers()[0]
    if actual_ep != chosen.value:
        logger.warning(
            "ORT silently downgraded EP: requested=%s actual=%s",
            chosen.value,
            actual_ep,
        )
    else:
        logger.info("ORT session active | ep=%s model=%s", actual_ep, Path(model_path).name)

    return session
