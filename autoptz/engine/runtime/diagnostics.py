"""Runtime diagnostics: service availability + live system metrics.

This is the single place the UI asks "what's actually running?" — detector
model, inference EP, tracker backend, face recognition, ReID — and "how hard is
the machine working?" (system + per-process CPU / memory).

Every probe is defensive: a missing optional dependency is reported as an
unavailable/degraded service, never an exception.  Imports that could be heavy
(insightface) are detected with :func:`importlib.util.find_spec` so opening the
Services panel never blocks on a model load.

State vocabulary (the UI maps these to colours):
    "ok"      — healthy / fully functional        (green)
    "warn"    — degraded but functional fallback   (amber)
    "off"     — unavailable / not installed         (grey/red)
    "running" / "stopped" — engine lifecycle        (green / grey)
"""

from __future__ import annotations

import importlib.util
import logging
import os
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _entry(key: str, name: str, state: str, detail: str) -> dict[str, str]:
    return {"key": key, "name": name, "state": state, "detail": detail}


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001 — a broken meta-path finder must not crash us
        return False


# ── individual service probes ───────────────────────────────────────────────────


def inference_status() -> dict[str, str]:
    """ONNX Runtime presence + available execution providers (best first)."""
    try:
        import onnxruntime as ort  # noqa: PLC0415

        provs = list(ort.get_available_providers())
        labels = [p.replace("ExecutionProvider", "") for p in provs] or ["none"]
        return _entry("inference", "Inference runtime", "ok", "ONNX Runtime · " + ", ".join(labels))
    except Exception:  # noqa: BLE001
        return _entry("inference", "Inference runtime", "off", "onnxruntime not importable")


def detector_model_status() -> dict[str, str]:
    """Whether a usable detector ONNX is present (without triggering a download)."""
    env = os.environ.get("AUTOPTZ_MODEL_PATH")
    if env and Path(env).is_file():
        return _entry("detector", "Detector model", "ok", f"AUTOPTZ_MODEL_PATH · {Path(env).name}")
    try:
        from autoptz.engine.runtime.models import default_manager  # noqa: PLC0415

        rows = [
            row
            for row in default_manager().app_model_statuses()
            if row.get("kind") == "detector" and row.get("cached")
        ]
        if rows:
            names = ", ".join(str(row.get("label") or row.get("name")) for row in rows)
            return _entry(
                "detector",
                "Detector model",
                "ok",
                f"{len(rows)} detector tier(s) cached · {names}",
            )
        return _entry("detector", "Detector model", "off", "not downloaded - open Engine > Models")
    except Exception:  # noqa: BLE001
        return _entry("detector", "Detector model", "off", "lookup failed")


def tracker_status() -> dict[str, str]:
    """Tracker backend: BoT-SORT (boxmot) or the built-in lightweight fallback.

    Detected with ``find_spec`` only — importing boxmot pulls in torch (multi-
    second) and this probe runs on the GUI thread's Services-panel poll, so a
    real import here would freeze the event loop during startup.
    """
    if _module_present("boxmot"):
        return _entry("tracker", "Tracker", "ok", "BoT-SORT (boxmot)")
    return _entry(
        "tracker", "Tracker", "warn", "Lightweight IoU fallback · install boxmot for BoT-SORT"
    )


def reid_status() -> dict[str, str]:
    """Appearance ReID (OSNet) recovery after occlusion — provided by boxmot."""
    if _module_present("boxmot"):
        return _entry(
            "reid", "ReID (re-acquire)", "ok", "OSNet (boxmot) · required for Stable tracking"
        )
    return _entry(
        "reid", "ReID (re-acquire)", "off", "needs boxmot / OSNet — Stable tracking unavailable"
    )


def face_status() -> dict[str, str]:
    """Face recognition (insightface SCRFD + ArcFace) availability."""
    if _module_present("insightface"):
        return _entry("face", "Face recognition", "ok", "insightface SCRFD + ArcFace")
    return _entry(
        "face",
        "Face recognition",
        "off",
        "insightface not installed · manual click-to-track still works",
    )


def pose_status() -> dict[str, str]:
    """Pose model/dependency availability for skeleton + torso-stable aim."""
    try:
        from autoptz.engine.runtime.models import default_manager  # noqa: PLC0415

        onnx = default_manager().cache_dir / "yolo11n-pose.onnx"
        if onnx.is_file():
            size_mb = onnx.stat().st_size / (1 << 20)
            return _entry("pose", "Pose model", "ok", f"{onnx.name} · {size_mb:.1f} MB")
        if _module_present("ultralytics"):
            return _entry("pose", "Pose model", "warn", f"not cached · can export to {onnx}")
        return _entry(
            "pose",
            "Pose model",
            "off",
            f"not cached · needs bundled model or ultralytics export to {onnx}",
        )
    except Exception:  # noqa: BLE001
        return _entry("pose", "Pose model", "off", "lookup failed")


def optional_components() -> list[dict[str, str]]:
    """Detailed optional setup rows for ServicesPanel setup actions.

    These rows intentionally describe model/download details without performing
    downloads or package installs; the app stays usable while Services can offer
    an explicit model setup action with progress.
    """
    rows = []
    try:
        from autoptz.engine.runtime.models import default_manager  # noqa: PLC0415

        cache = default_manager().cache_dir
    except Exception:  # noqa: BLE001
        cache = Path("AutoPTZ/models")

    detector = detector_model_status()
    rows.append(
        {
            **detector,
            "source": "YOLO11 detector ONNX tiers",
            "size": "varies by selected tier",
            "path": str(cache),
            "why": "Person boxes, click-to-track, and all automatic PTZ following.",
            "managed": "AutoPTZ-managed cache; can be downloaded or removed here.",
            "network": "Can be bundled offline or exported from ultralytics.",
        }
    )

    reid = reid_status()
    rows.append(
        {
            **reid,
            "source": "boxmot OSNet weights",
            "size": "varies by tracker package",
            "path": str(cache / "reid"),
            "why": "Stable re-acquire after occlusion or crowds.",
            "managed": (
                "Managed by boxmot/torch upstream caches; AutoPTZ never deletes the "
                "files but unloads ReID from memory when its feature is off."
            ),
            "network": "May contact package/model hosts when prepared.",
        }
    )

    pose = pose_status()
    rows.append(
        {
            **pose,
            "source": "YOLO11n-pose ONNX",
            "size": "small model bundle",
            "path": str(cache / "yolo11n-pose.onnx"),
            "why": "Skeleton overlay and torso-stable framing.",
            "managed": "AutoPTZ-managed cache; can be downloaded or removed here.",
            "network": "Can be bundled offline or exported from ultralytics.",
        }
    )

    face = face_status()
    rows.append(
        {
            **face,
            "source": "insightface buffalo_l (SCRFD + ArcFace)",
            "size": "face model pack",
            "path": str(Path.home() / ".insightface" / "models"),
            "why": "Named-person confirmation and face identity matching.",
            "managed": (
                "Managed by the insightface upstream cache; AutoPTZ never deletes the "
                "files but unloads face recognition from memory when its feature is off."
            ),
            "network": "insightface may download its model pack on first prepare.",
        }
    )
    return rows


def engine_status(running: bool, ep: str) -> dict[str, str]:
    if running:
        return _entry("engine", "Engine", "running", f"running{(' · ' + ep) if ep else ''}")
    return _entry("engine", "Engine", "stopped", "stopped")


def collect_services(*, engine_running: bool, engine_ep: str) -> list[dict[str, str]]:
    """Return the ordered list of service-status rows for the Services panel."""
    return [
        engine_status(engine_running, engine_ep),
        inference_status(),
        detector_model_status(),
        tracker_status(),
        reid_status(),
        pose_status(),
        face_status(),
    ]


# ── live system metrics (psutil, optional) ──────────────────────────────────────

_PROC: Any | None = None
_PRIMED = False

# A short snapshot cache so multiple callers in the same beat share ONE sample.
# ``psutil.cpu_percent(interval=None)`` returns the load since the *previous*
# call, so when the status bar and the Services panel each call this on their own
# ~1.5 s timer, every other call sees a tiny/irregular delta window and the two
# readouts diverge.  Caching the computed dict for a fraction of the poll period
# means both surfaces read identical numbers and the CPU delta is measured over a
# stable ~1.5 s window (the gap between actual recomputes).
_CACHE: dict[str, Any] | None = None
_CACHE_T: float = 0.0
_CACHE_TTL_S = 1.0


def system_metrics() -> dict[str, Any]:
    """Return live CPU / memory metrics (system-wide + this process).

    Uses ``psutil`` when available.  ``cpu_percent`` is sampled relative to the
    previous *recompute*, so the first call after start returns 0 and subsequent
    polls (the UI polls ~1 Hz) are meaningful.  Results are cached for
    :data:`_CACHE_TTL_S` so the status bar and Services panel report identical
    numbers.  Degrades to ``{"available": False}`` when psutil is missing so the
    status bar simply shows placeholders.
    """
    global _PROC, _PRIMED, _CACHE, _CACHE_T
    now = time.monotonic()
    if _CACHE is not None and (now - _CACHE_T) < _CACHE_TTL_S:
        return dict(_CACHE)

    out: dict[str, Any] = {
        "available": False,
        "cpu_percent": 0.0,
        "mem_percent": 0.0,
        "app_cpu_percent": 0.0,
        "app_rss_mb": 0.0,
        "app_mem_percent": 0.0,
    }
    try:
        import psutil  # noqa: PLC0415
    except Exception:  # noqa: BLE001 — psutil optional
        _CACHE, _CACHE_T = out, now
        return dict(out)

    try:
        if _PROC is None:
            _PROC = psutil.Process(os.getpid())
        ncpu = psutil.cpu_count(logical=True) or 1

        if not _PRIMED:
            # Prime the deltas so the next poll reports real numbers.
            psutil.cpu_percent(interval=None)
            _PROC.cpu_percent(interval=None)
            _PRIMED = True

        out["available"] = True
        out["cpu_percent"] = round(float(psutil.cpu_percent(interval=None)), 1)
        vm = psutil.virtual_memory()
        out["mem_percent"] = round(float(vm.percent), 1)
        # Process CPU can exceed 100% across cores; normalise to the whole machine.
        out["app_cpu_percent"] = round(
            float(_PROC.cpu_percent(interval=None)) / float(ncpu),
            1,
        )
        rss = _PROC.memory_info().rss
        out["app_rss_mb"] = round(rss / (1 << 20), 1)
        total = max(1, int(getattr(vm, "total", 0) or 0))
        out["app_mem_percent"] = round((float(rss) / float(total)) * 100.0, 1)
    except Exception:  # noqa: BLE001
        log.debug("system_metrics sampling failed", exc_info=True)
    _CACHE, _CACHE_T = out, now
    return dict(out)
