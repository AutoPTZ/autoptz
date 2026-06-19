"""Model discovery & bootstrap for the detection stack.

``ModelManager`` is the single place that owns "where do the model weights
live, and how do we get one if it's missing".  Today it manages the YOLO11
person-detection ONNX; face/ReID weights auto-download into their own caches
(`~/.insightface`, boxmot's cache) and are out of scope here.

Design
------
- Cache dir is the platform app-data dir (the same resolution
  :func:`autoptz.config.store.default_config_dir` uses) under ``models/``.
- :meth:`ensure_detector` returns a path to a usable YOLO11 person/COCO
  detection ONNX.  Resolution order, **torch-free first**:

    1. ``AUTOPTZ_MODEL_PATH`` env override (existing file → returned verbatim).
    2. A previously-cached ONNX in the cache dir.
    3. **Download a prebuilt YOLO11 ONNX** (no torch / ultralytics needed) from
       a reliable HuggingFace-hosted export.  URL overridable via
       ``AUTOPTZ_MODEL_URL``.
    4. Fall back to the ultralytics ``.pt`` → ONNX export only when the prebuilt
       download is unreachable *and* ultralytics is installed.

  It is **never** fatal: any failure (no network, no ultralytics, export error)
  is logged and returns ``None`` so the engine degrades to live-preview-only.

The ONNX we prefer is the **NMS-free** ``[1, N, 6]`` layout that
:mod:`autoptz.engine.pipeline.detect` prefers (``_parse_raw_output`` handles
both NMS-free and pre-NMS, but NMS-free avoids the ultralytics batched-NMS
export bug; see ``detect.py`` and the v2 plan §"Recommended libraries").
"""
from __future__ import annotations

import logging
import os
import tempfile
import urllib.request
from pathlib import Path
from threading import Lock

log = logging.getLogger(__name__)

# Default detector: YOLO11 nano — smallest/fastest production-viable pick.
# The exported/downloaded ONNX file name is derived from the stem ("yolo11n.onnx").
_DEFAULT_DETECTOR_PT = "yolo11n.pt"
_DETECTOR_TIER_TO_PT = {
    "auto": _DEFAULT_DETECTOR_PT,
    "fast": "yolo11n.pt",
    "nano": "yolo11n.pt",
    "balanced": "yolo11s.pt",
    "small": "yolo11s.pt",
}


def detector_model_for_tier(tier: str | None) -> str:
    """Return the detector weight name for a user-facing model tier."""
    key = str(tier or "auto").strip().lower()
    return _DETECTOR_TIER_TO_PT.get(key, _DEFAULT_DETECTOR_PT)

# Optional prebuilt, torch-free YOLO11n ONNX URL.  No reliable public URL is
# wired by default (HuggingFace `resolve` links for community exports returned
# 401), so this is empty and acquisition falls through to the ultralytics export.
# Set AUTOPTZ_MODEL_URL to a reachable .onnx to enable the torch-free path
# (air-gapped / mirror / bundled deployments).
_DEFAULT_PREBUILT_URL = ""

# Download chunk size (bytes) when streaming the prebuilt ONNX to disk.
_DOWNLOAD_CHUNK = 1 << 16

# A sane lower bound (bytes) for a real ONNX; guards against saving an HTML
# error page / truncated file as a "model".
_MIN_ONNX_BYTES = 1 << 18  # 256 KiB

# Export knobs that match what detect.py expects (see module docstring).
_EXPORT_KWARGS = dict(format="onnx", nms=False, dynamic=False, opset=12)


def _models_cache_dir() -> Path:
    """Return the platform app-data ``…/AutoPTZ/models`` directory.

    Reuses :func:`autoptz.config.store.default_config_dir` so the model cache
    sits next to the config DB on every platform.
    """
    from autoptz.config.store import default_config_dir

    return default_config_dir() / "models"


class ModelManager:
    """Resolves / downloads / exports the models the engine needs.

    Args:
        cache_dir: Override the model cache directory (tests pass a tmp dir).
                   Defaults to the platform app-data ``…/AutoPTZ/models``.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir is not None else _models_cache_dir()
        self._lock = Lock()

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def ensure_detector(
        self,
        *,
        model_pt: str | None = None,
        tier: str | None = None,
    ) -> str | None:
        """Return a path to a YOLO11 person/COCO detection ONNX, or ``None``.

        Resolution order (torch-free first):

        1. ``AUTOPTZ_MODEL_PATH`` env override → returned verbatim if it is an
           existing file.
        2. A previously-cached ONNX already in the cache dir.
        3. **Download a prebuilt ONNX** (no torch / ultralytics) from
           ``AUTOPTZ_MODEL_URL`` (or the default HuggingFace export).
        4. Fall back to the ultralytics ``.pt`` → ONNX export only if the
           prebuilt download is unreachable.

        Never raises: a missing ultralytics / model / network logs a warning
        and returns ``None`` so the engine still delivers live preview.
        """
        # 1. Env override wins outright.
        env = os.environ.get("AUTOPTZ_MODEL_PATH")
        if env:
            if Path(env).is_file():
                log.info("Using detector model from AUTOPTZ_MODEL_PATH=%s", env)
                return env
            log.warning(
                "AUTOPTZ_MODEL_PATH=%s is set but not an existing file; ignoring.",
                env,
            )

        model_pt = model_pt or detector_model_for_tier(tier)
        onnx_path = self._cache_dir / (Path(model_pt).stem + ".onnx")

        # 2. Already cached earlier → reuse.
        if onnx_path.is_file():
            log.debug("Detector ONNX already cached at %s", onnx_path)
            return str(onnx_path)

        # 3+4. Acquire under a lock so concurrent camera workers don't race to
        #      fetch/export the same file.
        with self._lock:
            # Re-check inside the lock — another thread may have just produced it.
            if onnx_path.is_file():
                return str(onnx_path)

            # Prefer the torch-free prebuilt download.
            prebuilt = self._download_prebuilt(onnx_path)
            if prebuilt is not None:
                return prebuilt

            # Last resort: export from ultralytics (.pt) if it is installed.
            log.info(
                "Prebuilt detector ONNX unavailable; trying ultralytics export.",
            )
            return self._download_and_export(model_pt, onnx_path)

    def ensure_pose(self, *, model_pt: str = "yolo11n-pose.pt") -> str | None:
        """Return a path to a YOLO11 pose ONNX, downloading/exporting if needed.

        Mirrors :meth:`ensure_detector` for the COCO-17 pose model: env override
        (``AUTOPTZ_POSE_MODEL_PATH``) → cached ``yolo11n-pose.onnx`` → ultralytics
        ``.pt`` → ONNX export (one-time).  Unlike the original "never download"
        pose policy, pose now auto-provisions on first use so enabling the pose
        overlay / pose-stable framing actually works out of the box.  Never raises;
        returns ``None`` when ultralytics/network are unavailable (pose stays off).
        """
        env = os.environ.get("AUTOPTZ_POSE_MODEL_PATH")
        if env:
            if Path(env).is_file():
                return env
            log.warning("AUTOPTZ_POSE_MODEL_PATH=%s set but not a file; ignoring.", env)

        onnx_path = self._cache_dir / (Path(model_pt).stem + ".onnx")
        if onnx_path.is_file():
            return str(onnx_path)
        with self._lock:
            if onnx_path.is_file():
                return str(onnx_path)
            return self._download_and_export(model_pt, onnx_path)

    # ── internals ─────────────────────────────────────────────────────────────

    def _download_prebuilt(self, onnx_path: Path) -> str | None:
        """Download the prebuilt YOLO11 ONNX (no torch) to *onnx_path*.

        Streams ``AUTOPTZ_MODEL_URL`` (default HuggingFace export) to a temp
        file in the cache dir, then atomically moves it into place once the
        download is complete and looks like a real model.  Returns the path on
        success, ``None`` on any failure (logged) so ``ensure_detector`` can
        fall back to the ultralytics export.  Never raises.
        """
        url = os.environ.get("AUTOPTZ_MODEL_URL", _DEFAULT_PREBUILT_URL)
        if not url:
            return None

        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            log.warning("Could not create model cache dir %s", self._cache_dir,
                        exc_info=True)
            return None

        tmp_fd, tmp_name = tempfile.mkstemp(
            suffix=".onnx.part", dir=str(self._cache_dir),
        )
        tmp_path = Path(tmp_name)
        try:
            log.info("Downloading prebuilt detector ONNX from %s (first run, "
                     "~one-time)", url)
            written = 0
            with urllib.request.urlopen(url) as resp, os.fdopen(tmp_fd, "wb") as out:  # noqa: S310
                while True:
                    chunk = resp.read(_DOWNLOAD_CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    written += len(chunk)

            if written < _MIN_ONNX_BYTES:
                log.warning(
                    "Prebuilt detector download from %s looked truncated "
                    "(%d bytes); ignoring.", url, written,
                )
                return None

            tmp_path.replace(onnx_path)
            log.info("Detector ONNX ready at %s (%.1f MB, prebuilt, torch-free)",
                     onnx_path, written / (1 << 20))
            return str(onnx_path)
        except Exception:  # noqa: BLE001 — network / disk / URL errors
            log.warning(
                "Prebuilt detector download failed; will try ultralytics export.",
                exc_info=True,
            )
            return None
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:  # noqa: BLE001
                log.debug("Could not remove temp download %s", tmp_path,
                          exc_info=True)

    def _download_and_export(self, model_pt: str, onnx_path: Path) -> str | None:
        """Download the ultralytics ``.pt`` and export it to *onnx_path*.

        Returns the ONNX path on success, ``None`` on any failure (logged).
        """
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except Exception:  # noqa: BLE001 — ImportError or transitive import failure
            log.warning(
                "ultralytics not installed; detector unavailable "
                "(live-preview-only). Install it or pre-fetch with "
                "`python -m tools.fetch_models`.",
            )
            return None

        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            log.warning("Could not create model cache dir %s", self._cache_dir,
                        exc_info=True)
            return None

        # ultralytics downloads weights into the *current* working directory by
        # default and exports the ONNX next to the .pt.  Drive both from the
        # cache dir so nothing lands in the repo / cwd.
        prev_cwd = Path.cwd()
        try:
            os.chdir(self._cache_dir)
        except Exception:  # noqa: BLE001
            log.warning("Could not chdir to cache dir %s", self._cache_dir,
                        exc_info=True)
            return None

        try:
            log.info("Downloading + exporting detector %s → %s (first run, ~one-time)",
                     model_pt, onnx_path.name)
            model = YOLO(model_pt)  # downloads the .pt if missing
            exported = model.export(**_EXPORT_KWARGS)
            # ultralytics returns the exported path (str/Path) on success.
            exported_path = Path(exported) if exported else (self._cache_dir / onnx_path.name)
            if exported_path.is_file() and exported_path != onnx_path:
                try:
                    exported_path.replace(onnx_path)
                except Exception:  # noqa: BLE001
                    # If the rename fails, fall back to whatever path exists.
                    if exported_path.is_file():
                        log.info("Detector ONNX ready at %s", exported_path)
                        return str(exported_path)
        except Exception:  # noqa: BLE001 — network/export/runtime errors
            log.warning(
                "Detector model download/export failed; running "
                "live-preview-only. Pre-fetch offline with "
                "`python -m tools.fetch_models`.",
                exc_info=True,
            )
            return None
        finally:
            try:
                os.chdir(prev_cwd)
            except Exception:  # noqa: BLE001
                log.debug("Could not restore cwd to %s", prev_cwd, exc_info=True)

        if onnx_path.is_file():
            log.info("Detector ONNX ready at %s", onnx_path)
            return str(onnx_path)

        log.warning("Detector export reported success but %s is missing.", onnx_path)
        return None


# ── module-level convenience ─────────────────────────────────────────────────

_DEFAULT_MANAGER: ModelManager | None = None
_DEFAULT_MANAGER_LOCK = Lock()


def default_manager() -> ModelManager:
    """Return a process-wide shared :class:`ModelManager` (lazy singleton)."""
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        with _DEFAULT_MANAGER_LOCK:
            if _DEFAULT_MANAGER is None:
                _DEFAULT_MANAGER = ModelManager()
    return _DEFAULT_MANAGER


def ensure_detector() -> str | None:
    """Convenience: resolve the detector ONNX via the shared manager."""
    return default_manager().ensure_detector()
