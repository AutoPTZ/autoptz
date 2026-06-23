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
       the project's ``models-v1`` GitHub release.  URL overridable via
       ``AUTOPTZ_MODEL_URL``.
    4. Fall back to the ultralytics ``.pt`` → ONNX export only when the prebuilt
       download is unreachable *and* ultralytics is installed.

  It is **never** fatal: any failure (no network, no ultralytics, export error)
  is logged and returns ``None`` so the engine degrades to live-preview-only.

The ONNX we prefer is the **NMS-free** ``[1, N, 6]`` layout that
:mod:`autoptz.engine.pipeline.detect` prefers (``_parse_raw_output`` handles
both NMS-free and pre-NMS, but NMS-free avoids the ultralytics batched-NMS
export bug; see ``detect.py``).
"""

from __future__ import annotations

import logging
import os
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import Any

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
    "medium": "yolo11m.pt",
    # RT-DETR (NMS-free, anchor-free) — exportable via ultralytics. The detector's
    # pre-NMS parser already handles its COCO output; select via tools.fetch_models
    # or point AUTOPTZ_MODEL_PATH at an exported RT-DETR ONNX.
    "rtdetr": "rtdetr-l.pt",
    "rtdetr-l": "rtdetr-l.pt",
    "rtdetr-x": "rtdetr-x.pt",
}
_APP_MODEL_SUFFIXES = (".onnx", ".int8.onnx", ".onnx.part", ".pt")
_APP_MODEL_CATALOG: tuple[dict[str, str], ...] = (
    {
        "key": "detector_fast",
        "kind": "detector",
        "tier": "fast",
        "name": "Fast detector",
        "weight": "yolo11n.pt",
        "stem": "yolo11n",
        "label": "YOLO11n",
        "cost": "Light",
        "description": "Lowest latency detector. Best default for most multi-camera setups.",
        "why": "Person boxes, click-to-track, tracking, and automatic PTZ follow.",
    },
    {
        "key": "detector_balanced",
        "kind": "detector",
        "tier": "balanced",
        "name": "Balanced detector",
        "weight": "yolo11s.pt",
        "stem": "yolo11s",
        "label": "YOLO11s",
        "cost": "Medium",
        "description": "Better detections than Fast with a moderate runtime cost.",
        "why": "Useful when people are small, partially blocked, or lighting is uneven.",
    },
    {
        "key": "detector_accurate",
        "kind": "detector",
        "tier": "medium",
        "name": "Accurate detector",
        "weight": "yolo11m.pt",
        "stem": "yolo11m",
        "label": "YOLO11m",
        "cost": "Heavy",
        "description": "Largest built-in detector tier. Best quality, highest CPU/GPU cost.",
        "why": "Useful for difficult rooms or longer camera shots when hardware allows it.",
    },
    {
        "key": "pose",
        "kind": "pose",
        "tier": "",
        "name": "Pose model",
        "weight": "yolo11n-pose.pt",
        "stem": "yolo11n-pose",
        "label": "YOLO11n-pose",
        "cost": "Light",
        "description": "Body keypoints for skeleton overlay and torso-stable framing.",
        "why": "Pose overlay and steadier framing when the tracked person turns or bends.",
    },
)
_APP_MODEL_STEMS = tuple(spec["stem"] for spec in _APP_MODEL_CATALOG)


def detector_model_for_tier(tier: str | None) -> str:
    """Return the detector weight name for a user-facing model tier."""
    key = str(tier or "auto").strip().lower()
    return _DETECTOR_TIER_TO_PT.get(key, _DEFAULT_DETECTOR_PT)


# Prebuilt, torch-free YOLO11 ONNX models served from the project's own
# ``models-v1`` GitHub release (exported with nms=False/opset=12 to match
# detect.py).  ``{stem}`` is filled per tier (yolo11n / yolo11s / yolo11m /
# yolo11n-pose), so the in-app Model Manager downloads work on packaged builds
# without bundling ultralytics + torch.  Override with AUTOPTZ_MODEL_URL (a
# ``{stem}`` template, or a single .onnx for the default tier) to point at a
# mirror / air-gapped host.
_DEFAULT_PREBUILT_URL = "https://github.com/AutoPTZ/autoptz/releases/download/models-v1/{stem}.onnx"

# Download chunk size (bytes) when streaming the prebuilt ONNX to disk.
_DOWNLOAD_CHUNK = 1 << 16

# A sane lower bound (bytes) for a real ONNX; guards against saving an HTML
# error page / truncated file as a "model".
_MIN_ONNX_BYTES = 1 << 18  # 256 KiB

# Export knobs that match what detect.py expects (see module docstring).
_EXPORT_KWARGS = {"format": "onnx", "nms": False, "dynamic": False, "opset": 12}
_DISABLE_EXPORT_ENV = "AUTOPTZ_NO_MODEL_EXPORT"
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
ModelProgress = Callable[[str, int, int], None]


def app_model_specs() -> list[dict[str, str]]:
    """Return metadata for AutoPTZ-managed model files."""
    return [dict(spec) for spec in _APP_MODEL_CATALOG]


def detector_key_for_tier(tier: str | None) -> str:
    """Return the catalog key for a detector tier."""
    model_pt = detector_model_for_tier(tier)
    stem = Path(model_pt).stem
    for spec in _APP_MODEL_CATALOG:
        if spec["kind"] == "detector" and spec["stem"] == stem:
            return spec["key"]
    return "detector_fast"


def _format_bytes(value: int) -> str:
    size = float(max(0, int(value)))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} GB"


def _model_export_disabled() -> bool:
    """Return True when runtime model export is explicitly disabled."""
    return os.environ.get(_DISABLE_EXPORT_ENV, "").strip().lower() in _TRUE_ENV_VALUES


def _models_cache_dir() -> Path:
    """Return the platform app-data ``…/AutoPTZ/models`` directory.

    Reuses :func:`autoptz.config.store.default_config_dir` so the model cache
    sits next to the config DB on every platform.
    """
    from autoptz.config.store import default_config_dir

    return default_config_dir() / "models"


def bundled_models_dir() -> Path:
    """Return the ``autoptz/models`` dir shipped *inside* the app/package.

    Release installers bake the detector + pose ONNX here (see
    ``packaging/autoptz.spec`` + the build scripts), so a packaged app finds its
    models with **no download** — detection works on first launch.  Empty for a
    plain source checkout (the ``.gitkeep``-only dir), where the prebuilt
    download / ultralytics export still provision models on demand.
    """
    return Path(__file__).resolve().parents[2] / "models"


class ModelManager:
    """Resolves / downloads / exports the models the engine needs.

    Args:
        cache_dir: Override the model cache directory (tests pass a tmp dir).
                   Defaults to the platform app-data ``…/AutoPTZ/models``.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir is not None else _models_cache_dir()
        self._lock = Lock()
        # Human-readable reason the most recent detector resolution returned
        # None ("" = no failure).  Surfaced to the UI so a silent fall-back to
        # live-preview-only doesn't read as "the model tier doesn't exist".
        self._last_error = ""

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def last_error(self) -> str:
        """Why the last :meth:`ensure_detector` failed, or "" if it succeeded."""
        return self._last_error

    def ensure_detector(
        self,
        *,
        model_pt: str | None = None,
        tier: str | None = None,
        allow_download: bool = True,
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
        self._last_error = ""
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
        # 2b. Shipped inside the app/package → use it, no download needed.
        bundled = bundled_models_dir() / onnx_path.name
        if bundled.is_file():
            log.debug("Using bundled detector ONNX at %s", bundled)
            return str(bundled)
        if not allow_download:
            self._last_error = (
                f"{onnx_path.name}: not cached. Open Engine > Models to download it, "
                "or enable automatic model downloads."
            )
            return None

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

    def ensure_detector_int8(self, fp32_onnx: str | Path) -> str | None:
        """Return an INT8 dynamically-quantized copy of *fp32_onnx*, cached.

        Used when ``precision == "int8"`` — a CPU-side win for some models (most
        impactful on transformer-heavy graphs; evaluate accuracy on YOLO with
        ``tools/bench``).  Never raises: returns ``None`` (caller keeps the FP32
        model) if onnxruntime's quantizer or the source file is unavailable.
        """
        src = Path(fp32_onnx)
        if not src.is_file():
            return None
        dst = src.with_name(src.stem + ".int8.onnx")
        if dst.is_file():
            return str(dst)
        with self._lock:
            if dst.is_file():
                return str(dst)
            tmp = dst.with_suffix(".onnx.part")
            try:
                from onnxruntime.quantization import QuantType, quantize_dynamic

                quantize_dynamic(str(src), str(tmp), weight_type=QuantType.QUInt8)
                if not tmp.is_file() or tmp.stat().st_size < _MIN_ONNX_BYTES:
                    log.warning("INT8 quantization of %s produced no/!tiny file", src.name)
                    return None
                tmp.replace(dst)
                log.info("INT8 detector ready at %s (%.1f MB)", dst, dst.stat().st_size / (1 << 20))
                return str(dst)
            except Exception:  # noqa: BLE001 — quantization is best-effort
                log.warning("INT8 quantization failed; using FP32 detector.", exc_info=True)
                return None
            finally:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:  # noqa: BLE001
                    log.debug("could not remove temp %s", tmp, exc_info=True)

    def ensure_pose(
        self,
        *,
        model_pt: str = "yolo11n-pose.pt",
        allow_download: bool = True,
    ) -> str | None:
        """Return a path to a YOLO11 pose ONNX, downloading/exporting if needed.

        Mirrors :meth:`ensure_detector` for the COCO-17 pose model: env override
        (``AUTOPTZ_POSE_MODEL_PATH``) → cached ``yolo11n-pose.onnx`` → **prebuilt
        torch-free download** → ultralytics ``.pt`` → ONNX export (one-time).
        Unlike the original "never download" pose policy, pose now auto-provisions
        on first use so enabling the pose overlay / pose-stable framing actually
        works out of the box.  Never raises; returns ``None`` when the model can't
        be obtained (pose stays off).
        """
        env = os.environ.get("AUTOPTZ_POSE_MODEL_PATH")
        if env:
            if Path(env).is_file():
                return env
            log.warning("AUTOPTZ_POSE_MODEL_PATH=%s set but not a file; ignoring.", env)

        onnx_path = self._cache_dir / (Path(model_pt).stem + ".onnx")
        if onnx_path.is_file():
            return str(onnx_path)
        bundled = bundled_models_dir() / onnx_path.name
        if bundled.is_file():
            return str(bundled)
        if not allow_download:
            self._last_error = (
                f"{onnx_path.name}: not cached. Open Engine > Models to download it, "
                "or enable automatic model downloads."
            )
            return None
        with self._lock:
            if onnx_path.is_file():
                return str(onnx_path)
            # Prefer the torch-free prebuilt download; fall back to ultralytics.
            prebuilt = self._download_prebuilt(onnx_path)
            if prebuilt is not None:
                return prebuilt
            return self._download_and_export(model_pt, onnx_path)

    def ensure_app_models(
        self,
        *,
        include_pose: bool = True,
        keys: list[str] | tuple[str, ...] | set[str] | None = None,
        progress: ModelProgress | None = None,
    ) -> list[dict[str, str]]:
        """Ensure the ONNX models AutoPTZ can manage directly are cached.

        This covers detector tiers exposed in Services plus the YOLO pose model.
        Face and ReID assets are owned by their upstream packages and have
        separate licensing/cache behavior, so diagnostics reports them but this
        method does not fetch them.
        """
        selected = {str(k) for k in keys} if keys is not None else None
        jobs: list[tuple[str, Callable[[], str | None], Callable[[], str]]] = []
        seen_detector_weights: set[str] = set()
        job_keys: list[str] = []
        for spec in _APP_MODEL_CATALOG:
            key = spec["key"]
            kind = spec["kind"]
            if selected is not None and key not in selected:
                continue
            if kind == "pose":
                continue
            tier = spec["tier"]
            label = spec["name"]
            model_pt = detector_model_for_tier(tier)
            if model_pt in seen_detector_weights:
                continue
            seen_detector_weights.add(model_pt)

            def _ensure_tier(selected_tier: str = tier) -> str | None:
                return self.ensure_detector(tier=selected_tier)

            def _detector_error(weight: str = model_pt) -> str:
                return self.last_error or f"{weight}: unavailable"

            jobs.append(
                (
                    label,
                    _ensure_tier,
                    _detector_error,
                )
            )
            job_keys.append(key)
        if include_pose and (selected is None or "pose" in selected):

            def _pose_error() -> str:
                return self.last_error or "yolo11n-pose.pt: unavailable"

            jobs.append(
                (
                    "Pose model",
                    self.ensure_pose,
                    _pose_error,
                )
            )
            job_keys.append("pose")

        results: list[dict[str, str]] = []
        total = len(jobs)
        for index, (label, work, error) in enumerate(jobs, start=1):
            if progress is not None:
                progress(label, index - 1, total)
            path = work()
            key = job_keys[index - 1] if index - 1 < len(job_keys) else ""
            results.append(
                {
                    "key": key,
                    "name": label,
                    "state": "ok" if path else "failed",
                    "path": str(path or ""),
                    "error": "" if path else error(),
                }
            )
            if progress is not None:
                progress(label, index, total)
        return results

    def app_model_statuses(self) -> list[dict[str, Any]]:
        """Return model catalog rows with current AutoPTZ cache state."""
        rows: list[dict[str, Any]] = []
        bundled_dir = bundled_models_dir()
        for spec in _APP_MODEL_CATALOG:
            row: dict[str, Any] = dict(spec)
            stem = spec["stem"]
            onnx_path = self._cache_dir / f"{stem}.onnx"
            bundled_path = bundled_dir / f"{stem}.onnx"
            files = self._managed_files_for_stem(stem)
            cached = onnx_path.is_file()
            # A model shipped inside the app counts as available with no download;
            # it lives in the read-only package dir, so it is NOT user-removable.
            bundled = bundled_path.is_file() and not cached
            available = cached or bundled
            active_path = onnx_path if cached else (bundled_path if bundled else onnx_path)
            size = 0
            for path in files if cached else [bundled_path] if bundled else []:
                try:
                    size += path.stat().st_size
                except OSError:
                    pass
            row["state"] = "ok" if available else "missing"
            row["cached"] = available
            row["bundled"] = bundled
            row["removable"] = cached
            row["path"] = str(active_path)
            row["cache_dir"] = str(self._cache_dir)
            row["size_bytes"] = size
            row["size"] = _format_bytes(size) if size else ""
            if bundled:
                row["detail"] = f"Included with AutoPTZ ({bundled_path.name})"
            elif cached:
                row["detail"] = f"Cached at {onnx_path}"
            else:
                row["detail"] = f"Missing from AutoPTZ cache ({onnx_path.name})"
            row["managed_files"] = [str(path) for path in files]
            rows.append(row)
        return rows

    def managed_model_files(
        self,
        *,
        keys: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> list[Path]:
        """Return AutoPTZ-managed model files currently present in the cache.

        Only files that AutoPTZ itself downloads/exports are listed. Upstream
        package caches such as ``~/.insightface`` are intentionally excluded so
        the app does not delete model packs that may be shared by other tools.
        """
        try:
            if not self._cache_dir.is_dir():
                return []
            out: list[Path] = []
            selected = {str(k) for k in keys} if keys is not None else None
            stems = [
                spec["stem"]
                for spec in _APP_MODEL_CATALOG
                if selected is None or spec["key"] in selected
            ]
            for stem in stems:
                out.extend(self._managed_files_for_stem(stem))
            return sorted(out)
        except Exception:  # noqa: BLE001
            log.debug("managed model inventory failed", exc_info=True)
            return []

    def remove_app_models(
        self,
        *,
        keys: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> list[dict[str, str]]:
        """Delete AutoPTZ-managed detector/pose model files from the cache."""
        removed: list[dict[str, str]] = []
        with self._lock:
            for path in self.managed_model_files(keys=keys):
                try:
                    size = path.stat().st_size
                    path.unlink()
                    removed.append(
                        {
                            "name": path.name,
                            "state": "removed",
                            "path": str(path),
                            "size": str(size),
                            "error": "",
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    removed.append(
                        {
                            "name": path.name,
                            "state": "failed",
                            "path": str(path),
                            "size": "",
                            "error": str(exc),
                        }
                    )
                    log.warning("could not remove model file %s", path, exc_info=True)
        return removed

    def _managed_files_for_stem(self, stem: str) -> list[Path]:
        out: list[Path] = []
        for suffix in _APP_MODEL_SUFFIXES:
            path = self._cache_dir / f"{stem}{suffix}"
            if path.is_file():
                out.append(path)
        return out

    # ── internals ─────────────────────────────────────────────────────────────

    def _prebuilt_url_for(self, stem: str) -> str:
        """Resolve a prebuilt-ONNX download URL for a model *stem* ("yolo11s").

        Resolution order so a torch-free mirror can serve *every* tier:
        1. Per-model env override ``AUTOPTZ_MODEL_URL_<STEM>`` (e.g.
           ``AUTOPTZ_MODEL_URL_YOLO11M``).
        2. ``AUTOPTZ_MODEL_URL`` with a ``{stem}``/``{model}`` placeholder filled
           in, so one template (``https://mirror/{stem}.onnx``) covers all tiers.
        3. A placeholder-free ``AUTOPTZ_MODEL_URL`` is honoured **only** for the
           default detector, so it can't fetch the wrong weights for balanced/
           medium.
        """
        per_model = os.environ.get(f"AUTOPTZ_MODEL_URL_{stem.upper().replace('-', '_')}")
        if per_model:
            return per_model
        base = os.environ.get("AUTOPTZ_MODEL_URL", _DEFAULT_PREBUILT_URL)
        if not base:
            return ""
        if "{stem}" in base or "{model}" in base:
            return base.format(stem=stem, model=stem)
        return base if stem == Path(_DEFAULT_DETECTOR_PT).stem else ""

    def _download_prebuilt(self, onnx_path: Path) -> str | None:
        """Download the prebuilt YOLO11 ONNX (no torch) to *onnx_path*.

        Streams the tier-resolved URL (see :meth:`_prebuilt_url_for`) to a temp
        file in the cache dir, then atomically moves it into place once the
        download is complete and looks like a real model.  Returns the path on
        success, ``None`` on any failure (logged) so ``ensure_detector`` can
        fall back to the ultralytics export.  Never raises.
        """
        url = self._prebuilt_url_for(onnx_path.stem)
        if not url:
            return None

        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            log.warning("Could not create model cache dir %s", self._cache_dir, exc_info=True)
            return None

        tmp_fd, tmp_name = tempfile.mkstemp(
            suffix=".onnx.part",
            dir=str(self._cache_dir),
        )
        tmp_path = Path(tmp_name)
        try:
            log.info("Downloading prebuilt model ONNX from %s (~one-time)", url)
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
                    "Prebuilt model download from %s looked truncated (%d bytes); ignoring.",
                    url,
                    written,
                )
                return None

            tmp_path.replace(onnx_path)
            log.info(
                "Model ONNX ready at %s (%.1f MB, prebuilt, torch-free)",
                onnx_path,
                written / (1 << 20),
            )
            return str(onnx_path)
        except Exception:  # noqa: BLE001 — network / disk / URL errors
            log.warning(
                "Prebuilt model download failed; will try ultralytics export.",
                exc_info=True,
            )
            return None
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:  # noqa: BLE001
                log.debug("Could not remove temp download %s", tmp_path, exc_info=True)

    def _download_and_export(self, model_pt: str, onnx_path: Path) -> str | None:
        """Download the ultralytics ``.pt`` and export it to *onnx_path*.

        Returns the ONNX path on success, ``None`` on any failure (logged).
        """
        if _model_export_disabled():
            self._last_error = (
                f"{Path(model_pt).stem}: model export disabled by "
                f"{_DISABLE_EXPORT_ENV}; use a cached/prebuilt ONNX."
            )
            log.info(
                "Skipping %s export because %s is set; cached/prebuilt models only.",
                model_pt,
                _DISABLE_EXPORT_ENV,
            )
            return None

        try:
            from ultralytics import YOLO  # type: ignore[attr-defined]  # noqa: PLC0415
        except Exception:  # noqa: BLE001 — ImportError or transitive import failure
            self._last_error = (
                f"{Path(model_pt).stem}: ultralytics not installed and no "
                "prebuilt ONNX. Install ultralytics, run "
                "`python -m tools.fetch_models`, or set AUTOPTZ_MODEL_URL."
            )
            log.warning(
                "ultralytics not installed; detector unavailable "
                "(live-preview-only). Install it or pre-fetch with "
                "`python -m tools.fetch_models`.",
            )
            return None

        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            log.warning("Could not create model cache dir %s", self._cache_dir, exc_info=True)
            return None

        # ultralytics downloads weights into the *current* working directory by
        # default and exports the ONNX next to the .pt.  Drive both from the
        # cache dir so nothing lands in the repo / cwd.
        prev_cwd = Path.cwd()
        try:
            os.chdir(self._cache_dir)
        except Exception:  # noqa: BLE001
            log.warning("Could not chdir to cache dir %s", self._cache_dir, exc_info=True)
            return None

        try:
            log.info(
                "Downloading + exporting detector %s → %s (first run, ~one-time)",
                model_pt,
                onnx_path.name,
            )
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
        except Exception as exc:  # noqa: BLE001 — network/export/runtime errors
            self._last_error = (
                f"{Path(model_pt).stem}: download/export failed "
                f"({type(exc).__name__}). Check the network or pre-fetch with "
                "`python -m tools.fetch_models`."
            )
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

        self._last_error = (
            f"{onnx_path.stem}: export reported success but the ONNX file is missing."
        )
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
