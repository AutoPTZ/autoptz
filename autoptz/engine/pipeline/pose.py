"""Optional single-person pose (keypoint) estimator for pose-stable framing.

Why
---
The PTZ aim point and auto-zoom were derived from the YOLO person **bbox**
centre + a height fraction.  When the tracked person raises/extends an arm the
bbox grows and its centre shifts, jerking the camera.  This module estimates the
person's **torso keypoints** (shoulders + hips) so the worker can aim at a stable
point (shoulder/torso midpoint) and zoom on a stable span (shoulder→hip),
ignoring arm/leg motion.  See :mod:`autoptz.engine.pipeline.framing` for the pure
aim/height math this feeds.

Design
------
- Backed by the same EP-agnostic ONNX Runtime factory the detector uses
  (:func:`autoptz.engine.runtime.inference.make_session`) with a YOLO-pose ONNX
  (default ``yolo11n-pose.onnx``).  Accepts the standard YOLO-pose output
  ``[1, 56, anchors]`` (4 box + 1 person conf + 17×3 keypoints) or the
  transposed ``[1, anchors, 56]``; both are auto-detected.
- Runs on **one person crop** (the active target track's bbox, padded), not the
  whole frame, so it stays cheap.  It is intended to be called only a few times a
  second for the single target — the worker reuses the last keypoints between
  detect intervals.
- **Graceful absence**: if the pose model file can't be resolved, ORT can't load
  it, or anything raises, the estimator reports ``available == False`` and
  :meth:`estimate` returns ``None``.  It never blocks startup and never crashes
  the worker — the worker then falls back to the existing bbox-based math.

Model file + path resolution
-----------------------------
Default filename ``yolo11n-pose.onnx``.  The path is resolved the same way the
detector's model is, via the platform app-data ``…/AutoPTZ/models`` dir (see
:class:`autoptz.engine.runtime.models.ModelManager`), plus an
``AUTOPTZ_POSE_MODEL_PATH`` env override.  No auto-download is wired (pose is
optional); drop ``yolo11n-pose.onnx`` into that models dir, or point the env var
at one, to enable pose-stable framing.

Dependencies: onnxruntime + numpy + cv2 (all already required by the detector).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autoptz.engine.pipeline.framing import Keypoint

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

log = logging.getLogger(__name__)

# Default pose model filename, resolved in the same models dir as the detector.
DEFAULT_POSE_MODEL = "yolo11n-pose.onnx"

# Env override for the pose model path (mirrors AUTOPTZ_MODEL_PATH for detection).
_ENV_POSE_PATH = "AUTOPTZ_POSE_MODEL_PATH"

# YOLO-pose emits COCO-17 keypoints → 17 * 3 (x, y, conf) = 51 values, preceded
# by 4 box coords + 1 person confidence = 56 channels.
_NUM_KEYPOINTS = 17
_POSE_CHANNELS = 5 + _NUM_KEYPOINTS * 3  # 56

# Pad the target bbox by this fraction of its size before cropping, so shoulders
# near the box edge aren't clipped out of the pose input.
_CROP_PAD = 0.15

# One-time debug guard so an absent/broken pose model logs a single line per
# process rather than once per worker.
_LOGGED_NO_POSE = False


def _log_no_pose_once(reason: str) -> None:
    global _LOGGED_NO_POSE
    if _LOGGED_NO_POSE:
        return
    _LOGGED_NO_POSE = True
    log.debug(
        "pose estimator unavailable (%s); pose-stable framing off — falling "
        "back to bbox aim. Place %s in the models dir or set %s to enable.",
        reason, DEFAULT_POSE_MODEL, _ENV_POSE_PATH,
    )


def _model_input_size(shape: Any, fallback: int) -> int:
    """Return the model's required square input size from its input ``shape``.

    The exported YOLO-pose ONNX has a static ``[1, 3, H, W]`` input (640×640);
    we must letterbox to exactly that or ORT rejects the run.  Read the trailing
    H/W of *shape* and use it when static; fall back to *fallback* only when the
    spatial axes are dynamic (symbolic strings / ``None`` / ``<= 0``).
    """
    try:
        dims = []
        for v in list(shape or [])[-2:]:  # trailing (H, W) of [N, C, H, W]
            try:
                iv = int(v)
            except (TypeError, ValueError):
                continue
            if iv > 0:
                dims.append(iv)
        if dims:
            return max(dims)
    except Exception:  # noqa: BLE001 — never let introspection break the build
        pass
    return int(fallback)


def resolve_pose_model_path() -> str | None:
    """Return a usable pose ONNX path, auto-provisioning it on first use.

    Delegates to :meth:`ModelManager.ensure_pose`, which resolves in order:

    1. ``AUTOPTZ_POSE_MODEL_PATH`` env override → returned verbatim if it points
       at an existing file.
    2. ``yolo11n-pose.onnx`` already cached in the detector's models dir
       (:attr:`ModelManager.cache_dir`).
    3. **Download + export** from ultralytics (one-time) — so enabling pose
       (overlay or pose-stable framing) works out of the box instead of silently
       doing nothing because the file was never placed manually.

    Never raises / never blocks the UI: it runs on the worker's lazy pose-build
    (inference thread), and returns ``None`` when ultralytics/network are
    unavailable (pose simply stays off).
    """
    try:
        from autoptz.engine.runtime.models import default_manager

        return default_manager().ensure_pose()
    except Exception:  # noqa: BLE001 — model provisioning must never break the worker
        log.debug("pose model path resolution failed", exc_info=True)
    return None


class PoseEstimator:
    """Single-person keypoint estimator backed by a YOLO-pose ONNX session.

    Construct it cheaply (resolves the model path + builds the session); if no
    model is available it stays in an ``available == False`` state and
    :meth:`estimate` always returns ``None``.  The worker should lazily build one
    only when tracking actually needs it.

    Args:
        model_path: Explicit ONNX path.  ``None`` → resolve via
                    :func:`resolve_pose_model_path`.
        input_size: Fallback square input resolution.  The exported YOLO-pose
                    ONNX has a **fixed** input (640×640); the real size is read
                    from the session in :meth:`__init__` and overrides this — the
                    arg only matters for a dynamic-axis model.  (Feeding the wrong
                    size makes ORT reject every call, which is exactly the bug
                    that silently disabled pose: a 256 crop into a 640 model.)
        conf_threshold: Minimum person confidence to accept a pose.
        prefs:      ORT EP preferences (forwarded to ``make_session``).
        _session:   Injected ORT session (unit-test / CI override).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        input_size: int = 640,
        conf_threshold: float = 0.30,
        prefs: Any | None = None,
        _session: Any | None = None,
    ) -> None:
        self._input_size = int(input_size)
        self._conf_threshold = float(conf_threshold)
        self._session: Any | None = None
        self._input_name: str = ""
        self._output_name: str = ""

        if _session is not None:
            self._session = _session
        else:
            self._try_build_session(model_path, prefs)

        if self._session is not None:
            try:
                inp = self._session.get_inputs()[0]
                out = self._session.get_outputs()[0]
                self._input_name = inp.name
                self._output_name = out.name
                # CRITICAL: letterbox to the size the MODEL expects, not a guess.
                # YOLO-pose is exported with a fixed 640×640 input; feeding 256
                # made ORT raise on every estimate() (caught → None), so pose
                # silently produced no keypoints (no skeleton, bbox-only aim).
                self._input_size = _model_input_size(inp.shape, self._input_size)
                log.info("PoseEstimator ready | ep=%s input=%s%s output=%s size=%d",
                         self.ep, inp.name, getattr(inp, "shape", None), out.name,
                         self._input_size)
            except Exception:  # noqa: BLE001
                log.debug("pose session introspection failed", exc_info=True)
                self._session = None

    # ── public API ──────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True iff a pose session loaded and is ready to infer."""
        return self._session is not None

    @property
    def ep(self) -> str:
        if self._session is None:
            return ""
        try:
            return self._session.get_providers()[0]
        except Exception:  # noqa: BLE001
            return ""

    def estimate(
        self,
        frame: NDArray[np.uint8],
        bbox: tuple[float, float, float, float],
    ) -> list[Keypoint] | None:
        """Estimate COCO-17 keypoints for the person in *bbox* of *frame*.

        *bbox* is ``(x1, y1, x2, y2)`` in original-frame pixels (the target
        track box).  Returns a list of 17 :class:`~...framing.Keypoint` in
        original-frame pixel coordinates, or ``None`` when pose is unavailable,
        the crop is degenerate, or the model returns nothing confident.  Never
        raises — any failure degrades to ``None`` so the worker keeps the
        bbox-based aim.
        """
        if self._session is None:
            return None
        try:
            import cv2  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            return None

        try:
            h, w = frame.shape[:2]
            if w <= 0 or h <= 0:
                return None

            # Pad + clamp the crop region so torso points near the edge survive.
            x1, y1, x2, y2 = bbox
            bw, bh = (x2 - x1), (y2 - y1)
            if bw <= 1.0 or bh <= 1.0:
                return None
            px, py = bw * _CROP_PAD, bh * _CROP_PAD
            cx1 = max(0, int(round(x1 - px)))
            cy1 = max(0, int(round(y1 - py)))
            cx2 = min(w, int(round(x2 + px)))
            cy2 = min(h, int(round(y2 + py)))
            if cx2 - cx1 < 2 or cy2 - cy1 < 2:
                return None
            crop = frame[cy1:cy2, cx1:cx2]

            inp, scale, pad = self._letterbox(crop, np, cv2)
            raw = self._session.run(
                [self._output_name], {self._input_name: inp},
            )[0]
            raw = np.asarray(raw, dtype=np.float32)

            kps = self._parse_pose(raw, np)
            if kps is None:
                return None

            # Map keypoints: letterboxed-crop space → crop space → frame space.
            ratio, pad_x, pad_y = scale, pad[0], pad[1]
            out: list[Keypoint] = []
            for kx, ky, kc in kps:
                fx = (kx - pad_x) / ratio + cx1
                fy = (ky - pad_y) / ratio + cy1
                out.append(Keypoint(x=float(fx), y=float(fy), conf=float(kc)))
            return out
        except Exception:  # noqa: BLE001 — pose must never break the worker
            log.debug("pose estimate failed", exc_info=True)
            return None

    # ── internals ─────────────────────────────────────────────────────────────

    def _try_build_session(
        self, model_path: str | Path | None, prefs: Any | None,
    ) -> None:
        """Resolve the model + build an ORT session; stay unavailable on failure."""
        path = str(model_path) if model_path is not None else resolve_pose_model_path()
        if not path or not Path(path).is_file():
            _log_no_pose_once("no model file")
            return
        try:
            from autoptz.engine.runtime.inference import make_session

            self._session = make_session(Path(path), prefs)
        except Exception:  # noqa: BLE001 — load failure → estimator stays disabled
            _log_no_pose_once("session load failed")
            log.debug("pose session load failed for %s", path, exc_info=True)
            self._session = None

    def _letterbox(
        self, crop: NDArray[np.uint8], np: Any, cv2: Any,
    ) -> tuple[NDArray[np.float32], float, tuple[int, int]]:
        """Letterbox *crop* to a square ``input_size``; return (chw_batch, ratio, pad)."""
        size = self._input_size
        h, w = crop.shape[:2]
        ratio = min(size / h, size / w)
        new_h = max(1, int(round(h * ratio)))
        new_w = max(1, int(round(w * ratio)))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_y = (size - new_h) // 2
        pad_x = (size - new_w) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
        img = canvas[:, :, ::-1].astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]
        return img, ratio, (pad_x, pad_y)

    def _parse_pose(
        self, raw: NDArray[np.float32], np: Any,
    ) -> list[tuple[float, float, float]] | None:
        """Parse a YOLO-pose output into the best person's 17 keypoints.

        Accepts ``[1, 56, anchors]`` (channels-first, the ultralytics default) or
        ``[1, anchors, 56]`` / ``[anchors, 56]``.  Picks the single most-confident
        anchor and returns its 17 ``(x, y, conf)`` in input-square pixel space, or
        ``None`` if no anchor clears the person-confidence threshold.
        """
        if raw.ndim == 3:
            raw = raw[0]
        if raw.ndim != 2:
            return None

        # Normalise to [anchors, channels].
        if raw.shape[0] == _POSE_CHANNELS and raw.shape[1] != _POSE_CHANNELS:
            preds = raw.T
        elif raw.shape[1] == _POSE_CHANNELS:
            preds = raw
        elif raw.shape[0] == _POSE_CHANNELS:
            preds = raw.T
        else:
            return None

        if preds.shape[0] == 0:
            return None

        conf = preds[:, 4]
        best = int(np.argmax(conf))
        if float(conf[best]) < self._conf_threshold:
            return None

        kp_flat = preds[best, 5:]
        if kp_flat.shape[0] < _NUM_KEYPOINTS * 3:
            return None
        kp = kp_flat[: _NUM_KEYPOINTS * 3].reshape(_NUM_KEYPOINTS, 3)
        return [(float(x), float(y), float(c)) for x, y, c in kp]
