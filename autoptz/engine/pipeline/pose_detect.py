"""Unified person detector + pose head — one backbone, two heads.

Why
---
The pipeline historically ran two separate full backbones over the same pixels:
a detection YOLO (boxes) and, on a cropped target a few Hz, a YOLO-pose model
(keypoints).  A YOLO11-**pose** model already emits, per person, the box *and*
the 17 COCO keypoints in a single forward pass — the "one backbone, many heads"
shape (cf. how an autonomy stack runs one trunk feeding many task heads).  This
detector runs that single pass on the **full frame**, so every tracked person
gets keypoints every frame for free, and the separate per-crop pose pass goes
away (see the worker's unified-pose path).

It deliberately mirrors :class:`autoptz.engine.pipeline.detect.PersonDetector`'s
public surface (``detect()`` → ``list[Detection]``, ``ep``, ``reset()``,
``detect_interval``) so it drops into the same tracker path; the only addition is
that each returned :class:`Detection` carries ``keypoints``.

Output handling reuses the detector's letterbox / NMS / coord-mapping helpers and
the pose channel layout from :mod:`autoptz.engine.pipeline.pose` (4 box + 1 conf
+ 17×3 keypoints = 56 channels), accepting both ``[1, 56, anchors]`` and the
transposed layout.

Graceful absence: if the pose model can't be resolved/loaded the constructor
raises (the caller falls back to the plain detector + optional pose-crop path),
exactly like the detector does when its model is missing.

Dependencies: onnxruntime + numpy + cv2 (already required by the detector).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from autoptz.engine.pipeline.detect import (
    Detection,
    _letterbox,
    _nms,
    _to_orig_coords,
)
from autoptz.engine.runtime.inference import HardwarePrefs, make_session

if TYPE_CHECKING:
    import onnxruntime as ort

log = logging.getLogger(__name__)

# COCO-17 keypoints → 17×3 (x, y, conf) = 51, preceded by 4 box + 1 conf = 56.
_NUM_KEYPOINTS = 17
_POSE_CHANNELS = 5 + _NUM_KEYPOINTS * 3  # 56


class PoseDetector:
    """Full-frame YOLO11-pose detector returning boxes **and** keypoints.

    Args:
        model_path:      Path to a YOLO-pose ONNX.  ``None`` → resolve via
                         :func:`autoptz.engine.pipeline.pose.resolve_pose_model_path`.
        input_size:      Fallback square input; the real size is read from the
                         session (the export is fixed 640×640).
        conf_threshold:  Hard person-confidence floor for returned detections.
        detect_interval: Run inference every N frames; ``[]`` in between (the
                         tracker's Kalman predictor interpolates), matching
                         :class:`PersonDetector`.
        prefs:           ORT EP preferences (forwarded to ``make_session``).
        _session:        Injected ORT session (unit-test / CI override).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        input_size: int = 640,
        conf_threshold: float = 0.25,
        detect_interval: int = 1,
        prefs: HardwarePrefs | None = None,
        allow_download: bool = True,
        _session: ort.InferenceSession | None = None,
    ) -> None:
        self._input_size = int(input_size)
        self._conf_threshold = float(conf_threshold)
        self._conf_floor = max(0.1, conf_threshold - 0.15)
        self._detect_interval = max(1, int(detect_interval))

        if _session is not None:
            self._session = _session
        else:
            path = (
                str(model_path)
                if model_path is not None
                else _resolve_pose_model(allow_download=allow_download)
            )
            if not path or not Path(path).is_file():
                raise ValueError("PoseDetector: no usable YOLO-pose model file.")
            self._session = make_session(Path(path), prefs)

        inp = self._session.get_inputs()[0]
        out = self._session.get_outputs()[0]
        self._input_name: str = inp.name
        self._output_name: str = out.name
        self._input_size = _model_input_size(getattr(inp, "shape", None), self._input_size)
        self._frame_count = 0

        log.info(
            "PoseDetector ready | ep=%s input=%s%s output=%s size=%d",
            self.ep,
            inp.name,
            getattr(inp, "shape", None),
            out.name,
            self._input_size,
        )

    # ── public API (mirrors PersonDetector) ─────────────────────────────────────

    @property
    def ep(self) -> str:
        try:
            return str(self._session.get_providers()[0])
        except Exception:  # noqa: BLE001
            return ""

    @property
    def precision(self) -> str:
        from autoptz.engine.runtime.inference import effective_precision

        return effective_precision(self.ep)

    def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        """Run the unified pass on *frame* (BGR H×W×3).

        Returns ``[]`` on skipped frames (``detect_interval > 1``).  Each
        :class:`Detection` carries its COCO-17 ``keypoints`` (original-frame
        pixels).  Never raises into the worker — failures degrade to ``[]``.
        """
        self._frame_count += 1
        if (self._frame_count - 1) % self._detect_interval != 0:
            return []

        try:
            inp_tensor, scale_info = _letterbox(frame, self._input_size)
            raw = self._session.run([self._output_name], {self._input_name: inp_tensor})[0]
            raw = np.asarray(raw, dtype=np.float32)
            return self._parse(raw, scale_info)
        except Exception:  # noqa: BLE001 — detection must never crash the worker
            log.debug("pose-detect run failed", exc_info=True)
            return []

    def reset(self) -> None:
        self._frame_count = 0

    # ── internals ───────────────────────────────────────────────────────────────

    def _parse(self, raw: NDArray[np.float32], scale_info: Any) -> list[Detection]:
        """Decode YOLO-pose output → person ``Detection``s with keypoints."""
        preds = _to_anchors_major(raw)
        if preds is None or preds.shape[0] == 0:
            return []

        conf = preds[:, 4]
        mask = conf >= self._conf_floor
        preds = preds[mask]
        if preds.shape[0] == 0:
            return []
        conf = preds[:, 4]

        # box cx,cy,w,h → x1,y1,x2,y2 (input-square pixels)
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        kpts = preds[:, 5 : 5 + _NUM_KEYPOINTS * 3].reshape(-1, _NUM_KEYPOINTS, 3).copy()

        # Normalised coords → input-square pixels (boxes + keypoint x/y together).
        if x2.size and float(x2.max()) <= 1.0 + 1e-3:
            x1, y1, x2, y2 = (v * self._input_size for v in (x1, y1, x2, y2))
            kpts[:, :, 0] *= self._input_size
            kpts[:, :, 1] *= self._input_size

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = _nms(boxes, conf)

        results: list[Detection] = []
        for k in keep:
            if float(conf[k]) < self._conf_threshold:
                continue
            bbox = _to_orig_coords(
                float(x1[k]), float(y1[k]), float(x2[k]), float(y2[k]), scale_info
            )
            if bbox.area() < 1.0:
                continue
            kp_orig = _map_keypoints(kpts[k], scale_info)
            results.append(Detection(bbox=bbox, conf=float(conf[k]), class_id=0, keypoints=kp_orig))
        return results


class UnifiedPoseAdapter:
    """Expose the unified detector's keypoints through the ``PoseEstimator`` API.

    The worker's pose-stable aim calls ``estimate(frame, bbox)``; in unified mode
    the keypoints were already produced by the single detection pass, so this
    adapter returns the keypoints of the most-overlapping detection instead of
    running a second forward pass.  Drop-in for :class:`PoseEstimator` (same
    ``available`` / ``ep`` / ``estimate`` surface), so ``_pose_aim`` is unchanged.

    Args:
        source: zero-arg callable returning the current frame's detections
                (the worker passes ``lambda: self._last_detections``).
        ep:     EP label of the underlying unified detector (for telemetry).
    """

    _MIN_IOU = 0.3

    def __init__(self, source: Any, ep: str = "") -> None:
        self._source = source
        self._ep = ep

    @property
    def available(self) -> bool:
        return True

    @property
    def ep(self) -> str:
        return self._ep

    def estimate(
        self, frame: NDArray[np.uint8], bbox: tuple[float, float, float, float]
    ) -> list[Any] | None:
        del frame  # keypoints already computed on the detection pass
        from autoptz.engine.pipeline.detect import BBox
        from autoptz.engine.pipeline.framing import Keypoint

        try:
            dets = self._source() or []
        except Exception:  # noqa: BLE001
            return None
        target = BBox(*bbox)
        best = None
        best_iou = 0.0
        for d in dets:
            if getattr(d, "keypoints", None) is None:
                continue
            iou = target.iou(d.bbox)
            if iou > best_iou:
                best_iou = iou
                best = d
        if best is None or best_iou < self._MIN_IOU:
            return None
        return [Keypoint(x=x, y=y, conf=c) for (x, y, c) in best.keypoints]


def _to_anchors_major(raw: NDArray[np.float32]) -> NDArray[np.float32] | None:
    """Normalise raw output to ``[anchors, 56]`` regardless of layout."""
    if raw.ndim == 3:
        raw = raw[0]
    if raw.ndim != 2:
        return None
    if raw.shape[1] == _POSE_CHANNELS:
        return raw
    if raw.shape[0] == _POSE_CHANNELS:
        return raw.T
    return None


def _map_keypoints(
    kp: NDArray[np.float32], scale_info: Any
) -> tuple[tuple[float, float, float], ...]:
    """Map ``[17, 3]`` keypoints from letterboxed-input space to frame pixels."""
    ratio = scale_info.ratio
    pad_x, pad_y = scale_info.pad_x, scale_info.pad_y
    ow, oh = float(scale_info.orig_w), float(scale_info.orig_h)
    out: list[tuple[float, float, float]] = []
    for kx, ky, kc in kp:
        fx = (float(kx) - pad_x) / ratio
        fy = (float(ky) - pad_y) / ratio
        fx = max(0.0, min(ow, fx))
        fy = max(0.0, min(oh, fy))
        out.append((fx, fy, float(kc)))
    return tuple(out)


def _model_input_size(shape: Any, fallback: int) -> int:
    """Read the static square input size from the session input shape."""
    try:
        dims = []
        for v in list(shape or [])[-2:]:
            try:
                iv = int(v)
            except (TypeError, ValueError):
                continue
            if iv > 0:
                dims.append(iv)
        if dims:
            return max(dims)
    except Exception:  # noqa: BLE001
        pass
    return int(fallback)


def _resolve_pose_model(*, allow_download: bool = True) -> str | None:
    try:
        from autoptz.engine.pipeline.pose import resolve_pose_model_path

        return resolve_pose_model_path(allow_download=allow_download)
    except Exception:  # noqa: BLE001
        log.debug("pose-detect model resolution failed", exc_info=True)
        return None
