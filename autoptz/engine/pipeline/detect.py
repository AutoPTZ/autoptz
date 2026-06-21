"""YOLO26 person detector via ONNX Runtime (EP-agnostic).

Design
------
- Accepts any YOLO-family ONNX that produces either:
    • **NMS-free** output ``[batch, N, 5|6]``  (YOLOv10/26 style)
    • **Pre-NMS** output ``[batch, 4+C, anchors]``  (YOLOv8 style → built-in NMS)
  The output format is auto-detected from the array shape.

- Letterboxes to a square ``input_size`` (640 default, 480 on CPU tiers),
  then maps bbox predictions back to the original frame pixel coordinates.

- ``detect_interval`` (default 1 = every frame) skips ORT inference on
  intervening frames and returns ``[]``, letting the tracker's Kalman
  predictor interpolate.  A low-confidence floor is kept so BoT-SORT's
  second-association pass can still use weak boxes during occlusion.

- Exposes ``ep`` (active ORT Execution Provider name) for UI/telemetry.

Dependencies: onnxruntime (always), numpy, opencv-contrib-python.
Optional: onnx (only needed for the test-model helper).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from autoptz.engine.runtime.inference import HardwarePrefs, make_session

log = logging.getLogger(__name__)

# ── Shared BBox / Detection types (re-used by track.py) ──────────────────────


@dataclass(frozen=True)
class BBox:
    """Pixel-space bounding box in xyxy format."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) * 0.5

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) * 0.5

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def iou(self, other: BBox) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class Detection:
    """One person detection from the model.

    ``keypoints`` is populated only by the unified pose detector (one backbone
    emitting boxes *and* COCO-17 keypoints per person); it's ``None`` for the
    plain detector.  Stored as raw ``(x, y, conf)`` triples in original-frame
    pixels to avoid importing the framing ``Keypoint`` type here (the worker
    converts when feeding the pose-stable aim).
    """

    bbox: BBox
    conf: float
    class_id: int = 0  # 0 = person in COCO
    keypoints: tuple[tuple[float, float, float], ...] | None = None


# ── Letterbox helpers ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _ScaleInfo:
    ratio: float  # resize ratio (same for both axes)
    pad_x: int  # left padding in the padded image (pixels)
    pad_y: int  # top padding
    orig_h: int
    orig_w: int


def _letterbox(frame: NDArray[np.uint8], size: int) -> tuple[NDArray[np.float32], _ScaleInfo]:
    """Letterbox *frame* into a *size×size* square, return (chw_batch, ScaleInfo)."""
    h, w = frame.shape[:2]
    ratio = min(size / h, size / w)
    new_h = int(round(h * ratio))
    new_w = int(round(w * ratio))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_y = (size - new_h) // 2
    pad_x = (size - new_w) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    # BGR → RGB, HWC → CHW, normalise, add batch
    img = canvas[:, :, ::-1].astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]

    return img, _ScaleInfo(ratio=ratio, pad_x=pad_x, pad_y=pad_y, orig_h=h, orig_w=w)


def _to_orig_coords(x1: float, y1: float, x2: float, y2: float, si: _ScaleInfo) -> BBox:
    """Map bbox coordinates from the padded input space back to the original frame."""
    x1 = (x1 - si.pad_x) / si.ratio
    y1 = (y1 - si.pad_y) / si.ratio
    x2 = (x2 - si.pad_x) / si.ratio
    y2 = (y2 - si.pad_y) / si.ratio
    # Clamp to original frame bounds
    x1 = max(0.0, min(float(si.orig_w), x1))
    y1 = max(0.0, min(float(si.orig_h), y1))
    x2 = max(0.0, min(float(si.orig_w), x2))
    y2 = max(0.0, min(float(si.orig_h), y2))
    return BBox(x1, y1, x2, y2)


# ── Output format normalisation ───────────────────────────────────────────────


def _nms(
    boxes: NDArray[np.float32], scores: NDArray[np.float32], iou_thr: float = 0.45
) -> list[int]:
    """Vectorised greedy NMS.  Returns kept indices sorted by descending score."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def _parse_raw_output(
    raw: NDArray[np.float32],
    input_size: int,
    conf_floor: float,
) -> NDArray[np.float32]:
    """Normalise diverse YOLO output shapes to ``[N, 6]`` (x1,y1,x2,y2,conf,cls).

    Handles:
    - ``[1, N, 5|6]`` — NMS-free (YOLOv10/26): already post-processed
    - ``[1, 4+C, anchors]`` — Pre-NMS (YOLOv8): needs transpose + NMS

    Coordinates may be normalised (0–1) or pixel-space; both are detected
    and converted to pixel space relative to the padded input.
    """
    if raw.ndim == 3:
        raw = raw[0]  # drop batch

    # ── NMS-free format: [N, 5|6] ─────────────────────────────────────────────
    if raw.ndim == 2 and raw.shape[1] in (5, 6):
        if raw.shape[1] == 5:
            cls_col = np.zeros((raw.shape[0], 1), dtype=np.float32)
            raw = np.concatenate([raw, cls_col], axis=1)
        # Normalised coords → pixel space
        if raw.shape[0] > 0 and raw[:, :4].max() <= 1.0 + 1e-3:
            raw = raw.copy()
            raw[:, :4] *= input_size
        return raw[raw[:, 4] >= conf_floor]

    # ── Pre-NMS YOLOv8 format: [4+C, anchors] ─────────────────────────────────
    if raw.ndim == 2 and raw.shape[0] > raw.shape[1]:
        # Already [anchors, 4+C]
        pass
    else:
        raw = raw.T  # [4+C, anchors] → [anchors, 4+C]

    boxes_cxcy = raw[:, :4]  # cx, cy, w, h
    scores = raw[:, 4:]
    conf = scores.max(axis=1)
    cls = scores.argmax(axis=1).astype(np.float32)

    mask = conf >= conf_floor
    boxes_cxcy = boxes_cxcy[mask]
    conf = conf[mask]
    cls = cls[mask]

    if len(boxes_cxcy) == 0:
        return np.empty((0, 6), dtype=np.float32)

    # Convert cx,cy,w,h → x1,y1,x2,y2
    half_w, half_h = boxes_cxcy[:, 2] / 2, boxes_cxcy[:, 3] / 2
    x1 = boxes_cxcy[:, 0] - half_w
    y1 = boxes_cxcy[:, 1] - half_h
    x2 = boxes_cxcy[:, 0] + half_w
    y2 = boxes_cxcy[:, 1] + half_h

    # Normalised coords → pixel space
    if x2.max() <= 1.0 + 1e-3:
        x1, y1, x2, y2 = x1 * input_size, y1 * input_size, x2 * input_size, y2 * input_size

    # Per-class NMS
    result_rows: list[NDArray[np.float32]] = []
    unique_cls = np.unique(cls).astype(int)
    for c in unique_cls:
        c_mask = cls == c
        c_boxes = np.stack([x1[c_mask], y1[c_mask], x2[c_mask], y2[c_mask]], axis=1)
        c_conf = conf[c_mask]
        keep = _nms(c_boxes, c_conf)
        for k in keep:
            result_rows.append(
                np.array(
                    [
                        c_boxes[k, 0],
                        c_boxes[k, 1],
                        c_boxes[k, 2],
                        c_boxes[k, 3],
                        float(c_conf[k]),
                        float(c),
                    ],
                    dtype=np.float32,
                )
            )

    if not result_rows:
        return np.empty((0, 6), dtype=np.float32)
    return np.stack(result_rows)


# ── Detector ──────────────────────────────────────────────────────────────────


class PersonDetector:
    """YOLO26 person detector backed by an ONNX Runtime session.

    Args:
        model_path:       Path to the ONNX (or CoreML/MLPACKAGE) model file.
        input_size:       Square input resolution (640 default, 480 on CPU tier).
        conf_threshold:   Hard confidence floor for returned detections.
        detect_interval:  Run inference every N frames; return ``[]`` in between.
        person_class_id:  Class index to keep (0 = COCO person).
        prefs:            ORT EP preferences (forwarded to ``make_session()``).
        _session:         Injected ORT session (unit-test / CI override).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        input_size: int = 640,
        conf_threshold: float = 0.25,
        detect_interval: int = 1,
        person_class_id: int = 0,
        prefs: HardwarePrefs | None = None,
        _session: ort.InferenceSession | None = None,
    ) -> None:
        self._input_size = input_size
        self._conf_threshold = conf_threshold
        self._conf_floor = max(0.1, conf_threshold - 0.15)  # keep low-conf for 2nd assoc
        self._detect_interval = max(1, detect_interval)
        self._person_class_id = person_class_id

        if _session is not None:
            self._session = _session
        elif model_path is not None:
            self._session = make_session(Path(model_path), prefs)
        else:
            raise ValueError("Either model_path or _session must be provided.")

        inp = self._session.get_inputs()[0]
        out = self._session.get_outputs()[0]
        self._input_name: str = inp.name
        self._output_name: str = out.name

        self._frame_count = 0  # incremented before check; frame 1 always detects

        log.info(
            "PersonDetector ready | ep=%s input=%s→%s output=%s",
            self.ep,
            inp.name,
            inp.shape,
            out.name,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def ep(self) -> str:
        return str(self._session.get_providers()[0])

    @property
    def precision(self) -> str:
        """Effective inference precision ("fp16"/"fp32") for the active EP."""
        from autoptz.engine.runtime.inference import effective_precision

        return effective_precision(self.ep)

    def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        """Run detection on *frame* (BGR H×W×3).

        Returns ``[]`` on non-inference frames (``detect_interval > 1``).
        The caller should pass empty detections to the tracker; its Kalman
        predictor will interpolate.
        """
        self._frame_count += 1
        # (frame_count - 1) % interval == 0 → detect on frames 1, N+1, 2N+1, …
        if (self._frame_count - 1) % self._detect_interval != 0:
            return []

        inp_tensor, scale_info = _letterbox(frame, self._input_size)

        raw_outputs = self._session.run([self._output_name], {self._input_name: inp_tensor})
        raw: NDArray[np.float32] = raw_outputs[0].astype(np.float32)

        dets_np = _parse_raw_output(raw, self._input_size, self._conf_floor)

        # Filter to person class and apply hard confidence threshold
        results: list[Detection] = []
        for row in dets_np:
            x1, y1, x2, y2, conf, cls = (
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
            )
            if int(round(cls)) != self._person_class_id:
                continue
            if conf < self._conf_threshold:
                continue
            bbox = _to_orig_coords(x1, y1, x2, y2, scale_info)
            if bbox.area() < 1.0:
                continue
            results.append(Detection(bbox=bbox, conf=conf, class_id=int(round(cls))))

        log.debug(
            "detect() frame=%d detections=%d ep=%s",
            self._frame_count,
            len(results),
            self.ep,
        )
        return results

    def reset(self) -> None:
        """Reset the frame counter (e.g. on source change)."""
        self._frame_count = 0


# ── Model utilities ───────────────────────────────────────────────────────────


def detections_to_numpy(dets: list[Detection]) -> NDArray[np.float32]:
    """Convert detections to ``[N, 6]`` float32 array expected by BoxMOT trackers.

    Columns: x1, y1, x2, y2, conf, class_id.
    Returns an empty ``[0, 6]`` array if *dets* is empty.
    """
    if not dets:
        return np.empty((0, 6), dtype=np.float32)
    rows = [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2, d.conf, float(d.class_id)] for d in dets]
    return np.array(rows, dtype=np.float32)


def make_synthetic_detector_session(
    input_size: int = 640,
    *,
    detections: list[tuple[float, float, float, float, float]] | None = None,
) -> ort.InferenceSession:
    """Build a minimal ONNX session that returns hardcoded detections.

    Useful in unit tests and CI where no real model file is available.
    Each entry in *detections* is ``(x1, y1, x2, y2, conf)`` in pixel
    coords of the padded *input_size* image; class_id is always 0 (person).

    The session input name is ``"images"`` (``[1, 3, H, W]``),
    output name is ``"output0"`` (``[1, N, 6]``).
    """
    try:
        import onnx  # noqa: PLC0415
        from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("onnx package required for synthetic model; pip install onnx") from exc

    if detections is None:
        detections = [
            (100.0, 150.0, 200.0, 400.0, 0.90),
            (300.0, 100.0, 450.0, 420.0, 0.82),
        ]

    n = len(detections)
    # Build [1, N, 6] array: x1,y1,x2,y2,conf,cls(=0)
    data = np.zeros((1, n, 6), dtype=np.float32)
    for i, (x1, y1, x2, y2, conf) in enumerate(detections):
        data[0, i] = [x1, y1, x2, y2, conf, 0.0]

    const_tensor = numpy_helper.from_array(data, name="output0_const")
    const_node = helper.make_node("Constant", inputs=[], outputs=["output0"], value=const_tensor)

    # Graph: unused 'images' input + Constant → output
    images_in = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, input_size, input_size]
    )
    output_out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, n, 6])
    graph = helper.make_graph([const_node], "synthetic_yolo26", [images_in], [output_out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8

    import io  # noqa: PLC0415

    buf = io.BytesIO()
    onnx.save(model, buf)
    buf.seek(0)

    return ort.InferenceSession(buf.read(), providers=["CPUExecutionProvider"])
