"""OSNet body-appearance re-identification with hysteresis matching.

Purpose
-------
The *tracker* owns frame-to-frame continuity; **ReID owns recovery**.  When the
target track goes ``lost`` (occlusion, someone crosses in front), new tracks are
embedded with **OSNet** and matched against the target's appearance template so
we re-bind the *correct* track instead of locking onto an interloper.

- Model: OSNet (ONNX) via ``boxmot``'s ReID model zoo; weights auto-download on
  first use.  A 512-d appearance embedding per body crop.
- **Hysteresis** to avoid flicker: enter a lock at cosine > ``threshold_hi``,
  maintain it while > ``threshold_lo``.  A short EMA template tracks the target's
  current appearance so it adapts to lighting/pose drift.
- **Graceful.** If ``boxmot``/torch/weights are unavailable, ``available`` is
  ``False`` and every method is a safe no-op — tracking continues motion-only
  and the worker never hard-fails.

This module is appearance-only; track lifecycle stays in
``autoptz.engine.pipeline.track``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

_WARNED_UNAVAILABLE = False


def _normalize(vec: NDArray[np.floating]) -> NDArray[np.float32]:
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v if n <= 1e-8 else (v / n).astype(np.float32)


def _cosine(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    na, nb = _normalize(a), _normalize(b)
    if na.shape != nb.shape or na.size == 0:
        return 0.0
    return float(np.dot(na, nb))


def adaptive_threshold_hi(
    candidates: list[NDArray[np.floating]],
    base_hi: float,
    *,
    margin: float = 0.05,
    cap: float = 0.95,
) -> float:
    """Raise the recovery accept threshold toward the most-similar candidate pair.

    With <2 candidates there is no inter-person signal → return *base_hi*.
    Otherwise return ``clamp(max(base_hi, max_pairwise_cosine + margin), base_hi, cap)``
    so a scene full of look-alikes demands a higher cosine to re-bind, while a
    distinct scene keeps the base.  Never returns below *base_hi*.

    Degenerate (zero-norm) vectors are skipped; if fewer than 2 valid vectors
    remain after filtering, *base_hi* is returned.
    """
    # Filter to non-degenerate (non-zero) vectors only.
    valid: list[NDArray[np.float32]] = []
    for v in candidates:
        arr = np.asarray(v, dtype=np.float32).ravel()
        if arr.size > 0 and float(np.linalg.norm(arr)) > 1e-8:
            valid.append(arr)

    if len(valid) < 2:
        return base_hi

    # Find the maximum pairwise cosine similarity among all pairs.
    max_cos = 0.0
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            c = _cosine(valid[i], valid[j])
            if c > max_cos:
                max_cos = c

    raw = max(base_hi, max_cos + margin)
    return float(min(max(raw, base_hi), cap))


@dataclass
class _Template:
    """EMA appearance template for the locked target."""

    vec: NDArray[np.float32]
    ema: float = 0.6  # weight of the existing template when updating

    def update(self, new: NDArray[np.floating]) -> None:
        n = _normalize(new)
        if n.shape != self.vec.shape:
            self.vec = n
            return
        self.vec = _normalize(self.ema * self.vec + (1.0 - self.ema) * n)


@dataclass
class ReIDResult:
    """Outcome of a recovery attempt across candidate crops."""

    matched: bool = False
    best_score: float = 0.0
    best_index: int = -1
    scores: list[float] = field(default_factory=list)


def _pick_device(*, has_mps: bool, has_cuda: bool) -> str:
    """Device priority for OSNet ReID: Apple GPU (mps) → CUDA → CPU."""
    if has_mps:
        return "mps"
    if has_cuda:
        return "cuda"
    return "cpu"


def _best_reid_device() -> str:
    """Best available torch device for ReID, or "cpu" if torch is unavailable.

    Keeps the appearance pass off the CPU on Macs (Apple ``mps``) and CUDA boxes —
    it is otherwise a major per-frame cost that also stalls the inference thread
    through the GIL. Set ``AUTOPTZ_REID_DEVICE=cpu`` to force it back (e.g. if an
    OSNet op misbehaves on mps).
    """
    import os  # noqa: PLC0415

    forced = os.environ.get("AUTOPTZ_REID_DEVICE", "").strip().lower()
    if forced in ("cpu", "mps", "cuda"):
        return forced
    try:
        import torch  # noqa: PLC0415

        return _pick_device(
            has_mps=bool(torch.backends.mps.is_available() and torch.backends.mps.is_built()),
            has_cuda=bool(torch.cuda.is_available()),
        )
    except Exception:  # noqa: BLE001 — no torch / probe failure → CPU
        return "cpu"


class BodyReID:
    """OSNet appearance embeddings + hysteresis recovery matching.

    Args:
        weights:      Path to an OSNet ONNX/PT weight (boxmot auto-resolves a
                      default name like ``osnet_x0_25_msmt17.pt`` if None).
        device:       Torch/ORT device string.
        threshold_hi: Cosine to *enter* a lock (re-bind a recovered track).
        threshold_lo: Cosine to *maintain* an existing lock.
        _backend:     Injected embedder (tests/CI); must expose
                      ``get_features(xyxys, frame) -> ndarray[N, D]``.
    """

    def __init__(
        self,
        weights: Path | None = None,
        *,
        device: str | None = None,
        threshold_hi: float = 0.70,
        threshold_lo: float = 0.45,
        _backend: Any | None = None,
    ) -> None:
        self._threshold_hi = threshold_hi
        self._threshold_lo = threshold_lo
        self._backend = _backend
        self._available = _backend is not None
        # Injected test backends expose the legacy ``get_features`` contract.
        self._new_api = False
        self._template: _Template | None = None
        self._locked = False
        if _backend is None:
            self._try_init(weights, device)

    def _try_init(self, weights: Path | None, device: str | None) -> None:
        global _WARNED_UNAVAILABLE
        name = weights or Path("osnet_x0_25_msmt17.pt")
        device = device or _best_reid_device()
        # Try the resolved (GPU) device first; if it can't initialise (an op a
        # given backend doesn't support on mps, etc.) fall back to CPU before
        # finally degrading to motion-only.
        if self._init_on_device(name, device):
            return
        if device != "cpu" and self._init_on_device(name, "cpu"):
            log.info("BodyReID: %s device init failed; using CPU", device)
            return
        self._backend = None
        self._available = False
        if not _WARNED_UNAVAILABLE:
            _WARNED_UNAVAILABLE = True
            log.warning(
                "boxmot OSNet ReID unavailable; appearance recovery disabled "
                "(motion-only tracking still works).",
                exc_info=True,
            )

    def _init_on_device(self, name: Path, device: str) -> bool:
        """Build the OSNet backend on *device*; True on success, False to fall back.

        BoxMOT ≥ 11 exposes ``boxmot.reid.ReID`` (``__call__(frame, boxes=...)``);
        older releases used ``boxmot.appearance.reid_auto_backend.ReidAutoBackend``
        whose ``.model`` had ``get_features(xyxys, frame)``. Try the new API first,
        then the legacy one, so both major versions work.
        """
        try:
            from boxmot.reid import ReID  # noqa: PLC0415

            self._backend = ReID(weights=name, device=device, half=False)
            self._new_api = True
            self._available = True
            log.info("BodyReID ready (OSNet %s on %s, boxmot.reid API)", name, device)
            return True
        except Exception:  # noqa: BLE001 — fall through to the legacy API
            pass
        try:
            from boxmot.appearance.reid_auto_backend import (  # noqa: PLC0415
                ReidAutoBackend,
            )

            self._backend = ReidAutoBackend(weights=name, device=device, half=False).model
            self._new_api = False
            self._available = True
            log.info("BodyReID ready (OSNet %s on %s, legacy API)", name, device)
            return True
        except Exception:  # noqa: BLE001 — missing dep/weights/network must not raise
            return False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def locked(self) -> bool:
        return self._locked

    # ── embedding ────────────────────────────────────────────────────────────────

    def embed(
        self,
        boxes_xyxy: list[tuple[float, float, float, float]],
        frame: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        """Return ``[N, D]`` normalised appearance embeddings for *boxes_xyxy*.

        Empty ``[0, D]`` when disabled or on any error (never raises).
        """
        if not self._available or self._backend is None or not boxes_xyxy:
            return np.empty((0, 0), dtype=np.float32)
        try:
            xyxys = np.asarray(boxes_xyxy, dtype=np.float32)
            if self._new_api:
                # boxmot.reid.ReID.__call__(frame, boxes=xyxys) -> [N, D]
                feats = self._backend(frame, boxes=xyxys)
            else:
                feats = self._backend.get_features(xyxys, frame)
            feats = np.atleast_2d(np.asarray(feats, dtype=np.float32))
            return np.stack([_normalize(row) for row in feats])
        except Exception:  # noqa: BLE001
            log.debug("BodyReID.embed failed", exc_info=True)
            return np.empty((0, 0), dtype=np.float32)

    # ── target template + recovery ───────────────────────────────────────────────

    def set_target(self, embedding: NDArray[np.floating]) -> None:
        """Seed/refresh the locked target's appearance template."""
        self._template = _Template(vec=_normalize(embedding))
        self._locked = True

    def reset(self) -> None:
        """Drop the lock (target cleared / camera reset)."""
        self._template = None
        self._locked = False

    def update_target(self, embedding: NDArray[np.floating]) -> None:
        """Blend a fresh observation of the target into the EMA template."""
        if self._template is None:
            self.set_target(embedding)
        else:
            self._template.update(embedding)

    def recover(
        self,
        candidates: NDArray[np.float32],
        *,
        threshold: float | None = None,
        update: bool = True,
    ) -> ReIDResult:
        """Pick the candidate body crop that best matches the target template.

        Hysteresis: a *new* lock requires ``score > threshold_hi``; maintaining
        an existing lock only requires ``score > threshold_lo`` (set by whether
        :attr:`locked` is currently True).  Callers can pass *threshold* to force
        a stricter recovery gate and *update=False* to score candidates without
        blending ambiguous evidence into the template.  Returns the best
        index/score and whether it cleared the active threshold.
        """
        if self._template is None or candidates.size == 0:
            return ReIDResult()
        scores = [_cosine(self._template.vec, c) for c in np.atleast_2d(candidates)]
        best_i = int(np.argmax(scores))
        best = float(scores[best_i])
        thr = (
            float(threshold)
            if threshold is not None
            else (self._threshold_lo if self._locked else self._threshold_hi)
        )
        matched = best >= thr
        if matched and update:
            self._template.update(candidates[best_i])
            self._locked = True
        return ReIDResult(
            matched=matched,
            best_score=best,
            best_index=best_i,
            scores=scores,
        )
