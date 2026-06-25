"""InsightFace SCRFD + ArcFace: detect faces, embed, and bind track → identity.

Design
------
- Wraps ``insightface.app.FaceAnalysis("buffalo_l")`` — an SCRFD face detector
  plus a 512-d ArcFace recogniser.  Models **auto-download** to
  ``~/.insightface/models`` on first ``prepare()``; we force the CPU provider so
  there is no GPU dependency (the EP override is best-effort).

- **Graceful degradation.** If ``insightface`` is not importable, the model
  cannot be downloaded (no network), or ``prepare()`` fails, the recogniser logs
  **once** and disables itself.  ``available`` is then ``False`` and every method
  is a safe no-op returning empty results — manual click-to-track keeps working
  and the engine never hard-fails.

- **Matching.** Face embeddings are L2-normalised; identity gallery match is a
  cosine similarity (== dot product on normalised vectors) against each
  enabled identity's stored ArcFace templates, keeping the best above a
  threshold.

This module owns *embedding + matching*; gallery storage/CRUD lives in
``autoptz.engine.identity.service``.  The camera worker drives both.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from autoptz.engine.identity.service import IdentityService

log = logging.getLogger(__name__)

# Embedding dimensionality for buffalo_l ArcFace.
ARCFACE_DIM = 512

# Default cosine-similarity floor for a confident identity match.
DEFAULT_MATCH_THRESHOLD = 0.35

# Logged-once guard so a missing dependency is reported a single time.
_WARNED_UNAVAILABLE = False


# ── data types ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FaceObservation:
    """One detected face in a frame.

    ``bbox`` is pixel-space ``(x1, y1, x2, y2)`` in the original frame.
    ``embedding`` is the L2-normalised 512-d ArcFace vector (float32).
    ``kps`` are SCRFD's 5 facial landmarks (left-eye, right-eye, nose,
    left-mouth, right-mouth) as ``(x, y)`` pixel pairs, or ``None`` if the
    detector did not provide them — used to estimate frontal-ness (yaw) so the
    worker can reject profile faces before auto-harvesting an identity.
    """

    bbox: tuple[float, float, float, float]
    embedding: NDArray[np.float32]
    det_score: float = 0.0
    kps: tuple[tuple[float, float], ...] | None = None

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) * 0.5

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) * 0.5

    def yaw_ratio(self) -> float | None:
        """Return a frontal-ness measure from the 5-point landmarks, or ``None``.

        Computes the nose's horizontal offset from the eye-midpoint as a fraction
        of the inter-ocular distance: ~0.0 for a perfectly frontal face, growing
        toward / past ~1.0 as the head turns to a profile.  ``None`` when
        landmarks are unavailable or the eyes are degenerate.
        """
        kps = self.kps
        if not kps or len(kps) < 3:
            return None
        (lx, ly), (rx, ry), (nx, _ny) = kps[0], kps[1], kps[2]
        eye_mid_x = (lx + rx) * 0.5
        inter_ocular = ((rx - lx) ** 2 + (ry - ly) ** 2) ** 0.5
        if inter_ocular <= 1e-3:
            return None
        return float(abs(nx - eye_mid_x) / inter_ocular)


@dataclass(frozen=True)
class IdentityMatch:
    """A gallery match for a face embedding."""

    identity_id: str
    name: str
    score: float  # cosine similarity in [-1, 1]


# ── helpers ─────────────────────────────────────────────────────────────────────


def normalize(vec: NDArray[np.floating]) -> NDArray[np.float32]:
    """Return *vec* L2-normalised as float32 (zero-safe)."""
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    if n <= 1e-8:
        return v
    return (v / n).astype(np.float32)


def embedding_to_bytes(vec: NDArray[np.floating]) -> bytes:
    """Serialise a (normalised) embedding to compact float32 bytes for SQLite."""
    return normalize(vec).tobytes()


def embedding_from_bytes(blob: bytes) -> NDArray[np.float32]:
    """Deserialise a float32 embedding blob (re-normalised defensively)."""
    arr = np.frombuffer(blob, dtype=np.float32)
    return normalize(arr)


def cosine(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Cosine similarity of two vectors (normalises both)."""
    na, nb = normalize(a), normalize(b)
    if na.shape != nb.shape or na.size == 0:
        return 0.0
    return float(np.dot(na, nb))


# ── face model provisioning ────────────────────────────────────────────────────


def _insightface_pack_present(root: Path) -> bool:
    """True if *root* holds an insightface model pack (``root/models/*/*.onnx``)."""
    models = root / "models"
    try:
        return models.is_dir() and any(models.glob("*/*.onnx"))
    except OSError:
        return False


def insightface_root() -> str:
    """Resolve the insightface model storage root.

    Priority: ``$INSIGHTFACE_HOME`` (explicit override) → a pack **bundled inside
    the app** (``<bundled models>/insightface`` — what release installers ship and
    ``tools.fetch_models`` writes) → the **user model cache**
    (``<app-data models>/insightface``, populated by a dev's ``fetch_models``) →
    ``~/.insightface`` (insightface's own default, where it auto-downloads when
    online).  Threaded into ``FaceAnalysis(root=...)`` so an **offline packaged
    app finds its weights with no download** — the real fix for "faces never save
    on Windows".  Kept consistent with :func:`face_status` in
    :mod:`autoptz.engine.runtime.diagnostics`.
    """
    env = os.environ.get("INSIGHTFACE_HOME")
    if env:
        return env
    try:
        from autoptz.engine.runtime.models import (  # noqa: PLC0415
            _models_cache_dir,
            bundled_models_dir,
        )

        for base in (bundled_models_dir(), _models_cache_dir()):
            candidate = Path(base) / "insightface"
            if _insightface_pack_present(candidate):
                return str(candidate)
    except Exception:  # noqa: BLE001 — resolution must never raise into the model load
        pass
    return str(Path.home() / ".insightface")


def ensure_face_model(root: str | None = None, model_name: str = "buffalo_l") -> str | None:
    """Pre-download the insightface model pack into *root* for offline use.

    Constructing :class:`FaceAnalysis` triggers insightface's own download of the
    pack (needs network) into ``<root>/models/<model_name>``; once cached, later
    OFFLINE runs load it without network — the fix for "faces never save on an
    offline first-run" (run ``python -m tools.fetch_models`` once while online).
    Returns ``None`` on success, or a human-readable error string (never raises).
    """
    try:
        from insightface.app import FaceAnalysis  # noqa: PLC0415

        from autoptz.engine.runtime.inference import EP  # noqa: PLC0415

        app = FaceAnalysis(
            name=model_name,
            root=root or insightface_root(),
            providers=[EP.CPU.value],
            allowed_modules=["detection", "recognition"],
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))
        return None
    except Exception as exc:  # noqa: BLE001 — provisioning must never crash the caller
        return f"{type(exc).__name__}: {exc}"


# ── face recogniser ──────────────────────────────────────────────────────────────


class FaceRecognizer:
    """InsightFace face detector + ArcFace embedder, with graceful fallback.

    Args:
        model_name:  InsightFace model pack ("buffalo_l" default, "buffalo_s" on
                     the CPU tier).
        match_threshold: Cosine floor for a confident gallery match.
        det_size:    SCRFD detection input size (square).
        _app:        Pre-built ``FaceAnalysis`` (or duck-typed fake) for tests/CI;
                     bypasses the real model download/prepare.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        *,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
        det_size: int = 640,
        _app: Any | None = None,
    ) -> None:
        self._model_name = model_name
        self._match_threshold = match_threshold
        self._det_size = det_size
        self._app: Any | None = _app
        self._available = _app is not None
        # Human-readable reason the model failed to load (``None`` when healthy).
        # Surfaced so a silent insightface/model failure (the "faces never save"
        # symptom, common on offline Windows first-run) becomes diagnosable in
        # the Services panel / logs instead of vanishing into a swallowed except.
        self.last_error: str | None = None
        if _app is None:
            self._try_init()

    def _try_init(self) -> None:
        """Build + prepare the FaceAnalysis app; disable on any failure.

        The SCRFD detector + ArcFace embedder are ONNX models, so we run them on
        the same hardware EP the detector uses (CoreML on macOS / CUDA·DML
        elsewhere) instead of the historic CPU-only path — face recognition was
        the dominant per-frame cost (insightface on CPU is ~12-15 ms/run).  CPU
        is always appended as a fallback, and ``ctx_id`` follows the EP (>=0 for
        an accelerator, -1 only when CPU is the best we have).
        """
        global _WARNED_UNAVAILABLE
        try:
            from insightface.app import FaceAnalysis  # noqa: PLC0415

            from autoptz.engine.runtime.inference import EP  # noqa: PLC0415

            # Force CPU for face on purpose.  insightface/CoreML balloons the
            # buffalo pack to ~1.9 GB (CoreML compiles every sub-model + ANE/GPU
            # buffers); CPU keeps the same models at ~1.3 GB.  Face runs only a few
            # Hz on a single target (~12-15 ms/run on CPU), so the GPU buys nothing
            # here and the memory saving is large — the dominant cost in the app.
            providers = [EP.CPU.value]
            ctx_id = -1

            # Load ONLY the modules AutoPTZ uses — SCRFD detection (with its 5
            # keypoints) + the ArcFace embedding.  The default pack also loads 2D/3D
            # landmark + gender/age models we never use; skipping them trims memory
            # and load time.
            app = FaceAnalysis(
                name=self._model_name,
                root=insightface_root(),
                providers=providers,
                allowed_modules=["detection", "recognition"],
            )
            app.prepare(ctx_id=ctx_id, det_size=(self._det_size, self._det_size))
            self._app = app
            self._available = True
            self.last_error = None
            log.info(
                "FaceRecognizer ready (insightface %s, ep=CPU, modules=detection+recognition)",
                self._model_name,
            )
        except Exception as exc:  # noqa: BLE001 — missing dep/model/network must not raise
            self._app = None
            self._available = False
            # Capture the concrete reason (ImportError / model-not-found / ORT EP /
            # the NumPy-2 C-ext break) so the Services panel and logs can show
            # *why* face recognition is off instead of failing silently.
            self.last_error = f"{type(exc).__name__}: {exc}"
            if not _WARNED_UNAVAILABLE:
                _WARNED_UNAVAILABLE = True
                log.warning(
                    "insightface unavailable for model %r (%s); face recognition "
                    "disabled — manual click-to-track still works.",
                    self._model_name,
                    self.last_error,
                    exc_info=True,
                )

    @property
    def available(self) -> bool:
        return self._available

    @property
    def threshold(self) -> float:
        return self._match_threshold

    # ── detection + embedding ────────────────────────────────────────────────────

    def detect(self, frame: NDArray[np.uint8]) -> list[FaceObservation]:
        """Detect faces in *frame* (BGR) and return their ArcFace embeddings.

        Returns ``[]`` when disabled or on any inference error (never raises).
        """
        if not self._available or self._app is None or frame is None:
            return []
        try:
            faces = self._app.get(frame)
        except Exception:  # noqa: BLE001
            log.debug("FaceRecognizer.detect failed", exc_info=True)
            return []

        out: list[FaceObservation] = []
        for f in faces:
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                emb = getattr(f, "embedding", None)
            if emb is None:
                continue
            bbox = getattr(f, "bbox", None)
            if bbox is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in np.asarray(bbox).ravel()[:4])
            out.append(
                FaceObservation(
                    bbox=(x1, y1, x2, y2),
                    embedding=normalize(emb),
                    det_score=float(getattr(f, "det_score", 0.0)),
                    kps=self._extract_kps(f),
                )
            )
        return out

    @staticmethod
    def _extract_kps(face: Any) -> tuple[tuple[float, float], ...] | None:
        """Return the SCRFD 5-point landmarks as ``(x, y)`` pairs, or ``None``.

        insightface exposes them as ``kps`` (preferred) or ``landmark_2d_106``;
        we only need the first 5 ``(x, y)`` rows for the yaw estimate.  Never
        raises — any odd shape degrades to ``None``.
        """
        raw = getattr(face, "kps", None)
        if raw is None:
            raw = getattr(face, "landmark_2d_106", None)
        if raw is None:
            return None
        try:
            arr = np.asarray(raw, dtype=np.float32).reshape(-1, 2)
            if arr.shape[0] < 3:
                return None
            return tuple((float(x), float(y)) for x, y in arr[:5])
        except Exception:  # noqa: BLE001
            return None

    # ── gallery matching ─────────────────────────────────────────────────────────

    def match(
        self,
        embedding: NDArray[np.floating],
        service: IdentityService,
        *,
        threshold: float | None = None,
        include_disabled: bool = False,
    ) -> IdentityMatch | None:
        """Best gallery match for *embedding*, or None below threshold.

        By default only *enabled* identities are considered (auto-follow gallery).
        Pass ``include_disabled=True`` for **recognition / de-duplication**: this
        also matches auto-harvested ("Person N", disabled) records so the same
        face is recognised as already-known instead of being re-harvested.
        """
        thr = self._match_threshold if threshold is None else threshold
        candidates = (
            service.matchable_identities() if include_disabled else service.enabled_identities()
        )
        best: IdentityMatch | None = None
        for ident in candidates:
            score = service.best_score(ident.id, embedding)
            if score >= thr and (best is None or score > best.score):
                best = IdentityMatch(identity_id=ident.id, name=ident.name, score=score)
        return best
