"""AutoPTZ Mark — persisted benchmark session helpers.

The in-process Mark benchmark stores its latest setup in a single ``ConfigStore``
key (``mark_session``, JSON). ``ConfigStore`` has no ``delete_setting`` method, so
the ``clear_*`` helpers fall back to ``set_setting(key, None)`` and
``load_mark_session`` treats a falsy/absent value as "no session".
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

MARK_SESSION_KEY = "mark_session"
_GEOMETRY_KEYS = ("win_geometry", "win_state")

# The bundled demo clip (1080p H.264, real pedestrians).  Resolved the same way
# branding.logo_path() resolves the logo asset: ``sys._MEIPASS`` in a frozen
# bundle, else ``autoptz/assets/<file>`` relative to this package.
_CLIP_FILENAME = "mark_people_1080p.mp4"


@dataclass(frozen=True)
class ClipMetadata:
    """A selectable bundled Mark clip: its id, asset filename, and native cadence.

    ``native_fps`` is the clip's *own* recorded frame rate (24/30/60); the Mark
    engine feeds it to the synthetic camera so the source paces at the clip's true
    cadence rather than a fixed 30.  ``purpose`` is a free-form scenario tag
    ("tracking"/"crowd"/"framing"/"fast-motion") for grouping / UX copy.

    ``native_resolution`` is the master clip's *own* recorded frame size (w, h);
    the transcode cache tags every (res, fps) target relative to it as
    native/upscaled/downscaled so the UI can flag synthetic (upscaled) variants.
    ``capability_tags`` lists the AI features the scene meaningfully exercises
    ("tracking"/"reid"/"center-stage"/"face") for capability-gated UX copy.
    """

    id: str
    filename: str
    label: str
    native_fps: float
    purpose: str
    native_resolution: tuple[int, int]
    capability_tags: tuple[str, ...]


# The selectable clip library.  Keyed by stable id (what lands in MarkSession).
# The .mp4 assets are produced in parallel; entries reference them by filename and
# DO NOT require the file to exist on disk (clip_available() reports presence).
CLIP_LIBRARY: dict[str, ClipMetadata] = {
    "crowd": ClipMetadata(
        id="crowd",
        filename="mark_crowd_30.mp4",
        label="Crowd Crossing — 30 fps (re-ID)",
        native_fps=30.0,
        purpose="crowd",
        native_resolution=(1280, 720),
        capability_tags=("tracking", "reid"),
    ),
    "pedestrians": ClipMetadata(
        id="pedestrians",
        filename=_CLIP_FILENAME,
        label="Pedestrians — 30 fps",
        native_fps=30.0,
        purpose="tracking",
        native_resolution=(1920, 1080),
        capability_tags=("tracking",),
    ),
    "cinematic_24": ClipMetadata(
        id="cinematic_24",
        filename="mark_people_24.mp4",
        label="Cinematic People — 24 fps",
        native_fps=24.0,
        purpose="framing",
        native_resolution=(1920, 1080),
        capability_tags=("center-stage",),
    ),
    "cinematic_60": ClipMetadata(
        id="cinematic_60",
        filename="mark_people_60.mp4",
        label="Cinematic People — 60 fps",
        native_fps=60.0,
        purpose="high-fps",
        native_resolution=(1920, 1080),
        capability_tags=("center-stage",),
    ),
    "faces": ClipMetadata(
        id="faces",
        filename="mark_faces_30.mp4",
        label="Faces — 30 fps (recognition)",
        native_fps=30.0,
        purpose="face",
        native_resolution=(1280, 720),
        capability_tags=("face",),
    ),
}

# Default clip: the HD crowd crossing — 30 fps, many trackable people, the most
# striking "3DMark-style" first impression of the bundled scenes.
DEFAULT_CLIP_ID = "crowd"

# Resolution presets → synthetic frame size (w, h).  720p is the default/fallback.
_RESOLUTION_SIZES: dict[str, tuple[int, int]] = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}
# Model choice → the engine's detector-tier vocabulary.  "auto" keeps the default
# tier; "nano"/"small" map to the fast/balanced tiers the engine_client already
# understands (it aliases nano→fast, small→balanced itself, but we normalise here
# so the Mark engine can prime the tier directly without a round-trip).
_MODEL_TIERS: dict[str, str] = {
    "auto": "auto",
    "nano": "fast",
    "small": "balanced",
    "medium": "medium",
}


_TRANSCODE_CACHE: object | None = None


def _transcode_cache() -> object:
    """Lazily build (once) a shared TranscodeCache for the availability table.

    Imported lazily — the engine pulls in cv2 — and reused across calls because
    constructing one only mkdirs a cache dir.  ``valid_combos`` is a pure
    instance method (no I/O), so a single shared instance is safe.
    """
    global _TRANSCODE_CACHE
    if _TRANSCODE_CACHE is None:
        from autoptz.engine.pipeline.transcode_cache import TranscodeCache  # noqa: PLC0415

        _TRANSCODE_CACHE = TranscodeCache()
    return _TRANSCODE_CACHE


def _clip_path(filename: str = _CLIP_FILENAME) -> Path:
    """Absolute path to a bundled demo clip asset (source and frozen runs)."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        bundled = Path(meipass) / "autoptz" / "assets" / filename
        if bundled.is_file():
            return bundled
    # Source / editable: this module is autoptz/ui/mark_session.py → ../assets/<file>.
    return Path(__file__).resolve().parent.parent / "assets" / filename


@dataclass(frozen=True)
class MarkSession:
    profile: str = "simple_follow"
    source: str = "clip"  # "clip" | "ndi"
    floor_fps: float = 30.0
    max_cameras: int = 4
    dwell_s: float = 10.0
    resolution: str = "1080p"  # "720p" | "1080p" | "4k"
    model: str = "small"  # "auto" | "nano" | "small" | "medium"
    clip_id: str = ""  # selected CLIP_LIBRARY id; "" → DEFAULT_CLIP_ID

    def __post_init__(self) -> None:
        # The user-facing "synthetic" (drawn-people) source was removed: the only
        # sources are "clip" and "ndi".  Normalise here so a legacy persisted value
        # (or any unknown/malformed source) is tolerated by mapping to "clip" — the
        # drawn scene now survives only as the env-gated ground-truth scene, never
        # as a user-selectable source.  Frozen dataclass → set via object.__setattr__.
        normalized = str(self.source).strip().lower()
        if normalized != "ndi":
            normalized = "clip"
        if normalized != self.source:
            object.__setattr__(self, "source", normalized)

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "source": self.source,
            "floor_fps": self.floor_fps,
            "max_cameras": self.max_cameras,
            "dwell_s": self.dwell_s,
            "resolution": self.resolution,
            "model": self.model,
            "clip_id": self.clip_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> MarkSession:
        return cls(
            profile=str(d.get("profile", "simple_follow")),
            source=str(d.get("source", "clip")),
            floor_fps=float(d.get("floor_fps", 30.0)),  # type: ignore[arg-type]
            max_cameras=int(d.get("max_cameras", 4)),  # type: ignore[arg-type]
            dwell_s=float(d.get("dwell_s", 10.0)),  # type: ignore[arg-type]
            resolution=str(d.get("resolution", "1080p")),
            model=str(d.get("model", "small")),
            clip_id=str(d.get("clip_id", "")),
        )

    def resolution_size(self) -> tuple[int, int]:
        """The (width, height) for this session's resolution; 720p on any miss."""
        return _RESOLUTION_SIZES.get(str(self.resolution).strip().lower(), (1280, 720))

    def detector_tier(self) -> str:
        """The engine detector tier for this session's model; "auto" on any miss."""
        return _MODEL_TIERS.get(str(self.model).strip().lower(), "auto")

    def target_fps(self) -> float:
        """The benchmark pass-target fps for this session.

        For a CLIP source the target is the clip's *native* cadence: a 24fps clip
        physically can't sustain a 30fps target (the decoder caps at the clip's own
        rate), and a 60fps clip should be graded against 60 — so picking a scene
        sets the rate it's tested at.  For NDI sources (which can render
        at any rate) the user-chosen ``floor_fps`` is the target. (A 'synthetic' source
        value is normalised to 'clip', so it takes the native-fps branch above.)
        """
        if self.is_clip():
            return float(self.clip_info().native_fps)
        return float(self.floor_fps)

    def is_clip(self) -> bool:
        """True when this session's source is the bundled real-people clip."""
        return str(self.source).strip().lower() == "clip"

    def clip_info(self) -> ClipMetadata:
        """Resolve this session's clip id against :data:`CLIP_LIBRARY`.

        Empty id → the :data:`DEFAULT_CLIP_ID` entry; a non-empty but unknown id
        logs a warning and falls back to the default rather than failing the run.
        """
        cid = str(self.clip_id).strip()
        if not cid:
            return CLIP_LIBRARY[DEFAULT_CLIP_ID]
        meta = CLIP_LIBRARY.get(cid)
        if meta is None:
            log.warning(
                "Mark clip id %r is not in the clip library; falling back to %r.",
                cid,
                DEFAULT_CLIP_ID,
            )
            return CLIP_LIBRARY[DEFAULT_CLIP_ID]
        return meta

    def capability_tags(self) -> tuple[str, ...]:
        """The AI features this session's clip meaningfully exercises.

        Resolves the clip id against :data:`CLIP_LIBRARY` (default on miss) and
        returns its :attr:`ClipMetadata.capability_tags` so the UI can gate /
        annotate capability-specific controls (re-ID, center-stage, face).
        """
        return self.clip_info().capability_tags

    def available_variants(self) -> list[dict]:
        """The (res, fps) availability table for this session's clip.

        Resolves the clip metadata, then asks the transcode cache to tag every
        target combo relative to the master's native (resolution, fps).  Each
        dict carries ``res``/``fps``/``res_tag``/``fps_tag``/``synthetic`` — the
        UI renders these so the user sees which variants are real captured
        fidelity vs. upscaled / frame-interpolated (synthetic).
        """
        meta = self.clip_info()
        cache = _transcode_cache()
        return cache.valid_combos(meta.native_resolution, meta.native_fps)  # type: ignore[attr-defined]

    def clip_path(self) -> str:
        """Absolute path (str) to the selected bundled clip, for the SyntheticAdapter."""
        return str(_clip_path(self.clip_info().filename))

    def clip_available(self) -> bool:
        """True when the selected bundled clip actually exists on disk.

        The clip ships with frozen bundles (packaging trees ``autoptz/assets``),
        but it isn't guaranteed present in a fresh clone / CI checkout.  Callers
        use this to fall back to the drawn-people scene *transparently* (with a log
        line) instead of silently degrading the advertised "real people" demo.
        """
        return _clip_path(self.clip_info().filename).is_file()


def load_mark_session(store: object) -> MarkSession | None:
    raw = store.get_setting(MARK_SESSION_KEY, None)  # type: ignore[attr-defined]
    if not raw or not isinstance(raw, dict):
        return None
    return MarkSession.from_dict(raw)


def store_mark_session(store: object, session: MarkSession) -> None:
    store.set_setting(MARK_SESSION_KEY, session.to_dict())  # type: ignore[attr-defined]


def clear_mark_session(store: object) -> None:
    _delete(store, MARK_SESSION_KEY)


def clear_window_geometry(store: object) -> None:
    for key in _GEOMETRY_KEYS:
        _delete(store, key)


def _delete(store: object, key: str) -> None:
    deleter = getattr(store, "delete_setting", None)
    if callable(deleter):
        deleter(key)
    else:
        store.set_setting(key, None)  # type: ignore[attr-defined]
