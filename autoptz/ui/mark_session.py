"""AutoPTZ Mark — session handoff across the relaunch + relaunch helpers (pure).

The Mark-session config crosses the relaunch via a single ``ConfigStore`` key
(``mark_session``, JSON) rather than argv, so the relaunched process reads its
parameters from the store.  ``ConfigStore`` has no ``delete_setting`` method, so
the ``clear_*`` helpers fall back to ``set_setting(key, None)`` and
``load_mark_session`` treats a falsy/absent value as "no session".
"""

from __future__ import annotations

import logging
import subprocess
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
    """

    id: str
    filename: str
    label: str
    native_fps: float
    purpose: str


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
    ),
    "pedestrians": ClipMetadata(
        id="pedestrians",
        filename=_CLIP_FILENAME,
        label="Pedestrians — 30 fps",
        native_fps=30.0,
        purpose="tracking",
    ),
    "cinematic_24": ClipMetadata(
        id="cinematic_24",
        filename="mark_people_24.mp4",
        label="Cinematic People — 24 fps",
        native_fps=24.0,
        purpose="framing",
    ),
    "cinematic_60": ClipMetadata(
        id="cinematic_60",
        filename="mark_people_60.mp4",
        label="Cinematic People — 60 fps",
        native_fps=60.0,
        purpose="high-fps",
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
    profile: str = "full"
    source: str = "clip"  # "clip" | "synthetic" | "ndi"
    floor_fps: float = 30.0
    max_cameras: int = 4
    dwell_s: float = 10.0
    resolution: str = "1080p"  # "720p" | "1080p" | "4k"
    model: str = "small"  # "auto" | "nano" | "small" | "medium"
    clip_id: str = ""  # selected CLIP_LIBRARY id; "" → DEFAULT_CLIP_ID

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
            profile=str(d.get("profile", "full")),
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
        sets the rate it's tested at.  For synthetic / NDI sources (which can render
        at any rate) the user-chosen ``floor_fps`` is the target.
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


def relaunch_argv(*, mark: bool) -> list[str]:
    """DEPRECATED: build the relaunch arg-vector for the old subprocess Mark flow.

    AutoPTZ Mark is now an in-process swap (Help → Run AutoPTZ Mark…); nothing in
    the app calls this anymore.  Retained for backward compatibility / the
    deprecated ``--mark`` flag only.
    """
    if getattr(sys, "frozen", False):
        argv = [sys.executable]
    else:
        argv = [sys.executable, "-m", "autoptz"]
    if mark:
        argv.append("--mark")
    return argv


def relaunch(*, mark: bool) -> None:
    """DEPRECATED: spawn a fresh AutoPTZ process (old subprocess Mark flow).

    Unused by the in-process Mark lifecycle; kept for backward compatibility.
    """
    subprocess.Popen(relaunch_argv(mark=mark), close_fds=True)  # noqa: S603 — fixed argv
