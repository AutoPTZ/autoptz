"""AutoPTZ Mark — session handoff across the relaunch + relaunch helpers (pure).

The Mark-session config crosses the relaunch via a single ``ConfigStore`` key
(``mark_session``, JSON) rather than argv, so the relaunched process reads its
parameters from the store.  ``ConfigStore`` has no ``delete_setting`` method, so
the ``clear_*`` helpers fall back to ``set_setting(key, None)`` and
``load_mark_session`` treats a falsy/absent value as "no session".
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MARK_SESSION_KEY = "mark_session"
_GEOMETRY_KEYS = ("win_geometry", "win_state")

# The bundled demo clip (1080p H.264, real pedestrians).  Resolved the same way
# branding.logo_path() resolves the logo asset: ``sys._MEIPASS`` in a frozen
# bundle, else ``autoptz/assets/<file>`` relative to this package.
_CLIP_FILENAME = "mark_people_1080p.mp4"

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


def _clip_path() -> Path:
    """Absolute path to the bundled demo clip (source and frozen runs)."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        bundled = Path(meipass) / "autoptz" / "assets" / _CLIP_FILENAME
        if bundled.is_file():
            return bundled
    # Source / editable: this module is autoptz/ui/mark_session.py → ../assets/<file>.
    return Path(__file__).resolve().parent.parent / "assets" / _CLIP_FILENAME


@dataclass(frozen=True)
class MarkSession:
    profile: str = "full"
    source: str = "clip"  # "clip" | "synthetic" | "ndi"
    floor_fps: float = 30.0
    max_cameras: int = 4
    dwell_s: float = 10.0
    resolution: str = "1080p"  # "720p" | "1080p" | "4k"
    model: str = "small"  # "auto" | "nano" | "small" | "medium"

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "source": self.source,
            "floor_fps": self.floor_fps,
            "max_cameras": self.max_cameras,
            "dwell_s": self.dwell_s,
            "resolution": self.resolution,
            "model": self.model,
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
        )

    def resolution_size(self) -> tuple[int, int]:
        """The (width, height) for this session's resolution; 720p on any miss."""
        return _RESOLUTION_SIZES.get(str(self.resolution).strip().lower(), (1280, 720))

    def detector_tier(self) -> str:
        """The engine detector tier for this session's model; "auto" on any miss."""
        return _MODEL_TIERS.get(str(self.model).strip().lower(), "auto")

    def is_clip(self) -> bool:
        """True when this session's source is the bundled real-people clip."""
        return str(self.source).strip().lower() == "clip"

    def clip_path(self) -> str:
        """Absolute path (str) to the bundled demo clip, for the SyntheticAdapter."""
        return str(_clip_path())

    def clip_available(self) -> bool:
        """True when the bundled demo clip actually exists on disk.

        The clip ships with frozen bundles (packaging trees ``autoptz/assets``),
        but it isn't guaranteed present in a fresh clone / CI checkout.  Callers
        use this to fall back to the drawn-people scene *transparently* (with a log
        line) instead of silently degrading the advertised "real people" demo.
        """
        return _clip_path().is_file()


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
