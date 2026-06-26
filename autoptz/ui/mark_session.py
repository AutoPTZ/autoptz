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

MARK_SESSION_KEY = "mark_session"
_GEOMETRY_KEYS = ("win_geometry", "win_state")


@dataclass(frozen=True)
class MarkSession:
    profile: str = "full"
    source: str = "synthetic"  # "synthetic" | "ndi"
    floor_fps: float = 24.0
    max_cameras: int = 16
    dwell_s: float = 15.0

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "source": self.source,
            "floor_fps": self.floor_fps,
            "max_cameras": self.max_cameras,
            "dwell_s": self.dwell_s,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> MarkSession:
        return cls(
            profile=str(d.get("profile", "full")),
            source=str(d.get("source", "synthetic")),
            floor_fps=float(d.get("floor_fps", 24.0)),  # type: ignore[arg-type]
            max_cameras=int(d.get("max_cameras", 16)),  # type: ignore[arg-type]
            dwell_s=float(d.get("dwell_s", 15.0)),  # type: ignore[arg-type]
        )


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
    if getattr(sys, "frozen", False):
        argv = [sys.executable]
    else:
        argv = [sys.executable, "-m", "autoptz"]
    if mark:
        argv.append("--mark")
    return argv


def relaunch(*, mark: bool) -> None:
    subprocess.Popen(relaunch_argv(mark=mark), close_fds=True)  # noqa: S603 — fixed argv
