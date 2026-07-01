"""Benchmark profiles for AutoPTZ Mark.

A profile maps a human name to (a) the engine feature switches applied to every
synthetic camera for the run and (b) a score weight.  The feature keys mirror
``autoptz.engine.camera_worker._DEFAULT_FEATURES`` so the same dict drops
straight into ``Supervisor.prime_features`` / ``SetFeaturesCmd``.

``simple_follow`` — the 2.2 production target: detector + tracker only. This is
                    the first profile to compare when deciding whether optional
                    identity/pose services are overbuilt for a deployment.
``pose_follow``   — detector + tracker + pose. Isolates the cost of pose aiming
                    from face recognition and appearance ReID.
``full``          — all current vision services on: detection + tracking + face
                    + pose + ReID. Even on synthetic frames with no person, the
                    detector still runs and incurs its inference cost, so the
                    throughput number is valid without a bundled person asset.
``streams``       — all inference OFF (capture + preview only): an upper bound
                    on how many streams the machine can ingest/paint before ML.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkProfile:
    """One benchmark workload: feature toggles + a score weight."""

    name: str
    description: str
    features: dict[str, bool]
    weight: float


_ALL_ON: dict[str, bool] = {
    "detection": True,
    "tracking": True,
    "face_recognition": True,
    "pose": True,
    "reid": True,
}
_ALL_OFF: dict[str, bool] = {k: False for k in _ALL_ON}


PROFILES: dict[str, BenchmarkProfile] = {
    "simple_follow": BenchmarkProfile(
        name="simple_follow",
        description="Simple Follow: detector + tracker only.",
        features={
            "detection": True,
            "tracking": True,
            "face_recognition": False,
            "pose": False,
            "reid": False,
        },
        weight=1.0,
    ),
    "pose_follow": BenchmarkProfile(
        name="pose_follow",
        description="Detector + tracker + pose aiming; face/ReID off.",
        features={
            "detection": True,
            "tracking": True,
            "face_recognition": False,
            "pose": True,
            "reid": False,
        },
        weight=1.0,
    ),
    "full": BenchmarkProfile(
        name="full",
        description="Detection + tracking + face + pose + ReID + center-stage (full load).",
        features=dict(_ALL_ON),
        weight=1.0,
    ),
    "streams": BenchmarkProfile(
        name="streams",
        description="Capture + preview only (all inference off).",
        features=dict(_ALL_OFF),
        weight=0.8,
    ),
}


def get_profile(name: str) -> BenchmarkProfile:
    """Return the named profile, or raise ``ValueError`` listing the valid names."""
    try:
        return PROFILES[name]
    except KeyError:
        valid = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown benchmark profile {name!r}; valid: {valid}") from None
