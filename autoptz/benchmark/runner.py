"""AutoPTZ Mark — the headless ramp runner + score math.

The runner ramps synthetic cameras through the real engine pipeline and reports
the most cameras the machine can sustain above an fps floor, plus a single
weighted score.  The ramp/scoring control logic is isolated behind a
``sample_fn(n) -> list[float]`` seam: the real benchmark supplies a sampler that
drives a ``Supervisor`` (see ``run_benchmark``); unit tests supply a deterministic
one so the math is verified with no real inference.

Throughput is measured by running the full pipeline at the configured cadence on
synthetic frames.  The detector runs (and incurs its inference cost) even when it
finds nothing, so the number is valid without a bundled person asset.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from autoptz.benchmark.profiles import BenchmarkProfile

log = logging.getLogger(__name__)

_NOMINAL_FPS = 30.0  # score normaliser: sustained fps / 30 fps target
_DROP_POLICY = "steady_state_zero_source_mutation_grace_only"


# ── engine-reported tracking quality ──────────────────────────────────────────


@dataclass(frozen=True)
class QualityMetrics:
    """Tracking-quality summary for ONE camera over a measurement window.

    Derived purely from the engine's ``TelemetryMsg`` stream (no inference here):
    how fast the target was acquired, how often / how long it was lost, identity
    churn, how much of the window it was held, and the mean confidence while held.

    Frame-count derived durations are converted to seconds via the observed mean
    fps (falling back to the runner's ``fps_hint``).  ``time_to_first_acquire_s``
    is ``None`` when the target was never acquired during the window.
    """

    time_to_first_acquire_s: float | None
    total_lost_duration_s: float
    longest_lost_duration_s: float
    lost_event_count: int
    reacquire_count: int
    id_switch_count: int
    target_hold_pct: float
    mean_target_confidence: float
    fps: float
    dropped_frames: int
    app_induced_drops: int
    frames_delivered: int
    frames_dropped_est: int
    delivered_fps: float
    source_fps: float
    duplicate_frames: int
    stale_frames: int
    ndi_queue_depth: int
    ndi_queue_audio: int
    ndi_queue_metadata: int
    ndi_total_video_frames: int
    ndi_dropped_video_frames: int
    ndi_total_audio_frames: int
    ndi_dropped_audio_frames: int
    ndi_total_metadata_frames: int
    ndi_dropped_metadata_frames: int
    ndi_connections: int
    ndi_fourcc: str
    ndi_buffer_ms: float
    ndi_conversion_ms: float
    ndi_copy_ms: float

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly dict.  ``time_to_first_acquire_s`` is ``None`` (→ JSON
        ``null``) when the target was never acquired during the window."""
        ttfa = self.time_to_first_acquire_s
        return {
            "time_to_first_acquire_s": (None if ttfa is None else round(ttfa, 3)),
            "total_lost_duration_s": round(self.total_lost_duration_s, 3),
            "longest_lost_duration_s": round(self.longest_lost_duration_s, 3),
            "lost_event_count": self.lost_event_count,
            "reacquire_count": self.reacquire_count,
            "id_switch_count": self.id_switch_count,
            "target_hold_pct": round(self.target_hold_pct, 2),
            "mean_target_confidence": round(self.mean_target_confidence, 4),
            "fps": round(self.fps, 2),
            "dropped_frames": self.dropped_frames,
            "app_induced_drops": self.app_induced_drops,
            "frames_delivered": self.frames_delivered,
            "frames_dropped_est": self.frames_dropped_est,
            "delivered_fps": round(self.delivered_fps, 2),
            "source_fps": round(self.source_fps, 2),
            "duplicate_frames": self.duplicate_frames,
            "stale_frames": self.stale_frames,
            "ndi_queue_depth": self.ndi_queue_depth,
            "ndi_queue_audio": self.ndi_queue_audio,
            "ndi_queue_metadata": self.ndi_queue_metadata,
            "ndi_total_video_frames": self.ndi_total_video_frames,
            "ndi_dropped_video_frames": self.ndi_dropped_video_frames,
            "ndi_total_audio_frames": self.ndi_total_audio_frames,
            "ndi_dropped_audio_frames": self.ndi_dropped_audio_frames,
            "ndi_total_metadata_frames": self.ndi_total_metadata_frames,
            "ndi_dropped_metadata_frames": self.ndi_dropped_metadata_frames,
            "ndi_connections": self.ndi_connections,
            "ndi_fourcc": self.ndi_fourcc,
            "ndi_buffer_ms": round(self.ndi_buffer_ms, 3),
            "ndi_conversion_ms": round(self.ndi_conversion_ms, 3),
            "ndi_copy_ms": round(self.ndi_copy_ms, 3),
        }


class PerCameraQualityAccumulator:
    """Fold a single camera's ``TelemetryMsg`` stream into ``QualityMetrics``.

    Pure / headless / thread-agnostic: ``on_telemetry`` is called once per frame
    (in arrival order) and ``finalize`` returns the summary.  The target is the
    first ``TrackInfo`` with ``is_target`` set in each frame.  "Held" means a
    target track is present **and** not ``lost``; the target is considered LOST
    when its ``lost`` flag is set OR the ``is_target`` track disappears from the
    frame entirely after it was once acquired.
    """

    def __init__(self, fps_hint: float) -> None:
        self._fps_hint = float(fps_hint) if fps_hint and fps_hint > 0.0 else _NOMINAL_FPS
        self._total_frames = 0
        self._hold_frames = 0
        self._conf_sum = 0.0
        # First-acquire bookkeeping: frames observed before the first hold frame.
        self._acquired = False
        self._frames_before_acquire = 0
        # Lost-event bookkeeping (only meaningful once acquired).
        self._lost_event_count = 0
        self._reacquire_count = 0
        self._lost_frames_total = 0
        self._cur_lost_run = 0
        self._longest_lost_run = 0
        self._target_was_present = False  # held on the previous frame
        # Identity churn: the target's track_id (fallback identity_id) last seen
        # on a target-present frame.
        self._id_switch_count = 0
        self._last_target_key: object | None = None
        # Rate / drops.
        self._fps_sum = 0.0
        self._fps_samples = 0
        self._dropped_frames = 0
        self._initial_dropped_frames: int | None = None
        self._app_induced_drops = 0
        # Source-health telemetry, preserved in the Mark JSON artifact.
        self._frames_delivered = 0
        self._frames_dropped_est = 0
        self._delivered_fps_sum = 0.0
        self._delivered_fps_samples = 0
        self._source_fps_sum = 0.0
        self._source_fps_samples = 0
        self._duplicate_frames = 0
        self._stale_frames = 0
        self._ndi_queue_depth = -1
        self._ndi_queue_audio = -1
        self._ndi_queue_metadata = -1
        self._ndi_total_video_frames = 0
        self._ndi_dropped_video_frames = 0
        self._ndi_total_audio_frames = 0
        self._ndi_dropped_audio_frames = 0
        self._ndi_total_metadata_frames = 0
        self._ndi_dropped_metadata_frames = 0
        self._ndi_connections = -1
        self._ndi_fourcc = ""
        self._ndi_buffer_ms_sum = 0.0
        self._ndi_buffer_ms_samples = 0
        self._ndi_conversion_ms_sum = 0.0
        self._ndi_conversion_ms_samples = 0
        self._ndi_copy_ms_sum = 0.0
        self._ndi_copy_ms_samples = 0

    @staticmethod
    def _find_target(msg: Any) -> Any | None:
        for t in getattr(msg, "tracks", None) or ():
            if getattr(t, "is_target", False):
                return t
        return None

    @staticmethod
    def _target_key(track: Any) -> object:
        tid = getattr(track, "track_id", None)
        if tid is not None:
            return ("tid", tid)
        return ("iid", getattr(track, "identity_id", None))

    def on_telemetry(self, msg: Any) -> None:
        self._total_frames += 1

        # Rate math: prefer msg.fps when present, else the hint.
        msg_fps = float(getattr(msg, "fps", 0.0) or 0.0)
        if msg_fps > 0.0:
            self._fps_sum += msg_fps
            self._fps_samples += 1

        # Dropped frames: latest cumulative value wins (engine reports cumulative).
        drops = getattr(msg, "dropped_frames", None)
        if drops is not None:
            try:
                current_drops = max(0, int(drops))
            except (TypeError, ValueError):
                current_drops = None
            if current_drops is not None:
                self._dropped_frames = current_drops
                if self._initial_dropped_frames is None:
                    self._initial_dropped_frames = current_drops
                if current_drops >= self._initial_dropped_frames:
                    delta = current_drops - self._initial_dropped_frames
                else:
                    # Counter reset during the sample window; treat the new value
                    # as drops observed after reset rather than hiding it.
                    delta = current_drops
                    self._initial_dropped_frames = 0
                self._app_induced_drops = max(self._app_induced_drops, delta)

        delivered = getattr(msg, "frames_delivered", None)
        if delivered is not None:
            try:
                self._frames_delivered = max(0, int(delivered))
            except (TypeError, ValueError):
                pass

        dropped_est = getattr(msg, "frames_dropped_est", None)
        if dropped_est is not None:
            try:
                self._frames_dropped_est = max(0, int(dropped_est))
            except (TypeError, ValueError):
                pass

        delivered_fps = float(getattr(msg, "delivered_fps", 0.0) or 0.0)
        if delivered_fps > 0.0:
            self._delivered_fps_sum += delivered_fps
            self._delivered_fps_samples += 1

        source_fps = float(getattr(msg, "source_fps", 0.0) or 0.0)
        if source_fps > 0.0:
            self._source_fps_sum += source_fps
            self._source_fps_samples += 1

        duplicate_frames = getattr(msg, "duplicate_frames", None)
        if duplicate_frames is not None:
            try:
                self._duplicate_frames = max(0, int(duplicate_frames))
            except (TypeError, ValueError):
                pass

        stale_frames = getattr(msg, "stale_frames", None)
        if stale_frames is not None:
            try:
                self._stale_frames = max(0, int(stale_frames))
            except (TypeError, ValueError):
                pass

        queue_depth = getattr(msg, "ndi_queue_depth", None)
        if queue_depth is not None:
            try:
                self._ndi_queue_depth = int(queue_depth)
            except (TypeError, ValueError):
                pass

        for attr, name, default in (
            ("_ndi_queue_audio", "ndi_queue_audio", -1),
            ("_ndi_queue_metadata", "ndi_queue_metadata", -1),
            ("_ndi_total_video_frames", "ndi_total_video_frames", 0),
            ("_ndi_dropped_video_frames", "ndi_dropped_video_frames", 0),
            ("_ndi_total_audio_frames", "ndi_total_audio_frames", 0),
            ("_ndi_dropped_audio_frames", "ndi_dropped_audio_frames", 0),
            ("_ndi_total_metadata_frames", "ndi_total_metadata_frames", 0),
            ("_ndi_dropped_metadata_frames", "ndi_dropped_metadata_frames", 0),
            ("_ndi_connections", "ndi_connections", -1),
        ):
            raw = getattr(msg, name, default)
            try:
                setattr(self, attr, int(raw))
            except (TypeError, ValueError):
                pass

        fourcc = str(getattr(msg, "ndi_fourcc", "") or "").upper()
        if fourcc:
            self._ndi_fourcc = fourcc

        buffer_ms = float(getattr(msg, "ndi_buffer_ms", 0.0) or 0.0)
        if buffer_ms > 0.0:
            self._ndi_buffer_ms_sum += buffer_ms
            self._ndi_buffer_ms_samples += 1

        conversion_ms = float(getattr(msg, "ndi_conversion_ms", 0.0) or 0.0)
        if conversion_ms > 0.0:
            self._ndi_conversion_ms_sum += conversion_ms
            self._ndi_conversion_ms_samples += 1

        copy_ms = float(getattr(msg, "ndi_copy_ms", 0.0) or 0.0)
        if copy_ms > 0.0:
            self._ndi_copy_ms_sum += copy_ms
            self._ndi_copy_ms_samples += 1

        target = self._find_target(msg)
        held = target is not None and not getattr(target, "lost", False)

        if not self._acquired:
            if held:
                self._acquired = True
            else:
                # Still hunting — this frame counts toward time-to-first-acquire.
                self._frames_before_acquire += 1

        if held:
            self._hold_frames += 1
            self._conf_sum += float(getattr(target, "confidence", 0.0) or 0.0)
            # Identity churn: compare against the last target-present key.
            key = self._target_key(target)
            if self._last_target_key is not None and key != self._last_target_key:
                self._id_switch_count += 1
            self._last_target_key = key

        # Lost-event transitions only matter once the target was ever acquired.
        if self._acquired:
            if held:
                if not self._target_was_present and self._cur_lost_run > 0:
                    # found again after a lost run -> close the event.
                    self._reacquire_count += 1
                    self._longest_lost_run = max(self._longest_lost_run, self._cur_lost_run)
                    self._cur_lost_run = 0
                self._target_was_present = True
            else:
                # Lost: flag set OR the target track vanished after acquisition.
                if self._target_was_present:
                    self._lost_event_count += 1  # entering a new lost run
                self._cur_lost_run += 1
                self._lost_frames_total += 1
                self._target_was_present = False

    def _observed_fps(self) -> float:
        if self._fps_samples > 0:
            return self._fps_sum / self._fps_samples
        return self._fps_hint

    def _observed_delivered_fps(self) -> float:
        if self._delivered_fps_samples > 0:
            return self._delivered_fps_sum / self._delivered_fps_samples
        return 0.0

    def _observed_source_fps(self) -> float:
        if self._source_fps_samples > 0:
            return self._source_fps_sum / self._source_fps_samples
        return 0.0

    def _observed_ndi_conversion_ms(self) -> float:
        if self._ndi_conversion_ms_samples > 0:
            return self._ndi_conversion_ms_sum / self._ndi_conversion_ms_samples
        return 0.0

    def _observed_ndi_buffer_ms(self) -> float:
        if self._ndi_buffer_ms_samples > 0:
            return self._ndi_buffer_ms_sum / self._ndi_buffer_ms_samples
        return 0.0

    def _observed_ndi_copy_ms(self) -> float:
        if self._ndi_copy_ms_samples > 0:
            return self._ndi_copy_ms_sum / self._ndi_copy_ms_samples
        return 0.0

    def finalize(self) -> QualityMetrics:
        fps = self._observed_fps()
        per_frame_s = (1.0 / fps) if fps > 0.0 else 0.0

        # Close any still-open lost run into the longest tally.
        longest_run = max(self._longest_lost_run, self._cur_lost_run)

        if self._acquired:
            ttfa: float | None = self._frames_before_acquire * per_frame_s
        else:
            ttfa = None

        hold_pct = (self._hold_frames / self._total_frames * 100.0) if self._total_frames else 0.0
        mean_conf = (self._conf_sum / self._hold_frames) if self._hold_frames else 0.0

        return QualityMetrics(
            time_to_first_acquire_s=ttfa,
            total_lost_duration_s=self._lost_frames_total * per_frame_s,
            longest_lost_duration_s=longest_run * per_frame_s,
            lost_event_count=self._lost_event_count,
            reacquire_count=self._reacquire_count,
            id_switch_count=self._id_switch_count,
            target_hold_pct=hold_pct,
            mean_target_confidence=mean_conf,
            fps=fps,
            dropped_frames=self._dropped_frames,
            app_induced_drops=self._app_induced_drops,
            frames_delivered=self._frames_delivered,
            frames_dropped_est=self._frames_dropped_est,
            delivered_fps=self._observed_delivered_fps(),
            source_fps=self._observed_source_fps(),
            duplicate_frames=self._duplicate_frames,
            stale_frames=self._stale_frames,
            ndi_queue_depth=self._ndi_queue_depth,
            ndi_queue_audio=self._ndi_queue_audio,
            ndi_queue_metadata=self._ndi_queue_metadata,
            ndi_total_video_frames=self._ndi_total_video_frames,
            ndi_dropped_video_frames=self._ndi_dropped_video_frames,
            ndi_total_audio_frames=self._ndi_total_audio_frames,
            ndi_dropped_audio_frames=self._ndi_dropped_audio_frames,
            ndi_total_metadata_frames=self._ndi_total_metadata_frames,
            ndi_dropped_metadata_frames=self._ndi_dropped_metadata_frames,
            ndi_connections=self._ndi_connections,
            ndi_fourcc=self._ndi_fourcc,
            ndi_buffer_ms=self._observed_ndi_buffer_ms(),
            ndi_conversion_ms=self._observed_ndi_conversion_ms(),
            ndi_copy_ms=self._observed_ndi_copy_ms(),
        )


@dataclass(frozen=True)
class StepResult:
    """One ramp step: N cameras run for the dwell, with observed per-camera fps."""

    cameras: int
    min_fps: float
    mean_fps: float
    per_camera_fps: list[float] = field(default_factory=list)
    sustained: bool = False
    # Raw drops observed around this step. Release gating uses
    # ``steady_state_app_induced_drops`` so add/remove-source churn can be reported
    # separately instead of hidden or treated as a steady-state capture failure.
    app_induced_drops: int = 0
    steady_state_app_induced_drops: int | None = None
    source_mutation_events: int = 0
    source_mutation_allowed_drops: int = 0
    source_mutation_drop_grace_s: float = 0.0
    drop_policy: str = _DROP_POLICY
    # Engine-reported tracking quality keyed by camera id ({cid: QualityMetrics
    # dict}).  Empty when no quality reader is wired (e.g. the math-only unit
    # tests or the GUI/adopted path before Slice 5).
    per_camera_quality: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        total = max(0, int(self.app_induced_drops or 0))
        allowed = max(0, int(self.source_mutation_allowed_drops or 0))
        events = max(0, int(self.source_mutation_events or 0))
        grace = max(0.0, float(self.source_mutation_drop_grace_s or 0.0))
        if self.steady_state_app_induced_drops is None:
            steady = max(0, total - allowed)
        else:
            steady = max(0, int(self.steady_state_app_induced_drops or 0))
        object.__setattr__(self, "app_induced_drops", total)
        object.__setattr__(self, "steady_state_app_induced_drops", steady)
        object.__setattr__(self, "source_mutation_events", events)
        object.__setattr__(self, "source_mutation_allowed_drops", allowed)
        object.__setattr__(self, "source_mutation_drop_grace_s", grace)
        object.__setattr__(self, "drop_policy", str(self.drop_policy or _DROP_POLICY))

    def to_dict(self) -> dict[str, object]:
        return {
            "cameras": self.cameras,
            "min_fps": round(self.min_fps, 2),
            "mean_fps": round(self.mean_fps, 2),
            "per_camera_fps": [round(f, 2) for f in self.per_camera_fps],
            "sustained": self.sustained,
            "app_induced_drops": self.app_induced_drops,
            "steady_state_app_induced_drops": int(self.steady_state_app_induced_drops or 0),
            "source_mutation_events": self.source_mutation_events,
            "source_mutation_allowed_drops": self.source_mutation_allowed_drops,
            "source_mutation_drop_grace_s": round(self.source_mutation_drop_grace_s, 3),
            "drop_policy": self.drop_policy,
            "per_camera_quality": self.per_camera_quality,
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """The full ramp outcome + the AutoPTZ Mark score."""

    profile: str
    weight: float
    floor_fps: float
    max_cameras: int
    sustained_cameras: int
    min_fps_at_sustained: float
    score: float
    profile_description: str = ""
    profile_features: dict[str, bool] = field(default_factory=dict)
    experimental_flags: dict[str, str] = field(default_factory=dict)
    drop_policy: str = _DROP_POLICY
    steps: list[StepResult] = field(default_factory=list)
    # The Mark scene this ramp ran (CLIP_LIBRARY id), for the result/CSV context.
    scene_clip_id: str = ""
    # Ground-truth CLEAR-MOT accuracy per camera (only on synthetic scenes with
    # AUTOPTZ_MARK_GT): {camera_id: {miss_rate, id_switch_rate, motp, mota}}.
    ground_truth: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "profile_description": self.profile_description,
            "profile_features": dict(self.profile_features),
            "experimental_flags": dict(self.experimental_flags),
            "drop_policy": self.drop_policy,
            "weight": self.weight,
            "floor_fps": self.floor_fps,
            "max_cameras": self.max_cameras,
            "sustained_cameras": self.sustained_cameras,
            "min_fps_at_sustained": round(self.min_fps_at_sustained, 2),
            "score": self.score,
            "scene_clip_id": self.scene_clip_id,
            "steps": [s.to_dict() for s in self.steps],
            "ground_truth": self.ground_truth,
        }

    def summary(self) -> str:
        return (
            f"AutoPTZ Mark [{self.profile}]: score {self.score:.2f} — sustained "
            f"{self.sustained_cameras} camera(s) @ >={self.floor_fps:.0f} fps "
            f"(min {self.min_fps_at_sustained:.1f} fps at that count)."
        )


def _quality_app_induced_drops(quality: dict[str, dict]) -> int:
    """Return measured app-induced drops across one quality snapshot.

    Newer quality snapshots expose ``app_induced_drops`` as a per-window delta.
    Older/external snapshots may only carry ``dropped_frames``; for those, treat a
    positive cumulative value as a drop signal rather than accidentally passing an
    unhealthy run.
    """
    total = 0
    for row in quality.values():
        if not isinstance(row, dict):
            continue
        raw = row.get("app_induced_drops", row.get("dropped_frames", 0))
        try:
            total += max(0, int(raw or 0))
        except (TypeError, ValueError):
            continue
    return total


def _source_mutation_snapshot(
    reader: Callable[[], dict[str, object] | None] | None,
) -> dict[str, object]:
    """Read source-mutation accounting for the just-sampled step.

    The reader reports drops that happened during an explicit add/remove-source
    transition outside the steady measurement window. They are preserved in the
    artifact but do not fail the steady-state capture gate.
    """
    if reader is None:
        return {}
    try:
        raw = reader() or {}
    except Exception:  # noqa: BLE001 — accounting must never abort the ramp
        log.debug("benchmark source_mutation_reader failed", exc_info=True)
        return {}
    return dict(raw) if isinstance(raw, dict) else {}


def _drop_accounting(
    quality: dict[str, dict],
    source_mutation: dict[str, object] | None = None,
) -> tuple[int, int, int, int, float]:
    """Return ``(total, steady, events, allowed, grace_s)`` for a step."""
    steady = _quality_app_induced_drops(quality)
    mutation = source_mutation or {}
    try:
        allowed = max(0, int(mutation.get("source_mutation_allowed_drops", 0) or 0))
    except (TypeError, ValueError):
        allowed = 0
    try:
        events = max(0, int(mutation.get("source_mutation_events", 0) or 0))
    except (TypeError, ValueError):
        events = 0
    try:
        grace_s = max(0.0, float(mutation.get("source_mutation_drop_grace_s", 0.0) or 0.0))
    except (TypeError, ValueError):
        grace_s = 0.0
    total = steady + allowed
    return total, steady, events, allowed, grace_s


def _experimental_flag_snapshot() -> dict[str, str]:
    """Capture managed experimental env flags for reproducible Mark artifacts."""
    import os

    try:
        from autoptz.engine.runtime.experimental_flags import EXPERIMENTAL_FLAGS
    except Exception:  # noqa: BLE001
        return {}
    return {flag.env_key: os.environ.get(flag.env_key, flag.default) for flag in EXPERIMENTAL_FLAGS}


class BenchmarkRunner:
    """Ramp synthetic cameras until the min sustained fps drops below the floor."""

    def __init__(
        self,
        profile: BenchmarkProfile,
        *,
        floor_fps: float = 24.0,
        max_cameras: int = 16,
        dwell_s: float = 20.0,
        sample_fn: Callable[[int], list[float]],
        on_step: Callable[[StepResult], None] | None = None,
        quality_reader: Callable[[], dict[str, dict]] | None = None,
        source_mutation_reader: Callable[[], dict[str, object] | None] | None = None,
        fps_tolerance: float = 0.0,
    ) -> None:
        self._profile = profile
        self._floor = float(floor_fps)
        # Sub-fps jitter tolerance subtracted from the floor in the sustained check.
        # A real "N fps" source delivers a few tenths below nominal (rolling-fps
        # estimate + source pacing noise) while dropping ZERO frames; without this a
        # hard floor fails a perfectly healthy step on jitter alone.  The DROPS gate
        # stays strict, the reported floor is unchanged, and genuine under-delivery
        # (well below the floor) still fails.  Default 0.0 keeps the CLI/exact path.
        self._fps_tolerance = max(0.0, float(fps_tolerance))
        self._max_cameras = max(1, int(max_cameras))
        self._dwell_s = max(0.0, float(dwell_s))
        self._sample_fn = sample_fn
        self._on_step = on_step
        # Reads the most recent {cid: QualityMetrics dict} after each sample (the
        # headless sampler stashes it on ``last_quality``).  None -> no quality in
        # the step results (math-only tests / adopted GUI path before Slice 5).
        self._quality_reader = quality_reader
        self._source_mutation_reader = source_mutation_reader

    def run(self) -> BenchmarkResult:
        steps: list[StepResult] = []
        sustained_cameras = 0
        min_fps_at_sustained = 0.0

        for cameras in range(1, self._max_cameras + 1):
            per_camera = [float(f) for f in self._sample_fn(cameras)]
            if per_camera:
                min_fps = min(per_camera)
                mean_fps = sum(per_camera) / len(per_camera)
            else:
                min_fps = 0.0
                mean_fps = 0.0
            quality: dict[str, dict] = {}
            if self._quality_reader is not None:
                try:
                    quality = dict(self._quality_reader() or {})
                except Exception:  # noqa: BLE001 — quality is best-effort, never aborts the ramp
                    log.debug("benchmark quality_reader failed", exc_info=True)
            mutation = _source_mutation_snapshot(self._source_mutation_reader)
            (
                app_induced_drops,
                steady_state_app_induced_drops,
                source_mutation_events,
                source_mutation_allowed_drops,
                source_mutation_drop_grace_s,
            ) = _drop_accounting(quality, mutation)
            sustained = (
                min_fps >= self._floor - self._fps_tolerance and steady_state_app_induced_drops == 0
            )
            step = StepResult(
                cameras=cameras,
                min_fps=min_fps,
                mean_fps=mean_fps,
                per_camera_fps=per_camera,
                sustained=sustained,
                app_induced_drops=app_induced_drops,
                steady_state_app_induced_drops=steady_state_app_induced_drops,
                source_mutation_events=source_mutation_events,
                source_mutation_allowed_drops=source_mutation_allowed_drops,
                source_mutation_drop_grace_s=source_mutation_drop_grace_s,
                per_camera_quality=quality,
            )
            steps.append(step)
            if self._on_step is not None:
                try:
                    self._on_step(step)
                except Exception:  # noqa: BLE001 — a progress callback must not abort the run
                    log.debug("benchmark on_step callback failed", exc_info=True)
            if sustained:
                sustained_cameras = cameras
                min_fps_at_sustained = min_fps
            else:
                break  # this count failed the floor → ramp is done

        score = round(
            sustained_cameras * (min_fps_at_sustained / _NOMINAL_FPS) * self._profile.weight,
            2,
        )
        return BenchmarkResult(
            profile=self._profile.name,
            profile_description=self._profile.description,
            profile_features=dict(self._profile.features),
            experimental_flags=_experimental_flag_snapshot(),
            drop_policy=_DROP_POLICY,
            weight=self._profile.weight,
            floor_fps=self._floor,
            max_cameras=self._max_cameras,
            sustained_cameras=sustained_cameras,
            min_fps_at_sustained=min_fps_at_sustained,
            score=score,
            steps=steps,
        )


# ── real sampler: a headless Supervisor over N synthetic cameras ──────────────


def _default_fps_reader(client: Any, camera_id: str) -> float:
    """Read a camera's most recent telemetry fps from the client's model.

    ``CameraRecord.fps`` is a property over ``record.telemetry.fps`` (see
    ``autoptz/ui/list_models.py``), populated by ``CameraListModel.update_telemetry``
    on every ``push_telemetry``.  Adjust the attribute path here if that storage
    ever changes.
    """
    try:
        rec = client.cameraModel.get_record(camera_id)
    except Exception:  # noqa: BLE001
        return 0.0
    if rec is None:
        return 0.0
    return float(getattr(rec, "fps", 0.0) or 0.0)


def _add_synthetic_camera(
    client: Any,
    index: int,
    *,
    width: int = 0,
    height: int = 0,
    address: str = "anim",
    native_fps: float | None = None,
) -> str:
    """Register one self-paced synthetic camera on the client's model.

    Done directly via a ``CameraRecord`` (not ``client.addCamera``, which infers a
    USB source from the URI scheme).  The fps cap means the real worker paces the
    synthetic source so it never free-spins (~16000 fps would tear the shm
    triple-buffer).  A non-zero ``width``/``height`` (AutoPTZ Mark's resolution
    control) sizes the composed synthetic scene; 0 keeps the source default.

    ``address`` selects the synthetic content: the default ``"anim"`` draws moving
    synthetic people; a path to a video file (AutoPTZ Mark's bundled clip) makes
    the ``SyntheticAdapter`` loop that real clip instead (real decode, real people,
    no drawn overlay).

    ``native_fps`` (AutoPTZ Mark's selected clip cadence) sets the source fps so a
    24/30/60 clip paces at its true rate; ``None`` / non-positive falls back to 30,
    and any value is clamped to ``(0, 240]`` so a bad metadata value can't tear the
    triple-buffer.
    """
    from autoptz.config.models import CameraConfig, SourceConfig
    from autoptz.ui.list_models import CameraRecord

    camera_id = str(uuid.uuid4())
    name = f"AutoPTZ Mark {index + 1}"
    addr = (address or "anim").strip() or "anim"
    fps = native_fps if (native_fps and native_fps > 0) else 30.0
    fps = min(240.0, max(1.0, float(fps)))
    cfg = CameraConfig(
        id=camera_id,
        name=name,
        source=SourceConfig(
            type="synthetic",
            address=addr,
            fps=fps,
            width=int(width),
            height=int(height),
        ),
    )
    rec = CameraRecord(
        camera_id=camera_id,
        source_uri=f"synthetic://{addr}",
        display_name=name,
        camera_config=cfg,
    )
    client.cameraModel.add_camera(rec)
    return camera_id


def _default_supervisor_factory(client: Any, store: Any) -> Any:
    from autoptz.engine.supervisor import Supervisor

    return Supervisor(client, store=store)


class _SupervisorSampler:
    """Drives a headless Supervisor over N synthetic cameras and samples fps.

    The headful Mark window can **adopt** an already-built engine stack so a ramp
    reuses the ONE supervisor + pre-added cameras the Mark window already shows on
    the idle wall, instead of spinning up a second supervisor over a disjoint set
    of cameras (which doubled tiles and CPU).  Pass ``supervisor=`` (already
    primed) and ``cameras=`` (the pre-added camera ids) to adopt; ``adopted_started``
    says whether that supervisor's workers are already running so the first sample
    doesn't double-start it.
    """

    def __init__(
        self,
        profile: BenchmarkProfile,
        *,
        supervisor_factory: Callable[[Any, Any], Any] | None = None,
        client: Any | None = None,
        supervisor: Any | None = None,
        cameras: list[str] | None = None,
        adopted_started: bool = False,
        on_grow: Callable[[], str | None] | None = None,
    ) -> None:
        # When *client* is injected (the headful Mark window passes the SAME
        # EngineClient its CameraWall is bound to), the synthetic cameras land on
        # that client's model so tiles + frames actually render during the ramp.
        # The headless CLI passes None → we build a private client.
        self._profile = profile
        if client is None:
            from autoptz.ui.engine_client import EngineClient

            client = EngineClient()
        self._client = client
        # Adopt an existing supervisor (the Mark window's) when supplied so only
        # ONE stack ever runs; otherwise build a private one.
        self._adopted = supervisor is not None
        if supervisor is not None:
            self._sup = supervisor
        else:
            factory = supervisor_factory or _default_supervisor_factory
            store = getattr(client, "_store", None)
            self._sup = factory(self._client, store)
            self._sup.prime_features(dict(self._profile.features))
        # Pre-seed with adopted camera ids so _ensure_cameras never re-adds them.
        self._cameras: list[str] = list(cameras) if cameras else []
        self._started = bool(adopted_started)
        # Engine-reported tracking quality from the most recent headless sample,
        # keyed by camera id ({cid: QualityMetrics dict}).  The runner's
        # quality_reader reads this; the adopted/GUI path (Slice 5) wires it
        # separately.  Empty until the first headless sample finalizes.
        self.last_quality: dict[str, dict] = {}
        # Quality tap state (headless path only).  ``_install_quality_tap`` wraps
        # the client's ``push_telemetry`` ONCE, *before* the supervisor spawns its
        # workers (which capture the callback by reference at spawn time).  The
        # wrapper fans each TelemetryMsg into ``_active_accumulators`` while a
        # sample window is open (non-None), and is otherwise a transparent pass
        # -through.  ``_tap_installed`` guards against double-wrapping.
        self._tap_installed = False
        self._active_accumulators: dict[str, PerCameraQualityAccumulator] | None = None
        self._active_fps_hint = _NOMINAL_FPS
        # One-time warmup gate (adopted path): the detector model loads (~8s) during
        # the first dwell, so the first measured step would otherwise read ~0 fps →
        # "0 sustained" and the ramp stops immediately.  ``_warmed`` is flipped True
        # after the first sample waits for frames to flow + the model to finish
        # loading, so step 1 measures STEADY STATE, not model-load warmup.
        self._warmed = False
        # 3DMark-style progressive ramp (adopted path only): when the ramp steps to
        # N and the wall holds fewer, call ``on_grow`` to add the next camera ONE AT
        # A TIME on the SAME client + supervisor.  ``on_grow`` returns the new id and
        # is expected to register it on the model AND spawn its worker (the Mark
        # factory's ``add_next_camera``).  None → no growth (fixed pre-added set).
        self._on_grow = on_grow
        self._cancel_event: Any | None = None

    def set_cancel_event(self, event: Any | None) -> None:
        """Let the GUI controller interrupt warmup/dwell sleeps during teardown."""
        self._cancel_event = event

    def _cancelled(self) -> bool:
        event = self._cancel_event
        return bool(event is not None and event.is_set())

    def _wait(self, seconds: float) -> bool:
        """Wait up to *seconds*; return True when cancellation was requested."""
        seconds = max(0.0, float(seconds))
        event = self._cancel_event
        if event is not None:
            return bool(event.wait(seconds))
        time.sleep(seconds)
        return False

    @staticmethod
    def _drain_events() -> None:
        """Deliver queued telemetry signals from the worker thread.

        The real ``CameraWorker`` emits telemetry from its own thread, so
        ``EngineClient.push_telemetry`` marshals it onto the owning thread via a
        queued Qt signal.  Without draining the event loop the model never sees
        the update and every fps reads 0.  No-op when no ``QCoreApplication`` is
        running (the injected-fake-worker path emits synchronously).
        """
        from PySide6.QtCore import QCoreApplication

        app = QCoreApplication.instance()
        if app is not None:
            app.processEvents()

    def _ensure_cameras(self, n: int) -> None:
        while len(self._cameras) < n:
            self._cameras.append(_add_synthetic_camera(self._client, len(self._cameras)))

    def _dwell_observe(self, dwell_s: float) -> None:
        """Sleep the dwell while the external (GUI) pump drives the supervisor.

        Used by the adopted path: the Mark window's QTimer ticks the supervisor and
        delivers telemetry on the GUI thread, so the worker just waits before
        reading fps.  ``dwell_s == 0`` (tests) yields a single short settle so any
        already-queued telemetry has a chance to land.
        """
        self._wait(max(0.0, dwell_s) if dwell_s > 0.0 else 0.01)

    def _warmup(
        self,
        reader: Callable[[Any, str], float],
        *,
        min_fps: float = 10.0,
        timeout_s: float = 25.0,
        settle_s: float = 1.0,
        poll_s: float = 0.2,
    ) -> None:
        """Block until the current cameras are warmed up (frames + model loaded).

        The detector model loads (~8s) during the FIRST measured dwell, so without
        this gate step 1 reads ~0 fps and the ramp stops immediately.  This polls
        the current cameras' telemetry fps (the GUI pump updates the model on the
        GUI thread; this runs on a worker thread, so we just poll + sleep) until the
        slowest camera clears ``min_fps`` (frames flowing AND the model loaded) or a
        ~``timeout_s`` budget elapses, then settles ``settle_s`` so the rolling
        average reflects steady state before the first dwell measures it.  One-shot
        (``_warmed``); ``dwell_s == 0`` tests never enter the adopted-warmup path
        with real cameras so this stays fast.
        """
        if self._warmed:
            return
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline and not self._cancelled():
            cams = self._cameras
            if cams:
                fps = [reader(self._client, cid) for cid in cams]
                if fps and min(fps) >= min_fps:
                    break
            if self._wait(poll_s):
                break
        # Let the rolling fps average settle to steady state before the first dwell.
        self._wait(settle_s)
        self._warmed = True

    def sample(
        self,
        n: int,
        *,
        dwell_s: float,
        max_ticks: int,
        tick_sleep_s: float,
        fps_reader: Callable[[Any, str], float] | None = None,
    ) -> list[float]:
        reader = fps_reader or _default_fps_reader
        if self._adopted:
            # 3DMark-style progressive ramp: grow the wall to N one camera at a time
            # (the wall started at 1).  ``on_grow`` adds the next synthetic/NDI camera
            # on the SAME client + supervisor and spawns its worker.
            if self._on_grow is not None:
                while len(self._cameras) < n and not self._cancelled():
                    cid = self._on_grow()
                    if cid is None:
                        break  # hit the camera cap
                    self._cameras.append(cid)
            # The Mark window's GUI pump (33 ms QTimer) is the SOLE driver of the
            # adopted supervisor — ticking it here too would race two threads on one
            # supervisor.  Before the FIRST dwell, wait out the one-time engine warmup
            # (model load + frames flowing) so step 1 measures steady state, not the
            # ~8s model load.  Then observe the pre-added cameras over the dwell and
            # read the first ``n`` cameras' fps (the ramp is a measurement window).
            self._warmup(reader)
            self._dwell_observe(dwell_s)
            return [reader(self._client, cid) for cid in self._cameras[:n]]
        # Install the telemetry tap BEFORE the supervisor spawns workers — the
        # supervisor captures push_telemetry by reference at spawn time, so a tap
        # installed afterward would be bypassed.
        self._install_quality_tap()
        # Open the quality window now so every TelemetryMsg from this sample —
        # including any the worker emits at start() — folds into a per-camera
        # accumulator.  ``fps_hint`` seeds the rate math until the stream reports
        # its own fps.
        fps_hint = float(getattr(self._profile, "target_fps", 0.0) or 0.0) or _NOMINAL_FPS
        accumulators: dict[str, PerCameraQualityAccumulator] = {}
        self._active_fps_hint = fps_hint
        self._active_accumulators = accumulators
        try:
            had = len(self._cameras)
            self._ensure_cameras(n)
            if not self._started:
                # run_pump=False: we pump tick() ourselves so the dwell is bounded
                # and deterministic (no UI timer in a headless benchmark).
                self._sup.start(run_pump=False)
                self._started = True
            elif len(self._cameras) > had:
                # New cameras were appended this step.  The supervisor spawns
                # workers for the cameras present in the model at start() time, so
                # restart it to pick up the freshly added set.
                self._sup.stop()
                self._sup.start(run_pump=False)

            deadline = time.monotonic() + max(0.0, dwell_s)
            ticks = 0
            while (
                ticks < max_ticks
                and (ticks == 0 or time.monotonic() < deadline)
                and not self._cancelled()
            ):
                self._sup.tick()
                # Deliver any telemetry the worker thread queued onto this thread so
                # the model's fps reflects live frames (no-op for synchronous fakes).
                self._drain_events()
                ticks += 1
                if tick_sleep_s > 0.0:
                    if self._wait(tick_sleep_s):
                        break
            # Final drain so the last queued telemetry lands before we read fps.
            self._drain_events()
        finally:
            self._active_accumulators = None

        self.last_quality = {cid: acc.finalize().to_dict() for cid, acc in accumulators.items()}
        return [reader(self._client, cid) for cid in self._cameras[:n]]

    def _install_quality_tap(self) -> None:
        """Wrap the client's ``push_telemetry`` once so each pushed ``TelemetryMsg``
        also feeds the active per-camera quality accumulator.

        Installed before the supervisor spawns its workers (they capture the
        callback by reference at spawn time).  Transparent when no sample window is
        open (``_active_accumulators is None``): it simply forwards to the original.
        """
        if self._tap_installed:
            return
        original = getattr(self._client, "push_telemetry", None)
        if original is None:
            return

        def tapped(msg: Any) -> None:
            sink = self._active_accumulators
            if sink is not None:
                cid = getattr(msg, "camera_id", None)
                if cid is not None:
                    acc = sink.get(cid)
                    if acc is None:
                        acc = PerCameraQualityAccumulator(self._active_fps_hint)
                        sink[cid] = acc
                    acc.on_telemetry(msg)
            original(msg)

        self._client.push_telemetry = tapped  # type: ignore[method-assign]
        self._tap_installed = True

    def close(self) -> None:
        # Never tear down an ADOPTED supervisor — the Mark window owns its
        # lifecycle (closeEvent → engine.stop).  Only stop a supervisor we built.
        if self._adopted:
            return
        try:
            self._sup.stop()
        except Exception:  # noqa: BLE001
            log.debug("benchmark supervisor stop failed", exc_info=True)


def run_benchmark(
    *,
    profile: str = "full",
    floor_fps: float = 24.0,
    max_cameras: int = 16,
    dwell_s: float = 20.0,
    json_path: str | None = None,
    supervisor_factory: Callable[[Any, Any], Any] | None = None,
    fps_reader: Callable[[Any, str], float] | None = None,
    max_ticks: int = 2000,
    tick_sleep_s: float = 0.005,
) -> int:
    """Run AutoPTZ Mark and print/save the report.  Returns a process exit code.

    0 on success, 2 on an unknown profile.  Prints to stdout deliberately — this
    is the CLI face of the benchmark.
    """
    from autoptz.benchmark.profiles import get_profile

    try:
        prof = get_profile(profile)
    except ValueError as exc:
        print(str(exc))
        return 2

    # The EngineClient + workers are QObjects and worker telemetry is marshalled
    # back via queued Qt signals, so a QCoreApplication must exist (and own the
    # current thread) for that delivery — and for the sampler's processEvents()
    # drain to have an event loop to pump.  Under the UI this already exists; from
    # the headless CLI we create a minimal one.  Reuse any existing instance.
    from PySide6.QtCore import QCoreApplication

    if QCoreApplication.instance() is None:
        _ = QCoreApplication(sys.argv[:1])

    print(f"AutoPTZ Mark — profile {prof.name!r} ({prof.description})")
    print(f"  floor {floor_fps:.0f} fps - max {max_cameras} cameras - dwell {dwell_s:.0f}s")

    sampler = _SupervisorSampler(prof, supervisor_factory=supervisor_factory)

    def sample_fn(n: int) -> list[float]:
        return sampler.sample(
            n,
            dwell_s=dwell_s,
            max_ticks=max_ticks,
            tick_sleep_s=tick_sleep_s,
            fps_reader=fps_reader,
        )

    def quality_reader() -> dict[str, dict]:
        # Read whatever the sampler finalized for the most recent step.
        return dict(getattr(sampler, "last_quality", {}) or {})

    def on_step(step: StepResult) -> None:
        mark = "ok " if step.sustained else "STOP"
        print(
            f"  [{mark}] {step.cameras:2d} cam(s): min {step.min_fps:5.1f} fps "
            f"- mean {step.mean_fps:5.1f} fps"
        )

    try:
        runner = BenchmarkRunner(
            prof,
            floor_fps=floor_fps,
            max_cameras=max_cameras,
            dwell_s=dwell_s,
            sample_fn=sample_fn,
            on_step=on_step,
            quality_reader=quality_reader,
        )
        result = runner.run()
    finally:
        sampler.close()

    print(result.summary())

    if json_path:
        import json as _json
        from pathlib import Path

        Path(json_path).write_text(_json.dumps(result.to_dict(), indent=2))
        print(f"  wrote: {json_path}")
    return 0


# ── Follow-up (NOT implemented here): GUI "AutoPTZ Mark" window ───────────────
# A future task can add an Engine > Benchmark menu action that runs run_benchmark
# on a worker QThread (never the GUI thread) and renders BenchmarkResult.steps as
# a live ramp chart + the final score.  The runner is already UI-free and returns
# a fully serialisable BenchmarkResult, so the GUI only needs to call sample_fn /
# run() off-thread and marshal StepResult updates back via a queued signal
# (mirroring EngineClient._set_startup_progress).  Out of scope for this plan.
