"""Assertion group 2: velocity-estimate RMSE vs analytic ground truth."""

from __future__ import annotations

import math

from autoptz.engine.pipeline.track import Tracker
from tests.synthetic_tracking import (
    FRAME,
    constant_velocity_centres,
    detections_for_centres,
    make_mock_impl,
    sinusoid_centres,
    tracker_rows_for_centres,
)

FPS = 30.0


def _rmse(errors: list[float]) -> float:
    return math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else 0.0


def _run_velocity(centres: list[tuple[float, float]]) -> list[tuple[float, float]]:
    # One box per frame at the scripted centre; same track id (=1) throughout, so
    # update() returns exactly one Track per frame whose velocity is the wrapper's
    # consecutive-centre delta estimate.
    rows = [tracker_rows_for_centres([c], track_id=1)[0] for c in centres]
    tracker = Tracker(_impl=make_mock_impl(rows), min_hits=1)
    out: list[tuple[float, float]] = []
    for cx, cy in centres:
        dets = detections_for_centres([(cx, cy)])[0]
        tracks = tracker.update(dets, FRAME, fps=FPS)
        out.append(tracks[0].velocity)
    return out


def test_constant_velocity_rmse_is_tiny() -> None:
    """vx estimate (px/frame) must equal the true vx on every frame after the first."""
    vx_true = 5.0
    centres = constant_velocity_centres(x0=200.0, vx=vx_true, frames=20)
    vels = _run_velocity(centres)
    # Frame 0 velocity is (0,0) by design — exclude it from the RMSE.
    vx_err = [vx - vx_true for (vx, _vy) in vels[1:]]
    vy_err = [vy - 0.0 for (_vx, vy) in vels[1:]]
    assert _rmse(vx_err) < 1e-3
    assert _rmse(vy_err) < 1e-6


def test_sinusoid_velocity_tracks_derivative() -> None:
    """Finite-difference vx must approximate amp*omega*cos(omega*t) within a
    bounded RMSE (the 1-frame backward difference lags the analytic derivative)."""
    amp, omega = 60.0, 0.25  # rad per frame index (dt=1)
    n = 40
    centres = sinusoid_centres(cx=320.0, amp=amp, omega=omega, frames=n, dt=1.0)
    vels = _run_velocity(centres)
    errs: list[float] = []
    for i in range(2, n):  # skip frame 0 (zero) and frame 1 (warm-up)
        analytic = amp * omega * math.cos(omega * i)
        errs.append(vels[i][0] - analytic)
    # Backward-difference of a slow sinusoid: RMSE bounded by ~0.5*amp*omega^2.
    bound = 0.5 * amp * omega * omega + 1e-6
    assert _rmse(errs) < bound
