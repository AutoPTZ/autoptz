"""Ego-motion estimator + control-loop compensation tests.

Synthetic frames only; no hardware or models needed.  These lock in the fix for
hunting when the subject and the camera move at the same time: the camera's own
image motion must be measured and removed from the aim-error velocity.
"""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from autoptz.engine.pipeline.egomotion import EgoMotionEstimator


def _textured(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    """A textured BGR frame so optical flow has features to track."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0)


# ── estimator ────────────────────────────────────────────────────────────────


def test_first_frame_has_no_estimate() -> None:
    est = EgoMotionEstimator(smoothing=0.0)
    ego = est.estimate(_textured(), now=0.0)
    assert ego.source == "none"
    assert ego.vx == 0.0 and ego.vy == 0.0


def test_pan_right_gives_negative_x_velocity() -> None:
    """Camera pans right → content shifts left → error-space vx < 0."""
    est = EgoMotionEstimator(smoothing=0.0)
    base = _textured()
    est.estimate(base, now=0.0)
    # content shifted left 16px (proc scale 0.5 → 8px; /(320/2)=0.05; /0.1s=0.5)
    ego = est.estimate(np.roll(base, -16, axis=1), now=0.1, ptz_cmd=(0.5, 0.0, 0.0))
    assert ego.source == "flow"
    assert ego.vx == pytest.approx(-0.5, abs=0.05)
    assert ego.vy == pytest.approx(0.0, abs=0.05)


def test_pan_left_gives_positive_x_velocity() -> None:
    est = EgoMotionEstimator(smoothing=0.0)
    base = _textured()
    est.estimate(base, now=0.0)
    ego = est.estimate(np.roll(base, 16, axis=1), now=0.1, ptz_cmd=(-0.5, 0.0, 0.0))
    assert ego.source == "flow"
    assert ego.vx == pytest.approx(0.5, abs=0.05)


def test_tilt_content_down_gives_negative_y_velocity() -> None:
    """Content shifts down → image-y up is negated → error-space vy < 0."""
    est = EgoMotionEstimator(smoothing=0.0)
    base = _textured()
    est.estimate(base, now=0.0)
    ego = est.estimate(np.roll(base, 12, axis=0), now=0.1, ptz_cmd=(0.0, 0.5, 0.0))
    assert ego.source == "flow"
    # 12px → proc 6px → 6/(240/2)=0.05 → /0.1 = 0.5, negated
    assert ego.vy == pytest.approx(-0.5, abs=0.06)
    assert ego.vx == pytest.approx(0.0, abs=0.05)


def test_subject_motion_does_not_bias_estimate() -> None:
    """A subject moving inside a masked box must not look like camera motion."""
    est = EgoMotionEstimator(smoothing=0.0)
    base = _textured()
    est.estimate(base, now=0.0)
    box = (300.0, 150.0, 380.0, 400.0)
    moved = base.copy()
    moved[150:400, 280:360] = base[150:400, 300:380]  # shift the "subject" patch
    ego = est.estimate(moved, now=0.1, boxes=[box])
    assert abs(ego.vx) < 0.05 and abs(ego.vy) < 0.05


def test_command_fallback_after_learning() -> None:
    """Once the command→image gain is learned, a featureless frame still yields
    a command-based estimate instead of nothing."""
    est = EgoMotionEstimator(smoothing=0.0)
    base = _textured()
    # Learn: repeated 16px right-pans paired with a +0.5 pan command.
    prev = base
    t = 0.0
    est.estimate(prev, now=t)
    for i in range(1, 12):
        t = i * 0.1
        cur = np.roll(base, -16 * i, axis=1)
        ego = est.estimate(cur, now=t, ptz_cmd=(0.5, 0.0, 0.0))
        assert ego.source == "flow"
    # Low-texture scene: features are detected on the *previous* frame, so both
    # frames must be featureless for flow to give up and the command map to win.
    uniform = np.full_like(base, 127)
    est.estimate(uniform, now=t + 0.1, ptz_cmd=(0.5, 0.0, 0.0))  # prev ← uniform
    ego = est.estimate(uniform, now=t + 0.2, ptz_cmd=(0.5, 0.0, 0.0))
    assert ego.source == "command"
    # Learned gain is negative (ego_vx<0 for pan>0), so the command estimate must
    # be a meaningful negative velocity in the same direction the flow measured.
    assert ego.vx < -0.1


def test_resolution_change_is_safe() -> None:
    est = EgoMotionEstimator(smoothing=0.0)
    est.estimate(_textured(480, 640), now=0.0)
    ego = est.estimate(_textured(720, 1280), now=0.1)  # different shape
    assert ego.source == "none"  # rolls forward without crashing


# ── control-loop compensation (regression for the phantom-velocity oscillation) ─


def _aim_vel_fn():
    # Bind the real method onto a lightweight stand-in so we don't build a full
    # worker (threads/config) just to exercise the velocity math.
    from autoptz.engine.camera_worker import CameraWorker

    return CameraWorker._estimate_aim_velocity


def test_ego_comp_cancels_phantom_velocity_when_only_camera_moves() -> None:
    est_vel = _aim_vel_fn()
    ns = SimpleNamespace(
        _prev_aim_err=None,
        _prev_aim_t=0.0,
        _aim_vel=(0.0, 0.0),
        _ego_vel=(0.0, 0.0),
        _ego_fresh=True,  # ego freshly measured this tick → trusted/subtracted
    )
    est_vel(ns, (0.0, 0.0), 0.0)  # establish previous error
    # Subject stationary in the world; camera pans → error drifts at -0.5/s, all
    # of which the ego estimate accounts for.
    ns._ego_vel = (-0.5, 0.0)
    vx, vy = est_vel(ns, (-0.05, 0.0), 0.1)
    assert vx == pytest.approx(0.0, abs=1e-6)
    assert vy == pytest.approx(0.0, abs=1e-6)


def test_without_ego_comp_phantom_velocity_appears() -> None:
    est_vel = _aim_vel_fn()
    ns = SimpleNamespace(
        _prev_aim_err=None,
        _prev_aim_t=0.0,
        _aim_vel=(0.0, 0.0),
        _ego_vel=(0.0, 0.0),
        _ego_fresh=False,  # ego comp off / no fresh measurement → not subtracted
    )
    est_vel(ns, (0.0, 0.0), 0.0)
    # Same drift but ego comp off (ego_vel stays zero) → a non-zero phantom
    # velocity is fed forward (the legacy hunting behaviour).
    vx, _vy = est_vel(ns, (-0.05, 0.0), 0.1)
    assert vx < -0.1
