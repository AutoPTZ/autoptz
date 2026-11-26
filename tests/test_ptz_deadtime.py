"""Dead-time (Smith-predictor) compensation for PTZ follow.

Field data (live NDI PTZ, 2026-11-26): a pan/tilt command shows up in the video
~0.5-0.7 s later (encoder + network + motor ramp). The controller only knew the
~50 ms local pipeline latency, so it kept driving after the subject was centred
and swung past — tilt rang with a ~2-2.5 s period.
"""

from __future__ import annotations

import math
import random

import pytest

from autoptz.engine.ptz.deadtime import DeadTimeModel

# ── model identification ─────────────────────────────────────────────────────


def _feed_plant(
    model: DeadTimeModel,
    *,
    delay: float,
    g_pan: float,
    g_tilt: float,
    seconds: float = 20.0,
    seed: int = 1,
    noise: float = 0.02,
) -> None:
    """Random step commands; camera motion = G * cmd(t - delay), observed at 10 Hz."""
    rng = random.Random(seed)
    hist: list[tuple[float, float, float]] = []  # (t, pan, tilt)
    t = 0.0
    next_change = 0.0
    pan = tilt = 0.0
    while t < seconds:
        if t >= next_change:
            pan = rng.choice([-0.5, -0.3, 0.0, 0.3, 0.5])
            tilt = rng.choice([-0.4, -0.2, 0.0, 0.2, 0.4])
            hist.append((t, pan, tilt))
            model.record_command(t, pan, tilt)
            next_change = t + rng.uniform(0.4, 1.2)
        # observe at 10 Hz
        if abs((t * 10) - round(t * 10)) < 1e-6:
            td = t - delay
            cp = ct = 0.0
            for ht, hp, htl in hist:
                if ht <= td:
                    cp, ct = hp, htl
            model.observe_motion(
                t, g_pan * cp + rng.gauss(0, noise), g_tilt * ct + rng.gauss(0, noise)
            )
        t = round(t + 0.05, 6)


def test_identifies_delay_and_gain_from_commands_and_flow() -> None:
    m = DeadTimeModel()
    _feed_plant(m, delay=0.6, g_pan=-2.5, g_tilt=3.0)
    est = m.estimate(now=20.0)
    assert est.confident
    assert est.delay_s == pytest.approx(0.6, abs=0.1)
    assert est.gain_pan == pytest.approx(-2.5, rel=0.15)
    assert est.gain_tilt == pytest.approx(3.0, rel=0.15)


def test_not_confident_without_excitation() -> None:
    m = DeadTimeModel()
    for i in range(100):
        t = i * 0.1
        m.record_command(t, 0.0, 0.0)
        m.observe_motion(t, 0.0, 0.0)
    assert not m.estimate(now=10.0).confident
    assert m.pending_motion(capture_t=9.9, now=10.0) == (0.0, 0.0)


def test_pending_motion_integrates_commands_not_yet_visible() -> None:
    m = DeadTimeModel()
    _feed_plant(m, delay=0.6, g_pan=-2.5, g_tilt=3.0)
    est = m.estimate(now=20.0)
    # Hold a known command for the next second.
    m.record_command(20.0, 0.25, 0.0)
    # Frame captured 50 ms ago: commands from (21 - 0.05 - L) .. 21 are unseen.
    px, py = m.pending_motion(capture_t=20.95, now=21.0)
    expected = est.gain_pan * 0.25 * (0.05 + est.delay_s)
    assert px == pytest.approx(expected, rel=0.05)
    assert py == pytest.approx(0.0, abs=1e-6)


def test_pending_motion_is_clamped() -> None:
    m = DeadTimeModel()
    _feed_plant(m, delay=0.6, g_pan=-8.0, g_tilt=8.0)
    m.estimate(now=20.0)
    m.record_command(20.0, 1.0, 1.0)
    px, py = m.pending_motion(capture_t=20.5, now=21.0)
    assert abs(px) <= m.MAX_PENDING + 1e-9 and abs(py) <= m.MAX_PENDING + 1e-9


def test_presets_blank_the_training_window() -> None:
    """Motion we didn't command (preset recall, home) must not teach the model."""
    m = DeadTimeModel()
    m.note_uncommanded_motion(0.0, hold_s=30.0)
    _feed_plant(m, delay=0.6, g_pan=-2.5, g_tilt=3.0)
    assert not m.estimate(now=20.0).confident


# ── closed loop: the controller with and without compensation ────────────────


def _simulate(compensate: bool, *, delay: float = 0.6, seconds: float = 12.0) -> list[float]:
    """Real PTZController tilting toward a static subject through a delayed camera.

    Returns the error seen in the (delayed) video over time.
    """
    from autoptz.config.models import PTZConfig
    from autoptz.engine.ptz.base import PTZBackend, PTZCaps
    from autoptz.engine.ptz.controller import PTZController

    class _Backend(PTZBackend):
        def __init__(self) -> None:
            super().__init__()
            self.caps = PTZCaps(continuous_pan_tilt=True, continuous_zoom=True)

        def move_velocity(self, pan, tilt, zoom=0.0):
            pass

        def stop(self):
            pass

        def goto_preset(self, idx):
            pass

        def save_preset(self, idx):
            pass

        def close(self):
            pass

    # error-space units/s per unit tilt command (measured live |G| 2.6-3.2). Negative:
    # before invert flags a positive command reduces a positive error.
    gain = -3.0
    dt = 1.0 / 30.0
    ctrl = PTZController(_Backend(), PTZConfig(safe_zone_enabled=False), rate_hz=30.0)
    model = DeadTimeModel()
    # Pre-train the model from a short excitation run (what live flow provides).
    _feed_plant(model, delay=delay, g_pan=gain, g_tilt=gain, seconds=15.0, seed=7)
    model.estimate(now=15.0)
    t0 = 15.0  # continue the model's clock

    ctrl.deadtime = model
    subject = 0.6  # subject starts 0.6 above centre (error space)
    cmd_hist: list[tuple[float, float]] = [(t0 - 5.0, 0.0)]
    seen: list[float] = []
    t = t0
    while t < t0 + seconds:
        # error visible in the video now = subject + camera motion up to (t - delay)
        visible = subject + cam_hist_at(cmd_hist, t - delay, gain, t0 - 5.0, dt)
        seen.append(visible)
        err_y = visible
        _pan, tilt, _zoom = ctrl.step(
            (0.0, err_y),
            (0.0, 0.0),
            0.4,
            track_active=True,
            t=t,
            capture_t=t if compensate else None,
        )
        cmd_hist.append((t, tilt))
        t = round(t + dt, 9)
    return seen


def cam_hist_at(hist, until, gain, start, dt):
    """Integral of gain*cmd over [start, until] (piecewise-constant commands)."""
    total = 0.0
    for i, (ts, c) in enumerate(hist):
        te = hist[i + 1][0] if i + 1 < len(hist) else math.inf
        a, b = max(ts, start), min(te, until)
        if b > a:
            total += gain * c * (b - a)
    return total


def _overshoot_and_swings(seen: list[float]) -> tuple[float, int]:
    start_sign = 1.0 if seen[0] > 0 else -1.0
    overshoot = max(0.0, max(-start_sign * e for e in seen))
    swings = sum(
        1
        for a, b in zip(seen, seen[1:], strict=False)
        if (a > 0.02) != (b > 0.02) and abs(b) > 0.02
    )
    return overshoot, swings


def test_compensation_removes_dead_time_overshoot() -> None:
    raw_over, _raw_swings = _overshoot_and_swings(_simulate(compensate=False))
    comp_over, comp_swings = _overshoot_and_swings(_simulate(compensate=True))
    # Without compensation the 0.6 s dead time makes it swing past centre.
    assert raw_over > 0.15
    # With it: at most a small overshoot and no ringing.
    assert comp_over < 0.5 * raw_over
    assert comp_over < 0.1
    assert comp_swings <= 1


def test_compensated_loop_still_reaches_the_subject() -> None:
    seen = _simulate(compensate=True)
    assert abs(seen[-1]) < 0.1


def test_recall_immediately_invalidates_active_fit() -> None:
    model = DeadTimeModel()
    _feed_plant(model, delay=0.6, g_pan=-2.5, g_tilt=-3.0)
    assert model.estimate(now=20.0).confident
    model.note_uncommanded_motion(20.1)
    assert not model.estimate(now=20.1).confident
    assert model.pending_motion(20.0, 20.1) == (0.0, 0.0)


def test_learns_minimum_tilt_speed_at_small_commands() -> None:
    model = DeadTimeModel()
    rng = random.Random(82)
    commands = [rng.choice([-0.15, -0.035, 0, 0.035, 0.15]) for _ in range(80)]
    for i in range(400):
        now = i / 20
        model.record_command(now, 0.0, commands[i // 5])
        visible = commands[(i - 4) // 5] if i >= 4 else 0.0
        speed = -0.7 * visible - (0.30 * math.copysign(1, visible) if visible else 0)
        model.observe_motion(now, 0.0, speed)
    fit = model.estimate(20.0)
    assert fit.confident
    assert fit.floor_tilt == pytest.approx(-0.30, abs=0.04)
    model.record_command(20.0, 0.0, 0.035)
    dx, dy = model.pending_motion(20.8, 20.8)
    assert dx == 0.0
    assert dy == pytest.approx(-0.325 * fit.delay_s, abs=0.01)


# ── sticky delay + persistence ───────────────────────────────────────────────
# Live (2026-11-26): the fit quality was nearly flat across lags (tilt R² 0.37,
# 0.37, 0.34, 0.31, 0.31, 0.38, 0.42 for 0-0.6 s), so the learned delay wandered
# 250-550 ms within one run; and every restart relearned from scratch — the
# first move after launch overshot a full frame (pan -0.52 → +0.68).


def test_delay_is_sticky_when_another_lag_is_only_marginally_better() -> None:
    from autoptz.engine.ptz.deadtime import _select_lag

    scores = {0.30: 0.43, 0.45: 0.40, 0.60: 0.41}
    assert _select_lag(scores, prev_lag=0.60) == 0.60
    assert _select_lag(scores, prev_lag=None) == 0.30


def test_delay_switches_when_another_lag_is_clearly_better() -> None:
    from autoptz.engine.ptz.deadtime import _select_lag

    assert _select_lag({0.30: 0.55, 0.60: 0.40}, prev_lag=0.60) == 0.30


def test_snapshot_seeds_a_new_model_so_compensation_works_at_once() -> None:
    m = DeadTimeModel()
    _feed_plant(m, delay=0.6, g_pan=-2.5, g_tilt=3.0)
    snap = m.snapshot(now=20.0)
    assert snap is not None

    fresh = DeadTimeModel()
    fresh.seed(snap)
    est = fresh.estimate(now=0.0)
    assert est.confident
    assert est.delay_s == pytest.approx(0.6, abs=0.1)
    fresh.record_command(0.0, 0.25, 0.0)
    px, _ = fresh.pending_motion(capture_t=0.95, now=1.0)
    assert px == pytest.approx(est.gain_pan * 0.25 * (0.05 + est.delay_s), rel=0.05)


def test_fresh_confident_fit_replaces_the_seed() -> None:
    fresh = DeadTimeModel()
    fresh.seed({"delay_s": 0.3, "gain_pan": -1.0, "gain_tilt": 1.0, "r2": 0.9})
    _feed_plant(fresh, delay=0.6, g_pan=-2.5, g_tilt=3.0)
    est = fresh.estimate(now=20.0)
    assert est.delay_s == pytest.approx(0.6, abs=0.1)
    assert est.gain_pan == pytest.approx(-2.5, rel=0.15)


def test_unconfident_model_has_no_snapshot() -> None:
    assert DeadTimeModel().snapshot(now=0.0) is None


def test_learned_model_round_trips_through_disk(tmp_path) -> None:
    from autoptz.engine.ptz.deadtime import load_learned, save_learned

    path = tmp_path / "ptz_deadtime.json"
    snap = {"delay_s": 0.55, "gain_pan": -2.4, "gain_tilt": 2.9, "r2": 0.6}
    save_learned("cam-a", snap, path=path)
    save_learned("cam-b", {**snap, "delay_s": 0.2}, path=path)
    assert load_learned("cam-a", path=path) == snap
    assert load_learned("cam-b", path=path)["delay_s"] == 0.2
    assert load_learned("missing", path=path) is None


def test_corrupt_store_is_ignored(tmp_path) -> None:
    from autoptz.engine.ptz.deadtime import load_learned

    path = tmp_path / "ptz_deadtime.json"
    path.write_text("{not json")
    assert load_learned("cam-a", path=path) is None
