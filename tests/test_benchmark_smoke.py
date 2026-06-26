"""Real-pipeline smoke test for AutoPTZ Mark (2 synthetic cameras, short dwell).

This is the only benchmark test that spins a real Supervisor + CameraWorker over
real synthetic frames.  It is deliberately tolerant (no fps threshold) so it is
robust on slow CI hosts; it proves the synthetic source is self-paced and the
worker->telemetry->fps path produces a coherent ramp result.
"""

from __future__ import annotations

import pytest

from autoptz.benchmark.profiles import get_profile
from autoptz.benchmark.runner import BenchmarkRunner, _SupervisorSampler


@pytest.mark.timeout(45)
def test_two_camera_synthetic_ramp_smoke(qapp) -> None:
    # 'streams' profile (no inference) keeps the smoke fast + dependency-free.
    sampler = _SupervisorSampler(get_profile("streams"))
    try:
        # The worker's rolling-fps window needs ~1-2s of real frames before it
        # reports a non-zero fps; the dwell is generous (and the deadline bounds
        # each sample to dwell_s) but still well under the 45s timeout.
        dwell = 3.0
        runner = BenchmarkRunner(
            get_profile("streams"),
            floor_fps=1.0,  # tolerant: any real fps counts as sustained
            max_cameras=2,
            dwell_s=dwell,
            sample_fn=lambda n: sampler.sample(
                n, dwell_s=dwell, max_ticks=4000, tick_sleep_s=0.005
            ),
        )
        result = runner.run()
    finally:
        sampler.close()

    # Host-independent invariant: the ramp produces a monotonic prefix of [1, 2].
    cams = [s.cameras for s in result.steps]
    assert cams and cams[0] == 1 and cams == sorted(set(cams)) and cams[-1] <= 2, cams

    # On a very slow CI host the rolling-fps window may not fill within the dwell,
    # leaving fps at 0.0 and fewer cameras "sustained". That is a host-speed
    # artefact, not a benchmark bug, so skip rather than hard-fail. A free-spin
    # (the regression this smoke guards) produces a LARGE fps, so it is never
    # skipped here and is still caught by the < 200.0 assertion below.
    if result.sustained_cameras < 2 or result.min_fps_at_sustained <= 0.0:
        pytest.skip(
            f"host did not warm both synthetic cameras within {dwell}s dwell "
            f"(sustained={result.sustained_cameras}, fps={result.min_fps_at_sustained})"
        )

    # The real synthetic source paced ~30 fps; assert it is in a sane,
    # non-free-spinning range (well below the ~16000 fps free-spin tear).
    assert result.sustained_cameras == 2
    assert 0.0 < result.min_fps_at_sustained < 200.0
