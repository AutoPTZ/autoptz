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


@pytest.mark.timeout(30)
def test_two_camera_synthetic_ramp_smoke(qapp) -> None:
    # 'streams' profile (no inference) keeps the smoke fast + dependency-free.
    sampler = _SupervisorSampler(get_profile("streams"))
    try:
        # The worker's rolling-fps window needs ~1s of real frames before it
        # reports a non-zero fps, so the dwell is generous but still well under
        # the 30s timeout (2 cameras x ~2s).
        dwell = 2.0
        runner = BenchmarkRunner(
            get_profile("streams"),
            floor_fps=1.0,  # tolerant: any real fps counts as sustained
            max_cameras=2,
            dwell_s=dwell,
            sample_fn=lambda n: sampler.sample(
                n, dwell_s=dwell, max_ticks=2000, tick_sleep_s=0.005
            ),
        )
        result = runner.run()
    finally:
        sampler.close()

    # Two cameras were measured and the result is coherent.
    assert [s.cameras for s in result.steps] == [1, 2]
    assert result.sustained_cameras == 2
    # The real synthetic source paced ~30 fps; assert it is in a sane,
    # non-free-spinning range (well below the 16000 fps free-spin tear).
    assert 0.0 < result.min_fps_at_sustained < 200.0
