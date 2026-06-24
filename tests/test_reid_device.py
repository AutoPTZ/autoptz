"""ReID device auto-selection.

OSNet ReID was pinned to ``device="cpu"`` — a major per-frame cost on Macs that
also stalled the inference thread via the GIL. It should prefer the platform GPU
(Apple ``mps`` / CUDA) and fall back to CPU, so the appearance pass gets off the
CPU where possible.
"""

from __future__ import annotations

from autoptz.engine.pipeline.reid import _best_reid_device, _pick_device


class TestPickDevice:
    def test_prefers_mps(self):
        assert _pick_device(has_mps=True, has_cuda=False) == "mps"

    def test_prefers_cuda_when_no_mps(self):
        assert _pick_device(has_mps=False, has_cuda=True) == "cuda"

    def test_mps_wins_over_cuda(self):
        assert _pick_device(has_mps=True, has_cuda=True) == "mps"

    def test_cpu_when_nothing(self):
        assert _pick_device(has_mps=False, has_cuda=False) == "cpu"


class TestBestReidDevice:
    def test_returns_a_valid_device(self):
        assert _best_reid_device() in {"mps", "cuda", "cpu"}

    def test_env_override_forces_device(self, monkeypatch):
        monkeypatch.setenv("AUTOPTZ_REID_DEVICE", "cpu")
        assert _best_reid_device() == "cpu"

    def test_env_override_ignores_garbage(self, monkeypatch):
        monkeypatch.setenv("AUTOPTZ_REID_DEVICE", "banana")
        assert _best_reid_device() in {"mps", "cuda", "cpu"}
