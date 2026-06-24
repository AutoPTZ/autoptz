"""Tests for the consolidated feature-flag env resolvers (autoptz.engine.runtime.flags)."""

from __future__ import annotations

import cv2
import pytest

from autoptz.engine.runtime.flags import apply_opencv_thread_cap, env_unified_pose


@pytest.fixture(autouse=True)
def _restore_cv2_threads():
    """Save/restore the process-global OpenCV thread count around each test."""
    prev = cv2.getNumThreads()
    yield
    cv2.setNumThreads(prev)


class TestEnvUnifiedPose:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
    def test_truthy(self, monkeypatch, value):
        monkeypatch.setenv("AUTOPTZ_UNIFIED_POSE", value)
        assert env_unified_pose() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "maybe"])
    def test_falsy(self, monkeypatch, value):
        monkeypatch.setenv("AUTOPTZ_UNIFIED_POSE", value)
        assert env_unified_pose() is False

    def test_unset_is_false(self, monkeypatch):
        monkeypatch.delenv("AUTOPTZ_UNIFIED_POSE", raising=False)
        assert env_unified_pose() is False


class TestApplyOpenCVThreadCap:
    def test_cap_to_one_is_single_threaded(self):
        """Portable contract: capping to 1 makes OpenCV single-threaded.

        Honoured directly by TBB/OpenMP; reached via setNumThreads(0) on the
        macOS GCD backend (which ignores a positive count).
        """
        apply_opencv_thread_cap(1)
        assert cv2.getNumThreads() <= 1

    def test_reads_env_when_no_arg(self, monkeypatch):
        monkeypatch.setenv("AUTOPTZ_CV2_THREADS", "1")
        apply_opencv_thread_cap()
        assert cv2.getNumThreads() <= 1

    def test_noop_when_env_unset_and_no_arg(self, monkeypatch):
        monkeypatch.delenv("AUTOPTZ_CV2_THREADS", raising=False)
        before = cv2.getNumThreads()
        apply_opencv_thread_cap()  # no arg, no env → leave the current setting alone
        assert cv2.getNumThreads() == before

    def test_bad_env_is_ignored(self, monkeypatch):
        monkeypatch.setenv("AUTOPTZ_CV2_THREADS", "not-a-number")
        before = cv2.getNumThreads()
        apply_opencv_thread_cap()
        assert cv2.getNumThreads() == before

    def test_does_not_raise_on_multi_thread_target(self):
        """A multi-thread target must never raise, whatever the backend does with it."""
        apply_opencv_thread_cap(4)
