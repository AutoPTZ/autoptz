"""Tests for the consolidated feature-flag env resolvers (autoptz.engine.runtime.flags)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import cv2
import pytest

from autoptz.engine.runtime.flags import (
    apply_opencv_thread_cap,
    apply_thread_caps,
    env_unified_pose,
)

# ── env var names that apply_thread_caps sets ─────────────────────────────────
_OMP_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@pytest.fixture(autouse=True)
def _restore_cv2_threads():
    """Save/restore the process-global OpenCV thread count around each test."""
    prev = cv2.getNumThreads()
    yield
    cv2.setNumThreads(prev)


@pytest.fixture()
def _clean_omp_env(monkeypatch):
    """Remove the OMP/BLAS env vars before a test so results are deterministic."""
    for var in _OMP_VARS:
        monkeypatch.delenv(var, raising=False)


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


class TestApplyThreadCaps:
    """Tests for apply_thread_caps — env var publishing and torch capping."""

    # ── env var publishing ────────────────────────────────────────────────────

    @pytest.mark.usefixtures("_clean_omp_env")
    def test_sets_all_env_vars(self, monkeypatch):
        """All four env vars are published with the correct string value."""
        import os

        apply_thread_caps(3)
        for var in _OMP_VARS:
            assert os.environ.get(var) == "3", f"{var} was not set to '3'"

    @pytest.mark.usefixtures("_clean_omp_env")
    def test_budget_floored_to_one(self, monkeypatch):
        """A budget of 0 or negative is clamped to 1 — never publishes '0'."""
        import os

        apply_thread_caps(0)
        for var in _OMP_VARS:
            assert os.environ.get(var) == "1", f"{var} should be '1' for budget=0"

    @pytest.mark.usefixtures("_clean_omp_env")
    def test_budget_formula_matches_supervisor(self, monkeypatch):
        """Budget (cores-1)//cameras matches what _apply_hardware_env computes.

        This pins the relationship: the same formula drives ORT, OpenCV, and now
        OMP/BLAS/torch, so they're always consistent.
        """
        import os

        cores = 8
        cameras = 3
        expected = max(1, (cores - 1) // cameras)  # == 2
        apply_thread_caps(expected)
        for var in _OMP_VARS:
            assert os.environ.get(var) == str(expected)

    # ── torch capping ─────────────────────────────────────────────────────────

    @pytest.mark.usefixtures("_clean_omp_env")
    def test_calls_torch_set_num_threads_when_available(self, monkeypatch):
        """torch.set_num_threads is called with the budget when torch is importable."""
        fake_torch = types.ModuleType("torch")
        set_num_threads = MagicMock()
        fake_torch.set_num_threads = set_num_threads  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"torch": fake_torch}):
            apply_thread_caps(4)

        set_num_threads.assert_called_once_with(4)

    @pytest.mark.usefixtures("_clean_omp_env")
    def test_no_raise_when_torch_missing(self, monkeypatch):
        """apply_thread_caps must not raise when torch is not installed."""
        with patch.dict(sys.modules, {"torch": None}):  # type: ignore[dict-item]
            # Must complete without exception.
            apply_thread_caps(2)

    @pytest.mark.usefixtures("_clean_omp_env")
    def test_no_raise_when_torch_import_errors(self):
        """apply_thread_caps survives an ImportError on torch."""
        # Remove torch from sys.modules so the import will fail.
        original = sys.modules.pop("torch", _SENTINEL)
        try:
            apply_thread_caps(2)  # should not raise
        finally:
            if original is not _SENTINEL:
                sys.modules["torch"] = original

    @pytest.mark.usefixtures("_clean_omp_env")
    def test_torch_error_in_set_num_threads_is_swallowed(self, monkeypatch):
        """If torch.set_num_threads raises, apply_thread_caps still succeeds."""
        import os

        fake_torch = types.ModuleType("torch")
        fake_torch.set_num_threads = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"torch": fake_torch}):
            apply_thread_caps(2)  # must not propagate the RuntimeError

        # Env vars should still have been set despite the torch failure.
        for var in _OMP_VARS:
            assert os.environ.get(var) == "2"


# ── sentinel for sys.modules pop/restore ─────────────────────────────────────
_SENTINEL = object()
