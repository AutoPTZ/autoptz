"""Unit tests for autoptz.benchmark.profiles."""

from __future__ import annotations

import pytest

from autoptz.benchmark.profiles import PROFILES, BenchmarkProfile, get_profile


def test_full_profile_enables_all_inference() -> None:
    prof = get_profile("full")
    assert isinstance(prof, BenchmarkProfile)
    assert prof.name == "full"
    assert prof.weight == 1.0
    # Every ML subsystem on (mirrors camera_worker._DEFAULT_FEATURES keys).
    assert prof.features == {
        "detection": True,
        "tracking": True,
        "face_recognition": True,
        "pose": True,
        "reid": True,
    }


def test_streams_profile_disables_all_inference() -> None:
    prof = get_profile("streams")
    assert prof.weight == 0.8
    assert set(prof.features) == {
        "detection",
        "tracking",
        "face_recognition",
        "pose",
        "reid",
    }
    assert not any(prof.features.values())  # capture + preview only


def test_profiles_registry_keys() -> None:
    assert set(PROFILES) == {"full", "streams"}


def test_get_profile_unknown_raises_with_valid_names() -> None:
    with pytest.raises(ValueError) as exc:
        get_profile("nope")
    assert "full" in str(exc.value) and "streams" in str(exc.value)
