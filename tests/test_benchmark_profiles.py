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


def test_simple_follow_profile_is_production_baseline() -> None:
    prof = get_profile("simple_follow")
    assert prof.weight == 1.0
    assert prof.features == {
        "detection": True,
        "tracking": True,
        "face_recognition": False,
        "pose": False,
        "reid": False,
    }


def test_pose_follow_isolates_pose_cost() -> None:
    prof = get_profile("pose_follow")
    assert prof.weight == 1.0
    assert prof.features == {
        "detection": True,
        "tracking": True,
        "face_recognition": False,
        "pose": True,
        "reid": False,
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
    assert set(PROFILES) == {"simple_follow", "pose_follow", "full", "streams"}


def test_get_profile_unknown_raises_with_valid_names() -> None:
    with pytest.raises(ValueError) as exc:
        get_profile("nope")
    msg = str(exc.value)
    assert "simple_follow" in msg and "pose_follow" in msg
    assert "full" in msg and "streams" in msg
