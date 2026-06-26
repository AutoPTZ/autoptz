"""Inventory well-formedness for the experimental-flags source of truth."""

from __future__ import annotations

from autoptz.engine.runtime.experimental_flags import (
    EXPERIMENTAL_FLAGS,
    TRACKING_DEFAULT_FIELDS,
    ExperimentalFlag,
)


def test_all_env_keys_unique_and_prefixed() -> None:
    keys = [f.env_key for f in EXPERIMENTAL_FLAGS]
    assert len(keys) == len(set(keys)), "duplicate env_key"
    assert all(k.startswith("AUTOPTZ_") for k in keys)


def test_kinds_and_choices_consistent() -> None:
    for f in EXPERIMENTAL_FLAGS:
        assert isinstance(f, ExperimentalFlag)
        assert f.kind in ("bool", "choice")
        if f.kind == "bool":
            assert f.choices == ()
            assert f.default in ("0", "1")
        else:
            assert len(f.choices) >= 2
            assert f.default in f.choices


def test_descriptions_present() -> None:
    for f in EXPERIMENTAL_FLAGS:
        assert f.label.strip()
        assert f.description.strip()


def test_expected_flags_inventoried() -> None:
    keys = {f.env_key for f in EXPERIMENTAL_FLAGS}
    assert keys == {
        "AUTOPTZ_UNIFIED_POSE",
        "AUTOPTZ_ASYNC_APPEARANCE",
        "AUTOPTZ_PTZ_PUMP",
        "AUTOPTZ_PROCESS_PER_CAMERA",
        "AUTOPTZ_REID_DEVICE",
        "AUTOPTZ_COREML_UNITS",
        "AUTOPTZ_NDI_COLOR_FORMAT",
        "AUTOPTZ_PTZ_SERIAL_AUTOPROBE",
    }


def test_ndi_color_format_uses_real_source_values() -> None:
    ndi = next(f for f in EXPERIMENTAL_FLAGS if f.env_key == "AUTOPTZ_NDI_COLOR_FORMAT")
    assert ndi.choices == ("fastest", "bgra")
    assert ndi.default == "fastest"


def test_tracking_default_fields() -> None:
    names = [t[0] for t in TRACKING_DEFAULT_FIELDS]
    assert names == ["unified_pose", "use_target_associator", "stage_spread", "group_framing"]
    defaults = {t[0]: t[3] for t in TRACKING_DEFAULT_FIELDS}
    # Mirror config/models.py TrackingConfig defaults exactly.
    assert defaults == {
        "unified_pose": False,
        "use_target_associator": False,
        "stage_spread": True,
        "group_framing": False,
    }
