"""Inventory well-formedness for the experimental-flags source of truth."""

from __future__ import annotations

import autoptz.engine.runtime.experimental_flags as ef_mod
from autoptz.engine.runtime.experimental_flags import (
    EXPERIMENTAL_FLAGS,
    ExperimentalFlag,
)

_KINDS = ("bool", "choice", "text", "path")


def test_all_env_keys_unique_and_prefixed() -> None:
    keys = [f.env_key for f in EXPERIMENTAL_FLAGS]
    assert len(keys) == len(set(keys)), "duplicate env_key"
    assert all(k.startswith("AUTOPTZ_") for k in keys)


def test_kinds_and_choices_consistent() -> None:
    for f in EXPERIMENTAL_FLAGS:
        assert isinstance(f, ExperimentalFlag)
        assert f.kind in _KINDS
        if f.kind == "bool":
            assert f.choices == ()
            assert f.default in ("0", "1")
        elif f.kind == "choice":
            assert len(f.choices) >= 2
            assert f.default in f.choices
        else:  # text / path — free-form, no choices
            assert f.choices == ()


def test_descriptions_and_sections_present() -> None:
    for f in EXPERIMENTAL_FLAGS:
        assert f.label.strip()
        assert f.description.strip()
        assert f.section.strip()


def test_expected_flags_inventoried() -> None:
    keys = {f.env_key for f in EXPERIMENTAL_FLAGS}
    assert keys == {
        # Experiments
        "AUTOPTZ_UNIFIED_POSE",
        "AUTOPTZ_ASYNC_APPEARANCE",
        "AUTOPTZ_PTZ_PUMP",
        "AUTOPTZ_PTZ_SERIAL_AUTOPROBE",
        "AUTOPTZ_TRUE_LATENCY_LEAD",
        "AUTOPTZ_MODEL_SERVER",
        # Devices & tuning
        "AUTOPTZ_REID_DEVICE",
        "AUTOPTZ_COREML_UNITS",
        "AUTOPTZ_NDI_COLOR_FORMAT",
        # Model overrides
        "AUTOPTZ_MODEL_PATH",
        "AUTOPTZ_POSE_MODEL_PATH",
        "AUTOPTZ_MODEL_URL",
        # Diagnostics
        "AUTOPTZ_MS_DIAG",
        "AUTOPTZ_SYNTH_DEBUG",
    }


def test_flags_grouped_into_sections() -> None:
    by_section: dict[str, set[str]] = {}
    for f in EXPERIMENTAL_FLAGS:
        by_section.setdefault(f.section, set()).add(f.env_key)
    assert by_section["Experiments"] == {
        "AUTOPTZ_UNIFIED_POSE",
        "AUTOPTZ_ASYNC_APPEARANCE",
        "AUTOPTZ_PTZ_PUMP",
        "AUTOPTZ_PTZ_SERIAL_AUTOPROBE",
        "AUTOPTZ_TRUE_LATENCY_LEAD",
        "AUTOPTZ_MODEL_SERVER",
    }
    assert by_section["Devices & tuning"] == {
        "AUTOPTZ_REID_DEVICE",
        "AUTOPTZ_COREML_UNITS",
        "AUTOPTZ_NDI_COLOR_FORMAT",
    }
    assert by_section["Model overrides"] == {
        "AUTOPTZ_MODEL_PATH",
        "AUTOPTZ_POSE_MODEL_PATH",
        "AUTOPTZ_MODEL_URL",
    }
    assert by_section["Diagnostics"] == {"AUTOPTZ_MS_DIAG", "AUTOPTZ_SYNTH_DEBUG"}


def test_model_path_flags_are_path_kind() -> None:
    for key in ("AUTOPTZ_MODEL_PATH", "AUTOPTZ_POSE_MODEL_PATH"):
        flag = next(f for f in EXPERIMENTAL_FLAGS if f.env_key == key)
        assert flag.kind == "path"
        assert flag.default == ""
    url = next(f for f in EXPERIMENTAL_FLAGS if f.env_key == "AUTOPTZ_MODEL_URL")
    assert url.kind == "text"
    assert url.default == ""


def test_dead_and_excluded_vars_absent_from_registry() -> None:
    keys = {f.env_key for f in EXPERIMENTAL_FLAGS}
    # Retired / dead — the alias is gone from the codebase entirely.
    assert "AUTOPTZ_PROCESS_PER_CAMERA" not in keys
    # Supervisor-managed hardware vars: a hardware-prefs path already writes these,
    # so surfacing them here would create two writers and silent drift.
    for hw in (
        "AUTOPTZ_FORCE_EP",
        "AUTOPTZ_PRECISION",
        "AUTOPTZ_ORT_INTRA_THREADS",
        "AUTOPTZ_CV2_THREADS",
    ):
        assert hw not in keys
    # Live but intentionally env-only (dev/CI controls, not GUI-worthy): they stay
    # out of the dialog and are documented in docs/flags.md.
    assert "AUTOPTZ_NO_MODEL_EXPORT" not in keys
    assert "AUTOPTZ_SKIP_CAMERA_PREFLIGHT" not in keys


def test_no_tracking_default_fields_export() -> None:
    # The dead "New-camera tracking defaults" section is gone: nothing consumed
    # those keys for a new camera; group_framing now lives on the per-camera panel.
    assert not hasattr(ef_mod, "TRACKING_DEFAULT_FIELDS")


def test_model_server_flag_registered() -> None:
    flag = next(f for f in EXPERIMENTAL_FLAGS if f.env_key == "AUTOPTZ_MODEL_SERVER")
    assert flag.kind == "bool"
    assert flag.default == "0"  # off by default — still experimental
    assert flag.restart_required is True


def test_ndi_color_format_uses_real_source_values() -> None:
    ndi = next(f for f in EXPERIMENTAL_FLAGS if f.env_key == "AUTOPTZ_NDI_COLOR_FORMAT")
    assert ndi.choices == ("fastest", "bgra")
    assert ndi.default == "fastest"
