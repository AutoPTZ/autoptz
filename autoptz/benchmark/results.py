"""AutoPTZ Mark — machine info capture + result persistence (pure, no Qt)."""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from autoptz import __version__
from autoptz.benchmark.runner import BenchmarkResult


def _ram_gb() -> float | None:
    try:
        import psutil  # noqa: PLC0415

        return round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:  # noqa: BLE001 — psutil optional; degrade gracefully
        return None


def _execution_providers() -> list[str]:
    try:
        from autoptz.engine.runtime.diagnostics import inference_status  # noqa: PLC0415

        detail = inference_status().get("detail", "")
    except Exception:  # noqa: BLE001
        return []
    # detail looks like "ONNX Runtime · CoreML, CPU"
    if "·" in detail:
        detail = detail.split("·", 1)[1]
    return [p.strip() for p in detail.split(",") if p.strip()]


def collect_machine_info() -> dict[str, object]:
    import os as _os  # noqa: PLC0415

    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": _os.cpu_count(),
        "ram_gb": _ram_gb(),
        "execution_providers": _execution_providers(),
        "app_version": __version__,
    }


@dataclass(frozen=True)
class MarkResultBundle:
    created_at: str
    app_version: str
    machine: dict[str, object]
    results: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "created_at": self.created_at,
            "app_version": self.app_version,
            "machine": self.machine,
            "results": self.results,
        }


def _build_bundle(results: list[BenchmarkResult]) -> MarkResultBundle:
    """Assemble the serializable bundle (machine info + results) for a save."""
    return MarkResultBundle(
        created_at=datetime.now(UTC).isoformat(),
        app_version=__version__,
        machine=collect_machine_info(),
        results=[r.to_dict() for r in results],
    )


def _mirror_to_store(store: object | None, bundle: MarkResultBundle) -> None:
    """Mirror the bundle under ``last_mark_result`` when a store is supplied."""
    if store is None:
        return
    set_setting = getattr(store, "set_setting", None)
    if callable(set_setting):
        set_setting("last_mark_result", bundle.to_dict())


def save_mark_result(
    results: list[BenchmarkResult],
    *,
    config_dir: Path | None = None,
    store: object | None = None,
) -> tuple[Path, MarkResultBundle]:
    from autoptz.config.store import default_config_dir  # noqa: PLC0415

    base = config_dir if config_dir is not None else default_config_dir()
    out_dir = Path(base) / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    bundle = _build_bundle(results)
    path = out_dir / f"autoptz-mark-{stamp}.json"
    path.write_text(json.dumps(bundle.to_dict(), indent=2))
    _mirror_to_store(store, bundle)
    return path, bundle


def save_mark_result_to_path(
    results: list[BenchmarkResult],
    path: Path,
    *,
    store: object | None = None,
) -> tuple[Path, MarkResultBundle]:
    """Write the result bundle to an EXPLICIT path (the Save-As completion dialog).

    Unlike :func:`save_mark_result` (which auto-names a file under the benchmarks
    dir), this honors the user-chosen ``path`` verbatim — creating parent dirs —
    and still mirrors the bundle under ``last_mark_result`` when a store is given.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = _build_bundle(results)
    path.write_text(json.dumps(bundle.to_dict(), indent=2))
    _mirror_to_store(store, bundle)
    return path, bundle


# ── CSV export (one row per ramp step × camera) ───────────────────────────────

# The flat spreadsheet schema.  Header order is load-bearing: external tooling
# reads these columns by name, and the tests pin the row verbatim.
_CSV_HEADER: list[str] = [
    "created_at",
    "app_version",
    "profile",
    "scene_clip_id",
    "step_cameras",
    "camera_idx",
    "per_camera_fps",
    "sustained",
    "min_fps",
    "mean_fps",
    "time_to_first_acquire_s",
    "total_lost_duration_s",
    "longest_lost_duration_s",
    "lost_event_count",
    "reacquire_count",
    "id_switch_count",
    "target_hold_pct",
    "mean_target_confidence",
    "dropped_frames",
    "app_induced_drops",
    "frames_delivered",
    "frames_dropped_est",
    "delivered_fps",
    "source_fps",
    "duplicate_frames",
    "stale_frames",
    "ndi_queue_depth",
    "ndi_queue_audio",
    "ndi_queue_metadata",
    "ndi_total_video_frames",
    "ndi_dropped_video_frames",
    "ndi_total_audio_frames",
    "ndi_dropped_audio_frames",
    "ndi_total_metadata_frames",
    "ndi_dropped_metadata_frames",
    "ndi_connections",
    "ndi_fourcc",
    "ndi_conversion_ms",
    "step_app_induced_drops",
    "steady_state_app_induced_drops",
    "source_mutation_events",
    "source_mutation_allowed_drops",
    "source_mutation_drop_grace_s",
    "drop_policy",
    "gt_miss_rate",
    "gt_id_switch_rate",
    "gt_motp",
]

# QualityMetrics-dict keys mapped to their CSV column (the GT columns are pulled
# separately — they live on the result/bundle, not the per-camera quality dict).
_QUALITY_COLUMNS: list[str] = [
    "time_to_first_acquire_s",
    "total_lost_duration_s",
    "longest_lost_duration_s",
    "lost_event_count",
    "reacquire_count",
    "id_switch_count",
    "target_hold_pct",
    "mean_target_confidence",
    "dropped_frames",
    "app_induced_drops",
    "frames_delivered",
    "frames_dropped_est",
    "delivered_fps",
    "source_fps",
    "duplicate_frames",
    "stale_frames",
    "ndi_queue_depth",
    "ndi_queue_audio",
    "ndi_queue_metadata",
    "ndi_total_video_frames",
    "ndi_dropped_video_frames",
    "ndi_total_audio_frames",
    "ndi_dropped_audio_frames",
    "ndi_total_metadata_frames",
    "ndi_dropped_metadata_frames",
    "ndi_connections",
    "ndi_fourcc",
    "ndi_conversion_ms",
]


def _cell(value: object) -> object:
    """``None`` → empty cell; everything else is written as-is (csv stringifies)."""
    return "" if value is None else value


def _result_scene_clip_id(result: object) -> str:
    """The scene clip id for a result, if the result/bundle carries one, else ``""``.

    The math-only ``BenchmarkResult`` has no such field yet, so this stays a
    forward-compatible best-effort read (attribute first, then a dict-ish lookup).
    """
    clip = getattr(result, "scene_clip_id", None)
    if clip is None and isinstance(result, dict):
        clip = result.get("scene_clip_id")
    return "" if clip is None else str(clip)


def _result_ground_truth(result: object) -> dict | None:
    """The ground-truth summary dict for a result, if present, else ``None``.

    Populated only when a GT run produced one (env ``AUTOPTZ_MARK_GT``); absent on
    a plain ramp, in which case the GT columns stay blank.
    """
    gt = getattr(result, "ground_truth", None)
    if gt is None and isinstance(result, dict):
        gt = result.get("ground_truth")
    return gt if isinstance(gt, dict) else None


def save_mark_result_csv(
    results: list[BenchmarkResult],
    path: Path,
    *,
    store: object | None = None,  # noqa: ARG001 — accepted for signature parity
) -> Path:
    """Write a flat CSV of every ramp step × camera, including quality metrics.

    One row per (step, camera): the camera's observed fps plus the engine-reported
    tracking quality (from ``step.per_camera_quality``) and, when a ground-truth run
    produced one, the CLEAR-MOT accuracy columns.  ``None`` values render as empty
    cells.  ``created_at`` / ``app_version`` are reused from the shared bundle
    metadata so a CSV and its JSON sibling agree.

    The ``StepResult`` exposes no camera-id list, so the index in
    ``per_camera_fps`` is the ``camera_idx`` and quality is matched by walking
    ``per_camera_quality`` in insertion order (camera id is not emitted).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = _build_bundle(results)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_HEADER)
        for result in results:
            created_at = bundle.created_at
            app_version = bundle.app_version
            profile = getattr(result, "profile", "")
            scene_clip_id = _result_scene_clip_id(result)
            gt = _result_ground_truth(result)
            gt_miss = _cell(gt.get("miss_rate")) if gt else ""
            gt_idsw = _cell(gt.get("id_switch_rate")) if gt else ""
            gt_motp = _cell(gt.get("motp")) if gt else ""

            for step in getattr(result, "steps", None) or ():
                per_camera_fps = list(getattr(step, "per_camera_fps", None) or [])
                quality = getattr(step, "per_camera_quality", None) or {}
                # Index → quality dict, by per_camera_quality insertion order.
                quality_by_idx = list(quality.values())
                for camera_idx, fps in enumerate(per_camera_fps):
                    q = quality_by_idx[camera_idx] if camera_idx < len(quality_by_idx) else {}
                    row = [
                        created_at,
                        app_version,
                        profile,
                        scene_clip_id,
                        step.cameras,
                        camera_idx,
                        _cell(fps),
                        step.sustained,
                        _cell(step.min_fps),
                        _cell(step.mean_fps),
                        *[_cell(q.get(col)) for col in _QUALITY_COLUMNS],
                        _cell(getattr(step, "app_induced_drops", 0)),
                        _cell(getattr(step, "steady_state_app_induced_drops", 0)),
                        _cell(getattr(step, "source_mutation_events", 0)),
                        _cell(getattr(step, "source_mutation_allowed_drops", 0)),
                        _cell(getattr(step, "source_mutation_drop_grace_s", 0.0)),
                        _cell(getattr(step, "drop_policy", "")),
                        gt_miss,
                        gt_idsw,
                        gt_motp,
                    ]
                    writer.writerow(row)
    return path
