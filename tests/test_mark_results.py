from __future__ import annotations

import csv
import json

from autoptz.benchmark.profiles import get_profile
from autoptz.benchmark.results import (
    MarkResultBundle,
    collect_machine_info,
    save_mark_result,
    save_mark_result_csv,
    save_mark_result_to_path,
)
from autoptz.benchmark.runner import BenchmarkResult, BenchmarkRunner, StepResult


def _result(profile="full"):
    prof = get_profile(profile)
    r = BenchmarkRunner(
        prof,
        sample_fn=lambda n: [30.0] * n,
        floor_fps=24.0,
        max_cameras=2,
        dwell_s=0.0,
    )
    return r.run()


class TestMachineInfo:
    def test_has_core_fields(self) -> None:
        m = collect_machine_info()
        for key in (
            "os",
            "os_release",
            "cpu_count",
            "ram_gb",
            "execution_providers",
            "app_version",
        ):
            assert key in m
        assert isinstance(m["execution_providers"], list)


class _FakeStore:
    def __init__(self) -> None:
        self.kv: dict[str, object] = {}

    def set_setting(self, key, value) -> None:
        self.kv[key] = value


class TestSave:
    def test_writes_json_and_store(self, tmp_path) -> None:
        store = _FakeStore()
        path, bundle = save_mark_result(
            [_result("full"), _result("streams")],
            config_dir=tmp_path,
            store=store,
        )
        assert path.exists() and path.parent.name == "benchmarks"
        assert path.name.startswith("autoptz-mark-") and path.suffix == ".json"
        data = json.loads(path.read_text())
        assert data["app_version"]
        assert len(data["results"]) == 2
        assert data["results"][0]["profile"] == "full"
        # ConfigStore mirror written under last_mark_result
        assert "last_mark_result" in store.kv
        assert store.kv["last_mark_result"]["results"][0]["sustained_cameras"] == 2
        assert isinstance(bundle, MarkResultBundle)


class TestSaveToPath:
    def test_save_mark_result_to_path_writes_json(self, tmp_path) -> None:
        target = tmp_path / "nested" / "my-mark.json"
        path, bundle = save_mark_result_to_path([_result("full")], target)
        # Writes to the EXACT path requested (parents created).
        assert path == target
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["app_version"]
        assert len(data["results"]) == 1
        assert data["results"][0]["profile"] == "full"
        assert isinstance(bundle, MarkResultBundle)

    def test_save_mark_result_to_path_updates_store(self, tmp_path) -> None:
        store = _FakeStore()
        target = tmp_path / "my-mark.json"
        save_mark_result_to_path([_result("full")], target, store=store)
        assert "last_mark_result" in store.kv
        assert store.kv["last_mark_result"]["results"][0]["sustained_cameras"] == 2


# Exact CSV header (one row per step x camera).
_CSV_HEADER = [
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


def _quality(*, ttfa: float | None = 0.5) -> dict[str, object]:
    """One camera's QualityMetrics dict (as runner emits via ``to_dict``)."""
    return {
        "time_to_first_acquire_s": ttfa,
        "total_lost_duration_s": 0.25,
        "longest_lost_duration_s": 0.1,
        "lost_event_count": 1,
        "reacquire_count": 1,
        "id_switch_count": 2,
        "target_hold_pct": 87.5,
        "mean_target_confidence": 0.9123,
        "fps": 30.0,
        "dropped_frames": 3,
        "app_induced_drops": 0,
        "frames_delivered": 900,
        "frames_dropped_est": 0,
        "delivered_fps": 30.0,
        "source_fps": 30.0,
        "duplicate_frames": 0,
        "stale_frames": 0,
        "ndi_queue_depth": -1,
        "ndi_queue_audio": -1,
        "ndi_queue_metadata": -1,
        "ndi_total_video_frames": 900,
        "ndi_dropped_video_frames": 0,
        "ndi_total_audio_frames": 900,
        "ndi_dropped_audio_frames": 0,
        "ndi_total_metadata_frames": 3,
        "ndi_dropped_metadata_frames": 0,
        "ndi_connections": 1,
        "ndi_fourcc": "",
        "ndi_conversion_ms": 0.0,
    }


def _step(cameras: int, *, with_quality: bool = True, ttfa_none_idx: int = -1) -> StepResult:
    per_camera_fps = [30.0 - i for i in range(cameras)]
    quality: dict[str, dict] = {}
    if with_quality:
        for i in range(cameras):
            ttfa = None if i == ttfa_none_idx else 0.5
            quality[f"cam-{cameras}-{i}"] = _quality(ttfa=ttfa)
    return StepResult(
        cameras=cameras,
        min_fps=min(per_camera_fps),
        mean_fps=sum(per_camera_fps) / len(per_camera_fps),
        per_camera_fps=per_camera_fps,
        sustained=True,
        app_induced_drops=0,
        steady_state_app_induced_drops=0,
        per_camera_quality=quality,
    )


def _two_step_result() -> BenchmarkResult:
    """A 2-step result: step 1 has 2 cameras, step 2 has 3 cameras (-> 5 rows)."""
    return BenchmarkResult(
        profile="full",
        weight=1.0,
        floor_fps=24.0,
        max_cameras=3,
        sustained_cameras=3,
        min_fps_at_sustained=28.0,
        score=2.8,
        steps=[_step(2, ttfa_none_idx=1), _step(3)],
    )


class TestSaveCsv:
    def test_header_is_exact(self, tmp_path) -> None:
        path = save_mark_result_csv([_two_step_result()], tmp_path / "mark.csv")
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        assert rows[0] == _CSV_HEADER

    def test_one_row_per_step_x_camera(self, tmp_path) -> None:
        path = save_mark_result_csv([_two_step_result()], tmp_path / "mark.csv")
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        # 1 header + (2 cams + 3 cams) = 5 data rows.
        assert len(rows) == 1 + 5

    def test_quality_columns_populated_and_none_blank(self, tmp_path) -> None:
        path = save_mark_result_csv([_two_step_result()], tmp_path / "mark.csv")
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        # Step 1, camera 1 has time_to_first_acquire_s == None -> empty cell.
        step1 = [r for r in rows if r["step_cameras"] == "2"]
        assert step1[1]["time_to_first_acquire_s"] == ""
        # ...but its other quality values are still present.
        assert step1[1]["target_hold_pct"] == "87.5"
        assert step1[1]["id_switch_count"] == "2"
        assert step1[1]["dropped_frames"] == "3"
        assert step1[1]["app_induced_drops"] == "0"
        assert step1[1]["frames_delivered"] == "900"
        assert step1[1]["frames_dropped_est"] == "0"
        assert step1[1]["delivered_fps"] == "30.0"
        assert step1[1]["source_fps"] == "30.0"
        assert step1[1]["duplicate_frames"] == "0"
        assert step1[1]["stale_frames"] == "0"
        assert step1[1]["ndi_queue_depth"] == "-1"
        assert step1[1]["ndi_queue_audio"] == "-1"
        assert step1[1]["ndi_queue_metadata"] == "-1"
        assert step1[1]["ndi_total_video_frames"] == "900"
        assert step1[1]["ndi_dropped_video_frames"] == "0"
        assert step1[1]["ndi_total_audio_frames"] == "900"
        assert step1[1]["ndi_dropped_audio_frames"] == "0"
        assert step1[1]["ndi_total_metadata_frames"] == "3"
        assert step1[1]["ndi_dropped_metadata_frames"] == "0"
        assert step1[1]["ndi_connections"] == "1"
        assert step1[1]["ndi_fourcc"] == ""
        assert step1[1]["ndi_conversion_ms"] == "0.0"
        assert step1[1]["step_app_induced_drops"] == "0"
        assert step1[1]["steady_state_app_induced_drops"] == "0"
        assert step1[1]["source_mutation_events"] == "0"
        assert step1[1]["source_mutation_allowed_drops"] == "0"
        assert step1[1]["source_mutation_drop_grace_s"] == "0.0"
        assert step1[1]["drop_policy"] == "steady_state_zero_source_mutation_grace_only"
        # Step 1, camera 0 has a real ttfa.
        assert step1[0]["time_to_first_acquire_s"] == "0.5"

    def test_camera_idx_and_fps_track_per_camera(self, tmp_path) -> None:
        path = save_mark_result_csv([_two_step_result()], tmp_path / "mark.csv")
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        step2 = [r for r in rows if r["step_cameras"] == "3"]
        assert [r["camera_idx"] for r in step2] == ["0", "1", "2"]
        assert [r["per_camera_fps"] for r in step2] == ["30.0", "29.0", "28.0"]
        assert step2[0]["sustained"] == "True"

    def test_gt_blank_without_ground_truth(self, tmp_path) -> None:
        path = save_mark_result_csv([_two_step_result()], tmp_path / "mark.csv")
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            assert r["gt_miss_rate"] == ""
            assert r["gt_id_switch_rate"] == ""
            assert r["gt_motp"] == ""
            assert r["scene_clip_id"] == ""

    def test_meta_from_bundle(self, tmp_path) -> None:
        path = save_mark_result_csv([_two_step_result()], tmp_path / "mark.csv")
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows
        for r in rows:
            assert r["app_version"]  # reused from the bundle metadata
            assert r["created_at"]
            assert r["profile"] == "full"

    def test_round_trips_via_dictreader(self, tmp_path) -> None:
        path = save_mark_result_csv([_two_step_result()], tmp_path / "mark.csv")
        assert path == tmp_path / "mark.csv"
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames == _CSV_HEADER
            rows = list(reader)
        assert len(rows) == 5
        # Cells round-trip as the strings we wrote.
        assert rows[0]["mean_target_confidence"] == "0.9123"

    def test_real_runner_result_no_quality(self, tmp_path) -> None:
        # The math-only runner emits no per_camera_quality -> quality + gt cells blank,
        # but per-camera fps rows are still emitted (one per sustained camera).
        path = save_mark_result_csv([_result("full")], tmp_path / "mark.csv")
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows
        for r in rows:
            assert r["target_hold_pct"] == ""
            assert r["gt_motp"] == ""

    def test_store_mirror_optional(self, tmp_path) -> None:
        store = _FakeStore()
        # store is accepted (kept signature-compatible) and does not break the write.
        path = save_mark_result_csv([_two_step_result()], tmp_path / "mark.csv", store=store)
        assert path.exists()

    def test_step_drop_accounting_columns_are_exported(self, tmp_path) -> None:
        step = StepResult(
            cameras=1,
            min_fps=30.0,
            mean_fps=30.0,
            per_camera_fps=[30.0],
            sustained=True,
            app_induced_drops=4,
            steady_state_app_induced_drops=0,
            source_mutation_events=1,
            source_mutation_allowed_drops=4,
            source_mutation_drop_grace_s=2.0,
            per_camera_quality={"cam-1": _quality()},
        )
        result = BenchmarkResult(
            profile="full",
            weight=1.0,
            floor_fps=30.0,
            max_cameras=1,
            sustained_cameras=1,
            min_fps_at_sustained=30.0,
            score=1.0,
            steps=[step],
        )
        path = save_mark_result_csv([result], tmp_path / "mark.csv")
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["app_induced_drops"] == "0"
        assert rows[0]["step_app_induced_drops"] == "4"
        assert rows[0]["steady_state_app_induced_drops"] == "0"
        assert rows[0]["source_mutation_events"] == "1"
        assert rows[0]["source_mutation_allowed_drops"] == "4"
        assert rows[0]["source_mutation_drop_grace_s"] == "2.0"
