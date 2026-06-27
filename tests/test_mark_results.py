from __future__ import annotations

import json

from autoptz.benchmark.profiles import get_profile
from autoptz.benchmark.results import (
    MarkResultBundle,
    collect_machine_info,
    save_mark_result,
    save_mark_result_to_path,
)
from autoptz.benchmark.runner import BenchmarkRunner


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
