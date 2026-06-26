"""CLI arg-routing tests for the --benchmark flag in autoptz.__main__."""

from __future__ import annotations

import sys

import pytest


def test_benchmark_flag_routes_to_run_benchmark(monkeypatch) -> None:
    import autoptz.__main__ as main_mod

    captured: dict[str, object] = {}

    def fake_run_benchmark(**kwargs):
        captured.update(kwargs)
        return 0

    # The import inside main() resolves autoptz.benchmark.run_benchmark.
    import autoptz.benchmark as bench_pkg

    monkeypatch.setattr(bench_pkg, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autoptz",
            "--benchmark",
            "--benchmark-profile",
            "streams",
            "--benchmark-duration",
            "3",
            "--benchmark-max-cameras",
            "5",
            "--benchmark-floor",
            "20",
            "--benchmark-json",
            "/tmp/mark.json",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main_mod.main()
    assert exc.value.code == 0
    assert captured["profile"] == "streams"
    assert captured["dwell_s"] == 3.0
    assert captured["max_cameras"] == 5
    assert captured["floor_fps"] == 20.0
    assert captured["json_path"] == "/tmp/mark.json"


def test_plain_bench_flag_still_routes_to_acceleration(monkeypatch) -> None:
    """--bench (no -mark) keeps routing to the inference-acceleration bench."""
    import autoptz.__main__ as main_mod

    called: dict[str, object] = {}

    def fake_accel(tier, json_path=None):
        called["tier"] = tier
        return 0

    import autoptz.engine.runtime.bench as accel_mod

    monkeypatch.setattr(accel_mod, "run_acceleration_bench", fake_accel)
    monkeypatch.setattr(sys, "argv", ["autoptz", "--bench", "--bench-tier", "auto"])
    with pytest.raises(SystemExit):
        main_mod.main()
    assert called["tier"] == "auto"
