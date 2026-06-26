"""AutoPTZ Mark — machine info capture + result persistence (pure, no Qt)."""

from __future__ import annotations

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
    bundle = MarkResultBundle(
        created_at=datetime.now(UTC).isoformat(),
        app_version=__version__,
        machine=collect_machine_info(),
        results=[r.to_dict() for r in results],
    )
    path = out_dir / f"autoptz-mark-{stamp}.json"
    path.write_text(json.dumps(bundle.to_dict(), indent=2))
    if store is not None:
        set_setting = getattr(store, "set_setting", None)
        if callable(set_setting):
            set_setting("last_mark_result", bundle.to_dict())
    return path, bundle
