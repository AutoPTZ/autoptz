"""CLI to pre-fetch / export AutoPTZ models offline.

Run once on a machine *with* network + the ML stack so the app starts with
detection working even when later run offline:

    python -m tools.fetch_models                # default cache dir
    python -m tools.fetch_models --cache-dir /tmp/m

It downloads the YOLO11 ``.pt`` via ultralytics and exports the NMS-free ONNX
into the platform app-data ``…/AutoPTZ/models`` dir (the same dir
:class:`autoptz.engine.runtime.models.ModelManager` reads at engine start).

Face (insightface) and ReID (boxmot) weights auto-download into their own
caches on first use and are out of scope for this CLI.

Exit code 0 if a usable detector ONNX is available afterwards, 1 otherwise.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fetch_models",
        description="Pre-download/export AutoPTZ detection models for offline use.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override the model cache directory (default: platform app-data "
             "…/AutoPTZ/models).",
    )
    parser.add_argument(
        "--tier",
        choices=("all", "auto", "fast", "balanced", "medium"),
        default="all",
        help="Which detector tier(s) to fetch (default: all, so the Balanced "
             "and Medium tiers work offline too).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("fetch_models")

    # Import lazily so --help works without the engine package importable.
    from autoptz.engine.runtime.models import detector_model_for_tier, ModelManager

    mgr = ModelManager(cache_dir=args.cache_dir)
    log.info("Model cache dir: %s", mgr.cache_dir)

    # Resolve the requested tier(s) to their distinct weight files so we don't
    # export the same .pt twice (auto and fast both map to yolo11n).
    tiers = ("auto", "fast", "balanced", "medium") if args.tier == "all" else (args.tier,)
    seen_models: set[str] = set()
    ok_any = False
    failed: list[str] = []
    for tier in tiers:
        model_pt = detector_model_for_tier(tier)
        if model_pt in seen_models:
            continue
        seen_models.add(model_pt)
        log.info("Fetching detector tier %r (%s)…", tier, model_pt)
        path = mgr.ensure_detector(tier=tier)
        if path is None:
            failed.append(f"{tier} ({model_pt}): {mgr.last_error or 'unavailable'}")
            log.error("  → FAILED: %s", mgr.last_error or "unavailable")
        else:
            ok_any = True
            log.info("  → ready: %s", path)

    if failed:
        log.error(
            "Some tiers could not be fetched (need `ultralytics` installed + "
            "network access, or set AUTOPTZ_MODEL_URL):\n  - %s",
            "\n  - ".join(failed),
        )
    return 0 if ok_any and not failed else (0 if ok_any else 1)


if __name__ == "__main__":
    sys.exit(main())
