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
    from autoptz.engine.runtime.models import ModelManager

    mgr = ModelManager(cache_dir=args.cache_dir)
    log.info("Model cache dir: %s", mgr.cache_dir)

    path = mgr.ensure_detector()
    if path is None:
        log.error(
            "Could not obtain a detector model. Ensure `ultralytics` is "
            "installed and you have network access, then retry.",
        )
        return 1

    log.info("Detector model ready: %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
