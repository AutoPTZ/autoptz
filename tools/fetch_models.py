"""CLI to pre-fetch / export AutoPTZ models offline.

Run once on a machine *with* network + the ML stack so the app starts with
detection working even when later run offline:

    python -m tools.fetch_models                # default cache dir
    python -m tools.fetch_models --cache-dir /tmp/m
    python -m tools.fetch_models --remove       # delete AutoPTZ-managed models

It downloads/exports the detector tiers and pose model into the platform
app-data ``…/AutoPTZ/models`` dir (the same dir
:class:`autoptz.engine.runtime.models.ModelManager` reads at engine start), and
pre-downloads the face-recognition pack (insightface ``buffalo_l``) into the
insightface cache so face enrolment works when later run OFFLINE — the fix for
"faces never save on an offline first-run".  Skip the face step with
``--detector-only``.  ReID (boxmot) weights use their upstream cache and are not
fetched here.

Exit code 0 if a usable detector ONNX is available afterwards, 1 otherwise.  A
failed face download is a warning only (face is optional), not a hard failure.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fetch_models",
        description="Pre-download/export AutoPTZ detector and pose models for offline use.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override the model cache directory (default: platform app-data …/AutoPTZ/models).",
    )
    parser.add_argument(
        "--detector-only",
        action="store_true",
        help="Fetch detector tiers only; skip the pose and face models.",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Delete AutoPTZ-managed detector/pose model files from the cache.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
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

    if args.remove:
        rows = mgr.remove_app_models()
        if not rows:
            log.info("No AutoPTZ-managed model files found.")
            return 0
        remove_failed = [row for row in rows if row["state"] != "removed"]
        for row in rows:
            if row["state"] == "removed":
                log.info("  → removed: %s", row["path"])
            else:
                log.error("  → FAILED: %s (%s)", row["path"], row["error"])
        return 1 if remove_failed else 0

    failed: list[str] = []
    ok_any = False
    for row in mgr.ensure_app_models(
        include_pose=not args.detector_only,
        progress=lambda label, value, total: log.info("[%d/%d] %s", value, total, label),
    ):
        if row["state"] == "ok":
            ok_any = True
            log.info("  → ready: %s", row["path"])
        else:
            failed.append(f"{row['name']}: {row['error']}")
            log.error("  → FAILED: %s", row["error"])

    if failed:
        log.error(
            "Some tiers could not be fetched (need `ultralytics` installed + "
            "network access, or set AUTOPTZ_MODEL_URL):\n  - %s",
            "\n  - ".join(failed),
        )

    # Face pack (insightface buffalo_l): pre-download so offline runs can enrol
    # faces.  Best-effort — a failure here is a warning, never a hard exit, since
    # face recognition is optional (manual click-to-track still works).
    if not args.detector_only:
        from autoptz.engine.pipeline.identify import ensure_face_model, insightface_root

        log.info("Fetching face model (insightface buffalo_l) into %s ...", insightface_root())
        face_err = ensure_face_model()
        if face_err:
            log.warning(
                "  → face model NOT available (offline face enrolment will be disabled): %s",
                face_err,
            )
        else:
            log.info("  → ready: insightface buffalo_l")

    return 0 if ok_any and not failed else (0 if ok_any else 1)


if __name__ == "__main__":
    sys.exit(main())
