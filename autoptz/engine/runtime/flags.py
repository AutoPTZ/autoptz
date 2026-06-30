"""Process-wide feature-flag env resolvers — the single source of truth.

These ``AUTOPTZ_*`` switches are read from more than one layer (the inference
pool, the worker stacks, the supervisor), so the parsing of "what counts as on"
lives here once instead of being re-implemented per call site.
"""

from __future__ import annotations

import os

# Strings that count as "on" for a boolean env flag.
_TRUE_VALUES = ("1", "true", "yes", "on")


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def env_process_per_camera() -> bool:
    """Whether camera workers should cross a process boundary.

    Lives here (not just in ``process_worker``) so lightweight callers like the
    inference layer can branch on it without importing the heavy worker module.
    The standalone ``AUTOPTZ_PROCESS_PER_CAMERA`` model-per-child experiment is
    retired and intentionally ignored.  The only remaining process-worker path is
    model-server mode, where each camera process delegates detection to one shared
    detector server instead of loading its own model set.
    """
    return env_model_server()


def env_model_server() -> bool:
    """Opt-in for the multi-process model-server architecture candidate.

    Each camera runs in its own process (escaping the GIL) and delegates detection
    to ONE shared model-server process (one model set → no per-process RAM cliff).
    This is not a product feature; it stays env-only until Mark artifacts prove the
    6/8-camera gates, CPU/RAM stability, and clean shutdown behavior.
    """
    return _env_true("AUTOPTZ_MODEL_SERVER")


def env_unified_pose() -> bool:
    """Process-wide opt-in for the unified (one-backbone) pose detector.

    Honoured in addition to the per-camera ``tracking.unified_pose`` config flag.
    """
    return _env_true("AUTOPTZ_UNIFIED_POSE")


def apply_opencv_thread_cap(threads: int | None = None) -> None:
    """Cap OpenCV's internal thread pool (resize/letterbox + optical flow).

    OpenCV defaults to *all* cores, so with several camera threads each firing
    cv2 work it oversubscribes the CPU — a real source of frame-time/CPU spikes.
    ``threads`` defaults to the ``AUTOPTZ_CV2_THREADS`` budget the supervisor
    publishes (so a spawned camera child can re-apply it in its own process).

    **Backend nuance:** with the TBB/OpenMP backends (typical on Linux/Windows)
    ``setNumThreads(n)`` honours *n*.  With the GCD/Concurrency backend (the macOS
    opencv-python wheels) a positive count is **ignored** and only ``0`` (force
    single-threaded) takes effect — so when we want a tight single-thread cap and
    the backend kept more, we fall back to disabling OpenCV's internal threading.
    Best-effort and must never raise into startup.
    """
    if threads is None:
        raw = os.environ.get("AUTOPTZ_CV2_THREADS", "").strip()
        if not raw:
            return
        try:
            threads = max(1, int(raw))
        except ValueError:
            return
    try:
        import cv2

        threads = max(1, int(threads))
        cv2.setNumThreads(threads)
        # GCD/Concurrency backends ignore a positive count; if we asked to squeeze
        # down to a single thread but it kept more, force OpenCV single-threaded so
        # per-camera cv2 work can't fan across every core under heavy multi-cam load.
        if threads <= 1 and cv2.getNumThreads() > 1:
            cv2.setNumThreads(0)
    except Exception:  # noqa: BLE001 — a thread-cap hint must never block startup
        pass


def apply_thread_caps(budget: int) -> None:
    """Cap OMP/BLAS/MKL/NumExpr env vars and the torch intra-op pool.

    Called from :func:`autoptz.engine.supervisor.Supervisor._apply_hardware_env`
    immediately after ORT and OpenCV are capped, with the same per-camera thread
    budget.

    **Two paths, two mechanisms:**

    *   **Process-per-camera (future)** — the child process inherits the env
        before any library is imported, so all four env vars take full effect.
    *   **In-process threaded path (current default)** — OMP/BLAS/MKL/NumExpr
        env vars only bind *before the library's first import*; by the time this
        runs those libraries may already be loaded and their thread pools already
        sized.  The only runtime knob that reliably reaches an already-imported
        library in-process is ``torch.set_num_threads``.  We therefore call it
        unconditionally (guarded by importability) so the torch pool — the
        heaviest in-process consumer via boxmot/insightface — is always capped.

    Must never raise; a thread-cap hint must never block startup.
    """
    n = max(1, int(budget))
    n_str = str(n)

    # Publish env vars so:
    #   • a model-server camera child inherits them before importing any lib, and
    #   • any library imported *after* this call (e.g. lazy-loaded reid backends)
    #     picks up the correct value automatically.
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = n_str

    # torch.set_num_threads is a runtime call — it caps the already-running
    # torch intra-op thread pool regardless of when torch was imported.
    # Guard so a torch-less install is unaffected and never hard-fails.
    _apply_torch_thread_cap(n)


def _apply_torch_thread_cap(n: int) -> None:
    """Set torch intra-op thread count to *n* if torch is importable; else no-op."""
    try:
        import torch

        torch.set_num_threads(n)
    except Exception:  # noqa: BLE001 — missing dep or import error → silent no-op
        pass


def env_torch_cap() -> int | None:
    """Return the torch thread count if torch is importable, else None.

    Utility for tests and diagnostics only — not used in hot paths.
    """
    try:
        import torch

        return int(torch.get_num_threads())
    except Exception:  # noqa: BLE001
        return None
