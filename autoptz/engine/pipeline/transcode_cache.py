"""Per-scene transcode cache for AutoPTZ Mark v3 (pure, headless — no Qt).

A *master* clip is the highest-fidelity recording we hold for a scene. Mark
needs to replay that scene at a grid of (resolution, fps) targets so the
benchmark can stress the pipeline at, say, 4K60 even when the master was only
shot at 1080p30. Re-decoding + re-scaling on every run is wasteful, so we
transcode each target *variant* once and cache it on disk keyed by clip.

Two honesty rules drive the design:

- **Availability table.** :meth:`TranscodeCache.valid_combos` tags every
  (res, fps) combo relative to the master as ``native`` / ``downscaled`` /
  ``upscaled`` and ``native`` / ``resampled`` / ``interpolated`` so the UI can
  tell the user which variants are *synthetic* (upscaled pixels or duplicated
  frames) rather than real captured fidelity.
- **Atomic writes.** Variants are written to a sibling ``.tmp.mp4`` and
  ``os.replace``-d into place so a crash mid-transcode can never leave a
  half-written file that a later run mistakes for a valid cache hit (this race
  corrupted a clip before).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import cv2

CACHE_VERSION = 1

# Minimum playable length of a built variant (seconds). The master is looped if
# it is shorter so every variant gives the benchmark a stable run window.
_MIN_OUTPUT_S = 8.0


class TranscodeCache:
    """Disk cache of per-scene transcoded variants, keyed by ``clip_id``."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        if cache_dir is None:
            # Imported lazily: the config store pulls in pydantic/sqlite, which
            # is heavy for a module the UI imports just to read valid_combos.
            from autoptz.config.store import default_config_dir  # noqa: PLC0415

            cache_dir = default_config_dir() / "transcode_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Paths / lookup ─────────────────────────────────────────────────────
    def _variant_path(self, clip_id: str, target_res: tuple[int, int], target_fps: float) -> Path:
        w, h = target_res
        name = f"{int(w)}x{int(h)}_{int(target_fps)}fps_v{CACHE_VERSION}.mp4"
        return self.cache_dir / clip_id / name

    def get_cached_variant(
        self, clip_id: str, target_res: tuple[int, int], target_fps: float
    ) -> Path | None:
        """Return the cached variant path iff it exists and is non-empty."""
        path = self._variant_path(clip_id, target_res, target_fps)
        try:
            if path.is_file() and path.stat().st_size > 0:
                return path
        except OSError:
            return None
        return None

    # ── Availability / honesty table ───────────────────────────────────────
    def valid_combos(
        self,
        master_res: tuple[int, int],
        master_fps: float,
        *,
        resolutions: tuple[tuple[int, int], ...] = (
            (1280, 720),
            (1920, 1080),
            (3840, 2160),
        ),
        fpses: tuple[float, ...] = (24.0, 30.0, 60.0),
    ) -> list[dict]:
        """Tag every (res, fps) target relative to the master.

        ``synthetic`` is True when the variant fabricates fidelity the master
        does not have — upscaled pixels or interpolated (duplicated) frames.
        """
        mw, mh = master_res
        master_px = mw * mh
        combos: list[dict] = []
        for res in resolutions:
            w, h = res
            px = w * h
            if px > master_px:
                res_tag = "upscaled"
            elif px < master_px:
                res_tag = "downscaled"
            else:
                res_tag = "native"
            for fps in fpses:
                if fps > master_fps:
                    fps_tag = "interpolated"
                elif fps < master_fps:
                    fps_tag = "resampled"
                else:
                    fps_tag = "native"
                synthetic = res_tag == "upscaled" or fps_tag == "interpolated"
                combos.append(
                    {
                        "res": res,
                        "fps": fps,
                        "res_tag": res_tag,
                        "fps_tag": fps_tag,
                        "synthetic": synthetic,
                    }
                )
        return combos

    # ── Build ──────────────────────────────────────────────────────────────
    def build_cached_variant(
        self,
        clip_id: str,
        master_path: Path,
        master_res: tuple[int, int],
        master_fps: float,
        target_res: tuple[int, int],
        target_fps: float,
    ) -> Path:
        """Transcode the master to (target_res, target_fps); cache atomically.

        Retiming uses a fractional accumulator: per *source* frame we add
        ``target_fps / master_fps`` and emit ``floor()`` copies of the resized
        frame. ``> 1`` duplicates (interpolation by frame-dup); ``< 1`` drops
        (resampling). The master is looped until at least ``_MIN_OUTPUT_S`` of
        output has been written. The result is written to a ``.tmp.mp4`` and
        ``os.replace``-d into place so readers never see a partial file.
        """
        final = self._variant_path(clip_id, target_res, target_fps)
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.parent / f".{final.name}.tmp.mp4"

        target_fps = float(target_fps)
        master_fps = float(master_fps) if master_fps else target_fps
        w, h = int(target_res[0]), int(target_res[1])
        min_frames = math.ceil(_MIN_OUTPUT_S * target_fps)
        step = target_fps / master_fps  # output frames emitted per source frame

        cap: cv2.VideoCapture | None = None
        writer: cv2.VideoWriter | None = None
        try:
            cap = cv2.VideoCapture(str(master_path))
            if not cap.isOpened():
                raise RuntimeError(f"cannot open master clip: {master_path}")

            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter.fourcc(*"mp4v"), target_fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"cannot open VideoWriter for {tmp}")

            written = 0
            accumulator = 0.0
            saw_any_frame = False
            # Loop the master until we have at least min_frames of output.
            while written < min_frames:
                ok, frame = cap.read()
                if not ok:
                    if not saw_any_frame:
                        raise RuntimeError(f"master produced no frames: {master_path}")
                    # Rewind and loop.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                saw_any_frame = True
                resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                accumulator += step
                emit = int(math.floor(accumulator))
                accumulator -= emit
                for _ in range(emit):
                    if written >= min_frames:
                        break
                    writer.write(resized)
                    written += 1

            writer.release()
            writer = None
            cap.release()
            cap = None

            os.replace(tmp, final)
            return final
        except Exception:
            if writer is not None:
                writer.release()
            if cap is not None:
                cap.release()
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            raise

    # ── Eviction ───────────────────────────────────────────────────────────
    def cleanup(self, max_total_bytes: int = 1_000_000_000) -> None:
        """LRU-by-mtime eviction of cached variants over ``max_total_bytes``."""
        files: list[tuple[float, int, Path]] = []
        for path in self.cache_dir.rglob("*.mp4"):
            if not path.is_file():
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            files.append((st.st_mtime, st.st_size, path))

        total = sum(size for _, size, _ in files)
        if total <= max_total_bytes:
            return

        # Evict oldest-first until under budget.
        files.sort(key=lambda t: t[0])
        for _, size, path in files:
            if total <= max_total_bytes:
                break
            try:
                path.unlink()
                total -= size
            except OSError:
                continue
