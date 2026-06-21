"""Camera-worker support modules.

Cohesive pieces extracted from the (large) ``camera_worker`` module so each
concern reads on its own:

- :mod:`autoptz.engine.worker.frame_source` — the synchronous FrameSource
  abstraction, ingest-adapter wrapper + fps pacing, and source construction.

``camera_worker`` re-exports the public names from here, so existing
``from autoptz.engine.camera_worker import …`` imports keep working.
"""
