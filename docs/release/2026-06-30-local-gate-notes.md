# 2026-06-30 Local Gate Notes

These notes summarize local fake-NDI artifacts collected on Apple Silicon during
the 2.2.0 reliability overhaul. They are evidence for PR review, not a full
release sign-off. Intel Mac, CPU-only Windows, accelerator-backed Windows, real
NDI, mixed USB/RTSP/ONVIF, and PTZ hardware tracking gates still need their own
artifacts.

## Artifacts

| Artifact | Profile | Runtime | Duration | Delivered fps | App-induced drops | E2E latency | CPU | RAM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `artifacts/2026-06-30-8x1080p30-simple-follow-30min.json` | `simple_follow` | production shared model | 30 min | 30.0 | 0 | 42.1 ms | 41.4% | 514 MB |
| `artifacts/2026-06-30-8x1080p30-simple-follow-30min-post-estimator.json` | `simple_follow` | production shared model | 30 min | 30.0 | 0 | 46.1 ms | 40.1% | 524 MB |
| `artifacts/2026-06-30-8x1080p30-full-default-60s.json` | `full` | production shared model | 60 s | 29.9 | 0 | 2267.7 ms | 12.2% | 2433 MB |
| `artifacts/2026-06-30-8x1080p30-full-model-server-60s.json` | `full` | labs model server | 60 s | 29.9 | 0 | 42.0 ms | 49.8% | 6407 MB |
| `artifacts/2026-06-30-8x1080p30-full-model-server-30min.json` | `full` | labs model server | 30 min | 29.7 | 0 | 42.1 ms | 50.3% | 3734 MB |
| `artifacts/2026-06-30-8x1080p30-pose-default-60s.json` | `pose_follow` | production shared model | 60 s | 30.0 | 0 | 41.6 ms | 32.7% | 2256 MB |
| `artifacts/2026-06-30-8x1080p30-pose-unified-60s.json` | `pose_follow` | unified pose flag | 60 s | 30.0 | 0 | 41.6 ms | 32.6% | 2244 MB |
| `artifacts/2026-06-30-8x1080p30-streams-drop-estimator-smoke.json` | `streams` | production shared model | 60 s | 30.0 | 0 | 41.5 ms | 6.6% | 1988 MB |

Sender confirmation for all local batches was `30.0 fps` on all eight fake NDI
sources.

## Decisions From This Batch

- `simple_follow` is the only locally sustained inference/tracking gate that
  passed for 30 minutes: 8x1080p30, zero steady-state app-induced drops, 100%
  detection-active samples, and about 42-46 ms median end-to-end latency. The
  post-estimator rerun also held zero source-drop estimate and zero NDI
  dropped-video frames while the sender reported 30.0 fps on all eight sources.
- `full` default is not acceptable as a production multi-camera path: it held
  capture fps and had zero app-induced drops, but median end-to-end latency was
  about 2.27 seconds.
- `AUTOPTZ_MODEL_SERVER=1` is the better latency architecture for the full
  feature stack, but it is not ready to become the default. The 30-minute run
  held zero app-induced drops and about 42 ms latency, but delivered-fps median
  was 29.7 rather than the strict 30.0 target. The run logs also show per-camera
  child processes still initializing InsightFace, so face/appearance ownership
  is not truly centralized yet.
- `AUTOPTZ_UNIFIED_POSE=1` stays gated. It matched default `pose_follow` capture
  and latency on this synthetic run, but did not improve latency/fps and raised
  median detector time from 20.4 ms to 23.0 ms. It still needs real tracking
  quality artifacts before promotion.
- Source-side `source_drop_est_*` in the older full/simple/pose artifacts was captured
  before the conservative source-drop estimator landed, so those values overstate
  normal receiver pacing jitter. NDI SDK dropped-video counters and app-induced
  drops stayed at zero. The post-fix streams smoke and 30-minute Simple Follow
  rerun both held 30.0 delivered fps with zero app-induced drops, zero NDI
  dropped-video, and zero source-drop estimate. Re-run the remaining full/pose
  fake-NDI artifacts after this fix before using
  `source_drop_est_*` for source-pacing triage.

## Remaining Release Gates

- Do not cut an RC from this branch until these gates have matching artifacts;
  local Apple Silicon fake-NDI evidence is not enough for release.
- Confirm the only allowed capture drops are during explicit source add/remove
  transitions. Any steady-state `app_induced_drops` value above zero blocks the
  release.
- Repeat the 30-minute Simple Follow gate on Intel Mac, CPU-only Windows, and
  accelerator-backed Windows.
- Re-run the 30-minute full-feature model-server gate after face/appearance model
  duplication is removed or bounded, and require delivered fps to hold the strict
  30.0 target.
- Produce real-person tracking artifacts for default pose versus unified pose:
  moving person, crossing people, occlusion, delayed frames, reduced inference
  cadence, and PTZ motion feedback.
- Validate Windows face recognition and double-click enrollment on an actual
  Windows build, including visible face boxes and saved crop/embedding match.
  The local code now covers the previously mismatched crop-space path where the
  preview thumbnail used cropped-preview coordinates but the worker interpreted
  the click as full-frame coordinates. Runtime diagnostics also now mark the
  Face service/stage as `failed` with the recognizer `last_error` when InsightFace
  or its model pack does not load, instead of reporting Face as active while no
  boxes can be drawn. Real Windows GUI validation is still required.
- The updater checksum lookup now treats Intel macOS asset aliases
  (`macos-intel`, `macos-x64`, `macos-x86_64`) as the same artifact name for
  manifest lookup while still requiring the downloaded file's SHA-256 to match.
  This addresses the reported Intel macOS "checksum file did not contain an
  entry" failure without disabling integrity verification.
- Validate Mark quit-in-middle behavior on an actual Windows GUI session. Headless
  CI covers the signal/teardown contract, but it is not a substitute for closing
  a running Mark window on Windows with the real event loop and build package.
