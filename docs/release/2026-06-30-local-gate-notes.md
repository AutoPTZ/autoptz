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
| `artifacts/2026-06-30-8x1080p30-full-default-60s.json` | `full` | production shared model | 60 s | 29.9 | 0 | 2267.7 ms | 12.2% | 2433 MB |
| `artifacts/2026-06-30-8x1080p30-full-model-server-60s.json` | `full` | labs model server | 60 s | 29.9 | 0 | 42.0 ms | 49.8% | 6407 MB |
| `artifacts/2026-06-30-8x1080p30-pose-default-60s.json` | `pose_follow` | production shared model | 60 s | 30.0 | 0 | 41.6 ms | 32.7% | 2256 MB |
| `artifacts/2026-06-30-8x1080p30-pose-unified-60s.json` | `pose_follow` | unified pose flag | 60 s | 30.0 | 0 | 41.6 ms | 32.6% | 2244 MB |

Sender confirmation for both local batches was `30.0 fps` on all eight fake NDI
sources.

## Decisions From This Batch

- `simple_follow` is the only locally sustained inference/tracking gate that
  passed for 30 minutes: 8x1080p30, zero steady-state app-induced drops, 100%
  detection-active samples, and about 42 ms median end-to-end latency.
- `full` default is not acceptable as a production multi-camera path: it held
  capture fps and had zero app-induced drops, but median end-to-end latency was
  about 2.27 seconds.
- `AUTOPTZ_MODEL_SERVER=1` is the better latency architecture for the full
  feature stack, but it is not ready to become the default. It used about 6.4 GB
  RAM and the run logs show per-camera child processes still initializing
  InsightFace, so face/appearance ownership is not truly centralized yet.
- `AUTOPTZ_UNIFIED_POSE=1` stays gated. It matched default `pose_follow` capture
  and latency on this synthetic run, but did not improve latency/fps and raised
  median detector time from 20.4 ms to 23.0 ms. It still needs real tracking
  quality artifacts before promotion.
- Source-side `source_drop_est_*` remains noisy: NDI SDK dropped-video counters
  stayed at zero and app-induced drops stayed at zero, but the local receiver
  still reported source-drop estimates. Treat that as an accounting/source-pacing
  triage item, not as a proven app-induced capture miss.

## Remaining Release Gates

- Repeat the 30-minute Simple Follow gate on Intel Mac, CPU-only Windows, and
  accelerator-backed Windows.
- Run a 30-minute full-feature model-server gate after face/appearance model
  duplication is removed or bounded.
- Produce real-person tracking artifacts for default pose versus unified pose:
  moving person, crossing people, occlusion, delayed frames, reduced inference
  cadence, and PTZ motion feedback.
- Validate Windows face recognition and double-click enrollment on an actual
  Windows build, including visible face boxes and saved crop/embedding match.
