AutoPTZ Notices
================

AutoPTZ is licensed under the GNU Affero General Public License version 3.0.
See `LICENSE.md`.

Third-party code and model assets can have their own licenses. AutoPTZ tries to
make those boundaries visible in the app under Services -> Optional Setup.

Models and optional features:

- YOLO detector and pose models are used for person boxes, automatic PTZ
  following, pose overlay, and torso-stable framing. Ultralytics states that
  its YOLO trained models are AGPL-3.0 by default. AutoPTZ can download/export
  these into its own model cache, and can remove the AutoPTZ-managed cache files
  from the Services panel.
- InsightFace is used only for optional face recognition. The InsightFace code
  is MIT-licensed, but the upstream model packs are labeled for non-commercial
  research unless a commercial license is obtained from InsightFace. AutoPTZ
  does not delete the upstream InsightFace cache because it may be shared by
  other tools.
- BoxMOT/ReID is used only for optional stable re-acquire after occlusion.
  Its package/model dependencies are managed by the upstream package ecosystem,
  not by AutoPTZ's model cache.
- NDI support is optional and depends on `cyndilib`.

Missing models or optional dependencies should never prevent AutoPTZ from
launching. The app disables unavailable feature controls and keeps live preview
available where possible.

AutoPTZ Mark benchmark video clips:

The bundled demo clips under `autoptz/assets/` are used only by the "AutoPTZ
Mark" benchmark to drive synthetic cameras. Each was trimmed/transcoded
(re-scaled, re-timed, audio removed) from a permissively-licensed source:

- `mark_people_1080p.mp4` — derived from the OpenCV sample `vtest.avi`
  (pedestrians). OpenCV samples are BSD-3-Clause.
- `mark_crowd_30.mp4` — derived from "Shibuya Scramble Crossing" by Gst,
  via Wikimedia Commons, licensed CC BY-SA 3.0. The AutoPTZ-derived clip is
  likewise made available under CC BY-SA 3.0 (ShareAlike).
- `mark_people_24.mp4` and `mark_people_60.mp4` — derived from "Tears of Steel"
  (CC) Blender Foundation | mango.blender.org, licensed CC BY 3.0. The
  `mark_people_60.mp4` variant was frame-interpolated to 60 fps.
- `mark_faces_30.mp4` — derived from a U.S. White House Daily Press Briefing
  (April 11, 2016). A work of the U.S. federal government, in the Public Domain
  (17 U.S.C. § 101). Used as the face-recognition benchmark scene.

These clips are demo assets only and are independent of the AGPL-3.0 license
that covers AutoPTZ's own source code.
