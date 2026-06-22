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
