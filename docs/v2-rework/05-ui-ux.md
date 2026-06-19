# 05 — UI / UX

Replaces v1's QWidget window (fixed pixel geometry, a `FlowLayout` of fixed‑size tiles, one
hard‑coded stylesheet, no layouts, no presets) with a fluid **PySide6 + Qt Quick (QML)** interface.

## 5.1 Principles

- **Simple to click, powerful underneath.** Common actions (select a camera, pick who to track,
  toggle tracking, recall a preset) are one click. Advanced tuning lives in a drawer, not in your
  face.
- **The wall is yours.** Drag to reorder, resize, and group tiles; save named layouts; per‑camera
  and global views.
- **Everything is remembered.** Layout, per‑camera settings, presets, identities, and window state
  persist (see `06`).
- **Never block the UI.** All video/inference is in the engine; the UI only renders preview frames
  + overlays from telemetry.

## 5.2 Information architecture

```
┌ Top bar ───────────────────────────────────────────────────────────────┐
│ [AutoPTZ]  Sources▾  Identities▾  Layouts▾  Theme▾        ●engine health │
├ Left rail (collapsible) ─┬ Camera Wall (QML grid) ───────────────────────┤
│  Sources (live discovery)│  ┌────────┐ ┌────────┐ ┌────────┐             │
│   • USB (hot-plug)       │  │ Cam A  │ │ Cam B  │ │ Cam C  │  drag to     │
│   • NDI (continuous find)│  │ ▶track │ │ ⏸      │ │ ▶track │  reorder /   │
│   • IP/ONVIF (discovered)│  └────────┘ └────────┘ └────────┘  resize      │
│   • + Add manually       │                                                │
│  Identities              │  selecting a tile opens ▶ right drawer         │
├──────────────────────────┴────────────────────────────────────────────── ┤
│ Right drawer (per-camera): Source · Tracking · PTZ · Presets · Tuning      │
└────────────────────────────────────────────────────────────────────────── ┘
```

## 5.3 Camera tile (`CameraTile.qml`)

- Live preview (from the worker's shared‑memory frame) with overlays drawn from telemetry:
  person boxes, the **target** highlighted, identity name + confidence, dead‑zone ellipse, center
  reticle, PTZ state, FPS/quality chip.
- Inline quick controls: **enable/disable tracking** toggle, target picker (click a box to follow
  that track, or choose an identity), preset quick‑recall, "active/selected" border state.
- States: `live`, `tracking`, `searching` (target lost), `reconnecting`, `no signal`, `error`.
- Click a person box → set as target. Click tile → select (opens drawer). Double‑click → maximize.

## 5.4 Per‑camera config drawer

Tabs:
- **Source** — type (USB/RTSP/ONVIF/NDI), address/credentials, sub‑stream vs main, resolution/fps,
  reconnect policy, rename.
- **Tracking** — tracker (BoT‑SORT/DeepOCSORT/ByteTrack), detect interval, ReID on/off + threshold,
  coast window, target mode (identity vs manual), face‑confirm on/off.
- **PTZ** — backend (auto‑detected, overridable), max speeds, invert axes, soft limits, dead‑zone
  size, controller gains (`Kp/Kd/Kv`), zoom framing preset (tight/medium/wide) + auto‑zoom on/off.
- **Presets** — named pan/tilt/zoom presets: save current, recall, set "home/default," set
  "on‑loss" fallback preset.
- **Tuning** — live sliders for dead‑zone/gains with a real‑time preview so an operator can dial in
  smoothness without editing files.

All changes are sent as `UpdateCameraConfig` commands and persisted immediately.

## 5.5 Global panels

- **Sources** — continuously discovered USB/NDI/ONVIF devices with add/remove; manual add by URL/IP;
  shows which are in use. (Fixes v1's startup‑only discovery.)
- **Identities** — enroll a person (capture face crops from a chosen camera), view/rename/delete,
  see which cameras are following whom. Backed by the InsightFace gallery (see `03`/`06`).
- **Layouts** — save/load/rename named wall layouts (tile positions, sizes, which cameras shown).
- **Theme** — light/dark + accent; tokens in a single QML theme file (no more stylesheet string in
  `constants.py`).

## 5.6 Rendering bridge (frames → QML)

- Worker writes the latest annotated preview into shared memory; the UI exposes it to QML via a
  `QQuickImageProvider` (or a custom `QQuickItem` that uploads to a texture). The UI overlays
  *dynamic* elements (boxes, reticle, labels) from telemetry in QML so overlays stay crisp and
  cheap and the preview JPEG/raw stays simple.
- Preview pull rate is independent of inference fps; tiles that are off‑screen or minimized drop to
  a low refresh to save cycles.

## 5.7 Accessibility & polish

- Keyboard: arrow keys nudge PTZ on the selected camera; number keys recall presets; space toggles
  tracking.
- Clear empty/onboarding states ("No cameras yet — add a source"), non‑blocking toasts for errors,
  and a visible engine‑health indicator (which EP is active: "GPU (TensorRT)" / "CPU").

> Implementation note: QML is the recommended path, but the engine contract is UI‑agnostic. If the
> team prefers QWidgets initially, a `QGraphicsView`/`QSplitter`‑based wall could be a stopgap —
> but QML is what delivers the "really nice, customizable" requirement, so prefer it.
</content>
