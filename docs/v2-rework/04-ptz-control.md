# 04 — PTZ Control

v1's `ptz_control()` is bang‑bang: it picks one of 8 directions when the centroid leaves a fixed
ellipse and stops when inside it. There is no velocity model, no smoothing, no lead, and no
auto‑zoom — so motion is jerky and lags moving subjects. v2 replaces this with a **unified backend
interface + a single closed‑loop controller** shared by all camera types.

## 4.1 Backend interface

```python
class PTZBackend(Protocol):
    capabilities: PTZCaps          # continuous? absolute? presets? zoom? speed ranges
    def move_velocity(self, pan: float, tilt: float, zoom: float) -> None  # normalized [-1,1]
    def move_absolute(self, pan: float, tilt: float, zoom: float) -> None  # if supported
    def stop(self) -> None
    def get_position(self) -> PTZState | None    # absolute pos if the device reports it
    def goto_preset(self, idx: int) -> None
    def save_preset(self, idx: int) -> None
    def close(self) -> None
```

Implementations (`engine/ptz/`):
- **`ndi_ptz.py`** — cyndilib: `recv_ptz_pan_tilt_speed`, absolute, zoom, presets, tally.
- **`visca_ip.py`** — VISCA over IP: continuous + absolute (PanTiltPosInq), presets, zoom.
- **`visca_usb.py`** — refactor v1's working serial VISCA path behind this interface.
- **`onvif_ptz.py`** — `ContinuousMove`, `AbsoluteMove`, `GotoPreset`, `SetPreset` via
  python‑onvif‑zeep / pyptz. **Biggest coverage win** for "nearly all IP cameras."

Each backend advertises `PTZCaps` so the controller adapts (e.g., fall back to velocity moves when
absolute isn't supported; hide preset UI when unsupported).

## 4.2 Normalized control space

All backends accept **normalized** pan/tilt/zoom in `[-1, 1]`. Each backend maps that to its native
range (VISCA speed steps, NDI float speeds, ONVIF velocity vectors). Per‑camera config stores
`max_pan_speed`, `max_tilt_speed`, `max_zoom_speed`, and an `invert_pan/tilt` flag. This removes
v1's hard‑coded magic speeds (`2 + eased*(7-2)` for USB, `0.03..0.2` for NDI) into tunable,
remembered settings.

## 4.3 Closed‑loop controller (`engine/ptz/controller.py`)

Inputs each tick: target `error=(ex,ey)` in normalized frame units, target `velocity=(vx,vy)`
(from the tracker's Kalman state), `subject_height`, and `zoom_target`.

```
# 1. Dead-zone: ignore tiny errors (per-camera ellipse), prevents micro-jitter
if |ex| < dz_x and |ey| < dz_y: ex_eff, ey_eff = 0, 0

# 2. Smoothing: one-euro filter on error and velocity (low latency, low jitter)
ex_f = oneEuro(ex_eff); ey_f = oneEuro(ey_eff)

# 3. PD + velocity feed-forward (keep relative speed):
pan_cmd  = Kp*ex_f + Kd*d(ex_f)/dt + Kv*vx
tilt_cmd = Kp*ey_f + Kd*d(ey_f)/dt + Kv*vy
#   Kv*v makes the camera LEAD a moving subject instead of always lagging it.

# 4. Clamp to per-camera max speeds; apply non-linear response curve so small
#    errors → gentle moves, large errors → fast catch-up (smooth ease-in/out).
pan_cmd  = shape(clamp(pan_cmd,  -1, 1))
tilt_cmd = shape(clamp(tilt_cmd, -1, 1))

# 5. Zoom controller (if enabled): error vs desired subject height, hysteresis band
zoom_cmd = zoom_pd(subject_height, zoom_target) with hysteresis + rate limit

backend.move_velocity(pan_cmd, tilt_cmd, zoom_cmd)
```

- **Coast on loss:** if the target becomes `lost`, keep the last `velocity` command for a short
  *coast window* (e.g., 300–600 ms) so a briefly‑occluded subject is followed through, then
  `stop()` and enter **search** (optionally zoom out) until ReID/face re‑binds (see `03`).
- **Rate limiting:** the PTZ thread sends commands at a bounded rate (e.g., 15–30 Hz) decoupled
  from inference fps; redundant identical commands are suppressed (don't spam the bus).
- **Gains** `Kp, Kd, Kv`, dead‑zone, max speeds, response curve, and coast window are **per‑camera,
  persisted** settings with sane defaults and a "tuning" panel in the UI.

## 4.4 Absolute position & presets (remembered states)

- When a backend reports absolute position (`get_position`), the engine periodically records it so
  the UI can show pan/tilt/zoom and so a camera can **return to a saved framing** on demand or at
  startup ("home"/"default" preset per camera).
- **Presets:** `save_preset(idx)` / `goto_preset(idx)` map to native presets where supported; where
  not, store the absolute position in SQLite and `move_absolute` to it. Presets are per camera,
  named, and persisted (e.g., "Pulpit," "Wide," "Stage L"). This is a flagship v2 feature absent
  in v1.
- Recall is a single command from the UI preset bar and is also usable as a tracking fallback
  (e.g., "on loss for >10 s, return to Wide preset").

## 4.5 Safety & limits

- **Soft limits:** optional per‑camera pan/tilt/zoom min/max so tracking can't swing into a wall or
  the ceiling. Clamp `move_absolute`/velocity at the limits.
- **Stop‑on‑exit / stop‑on‑disable:** worker exit, tracking‑disable, target‑cleared, and source‑drop
  all issue `stop()` immediately (v1 does some of this; make it universal and reliable).
- **One backend, one owner:** a PTZ device is owned by exactly one camera worker (addressed by
  stable id). No global "current PTZ device" — this removes another v1 crossing hazard.
</content>
