"""Center Stage auto-framer — compute a smoothed crop that frames the target.

Driving the digital crop through the physical PTZ velocity controller did not
work (its conservative auto-zoom never engaged). This computes the crop directly
from the target's bounding box each frame instead — centre on the subject, size
the crop so the subject fills a configured fraction of it, match the output
aspect ratio, clamp inside the frame, and EMA-smooth so it does not jitter.

The crop is constrained to a *window*: never larger than ``max_frac`` of the
frame (so there is always a visible, panning Center-Stage crop even when the
subject is close and already fills the sensor), and never smaller than
``min_frac`` (so a far subject doesn't over-zoom into a blurry postage stamp).
"""

from __future__ import annotations

from dataclasses import dataclass


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def desired_crop(
    bbox: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    *,
    out_aspect: float,
    fill: float,
    min_frac: float,
    max_frac: float,
    headroom: float = 0.10,
) -> tuple[float, float, float, float]:
    """The crop ``(x, y, w, h)`` (pixels) that frames *bbox*.

    ``fill`` = the fraction of the crop height the subject should occupy (smaller
    fill → looser shot). The resulting crop height is then constrained to
    ``[min_frac, max_frac]`` of the frame height, so the crop is always a visible
    window (never the whole frame) and never an extreme zoom. ``out_aspect`` keeps
    the crop matching the output so the resize doesn't distort. ``headroom`` lifts
    the centre so the head sits a little below the top.
    """
    bx1, by1, bx2, by2 = (float(v) for v in bbox)
    subj_h = max(1.0, by2 - by1)
    cx = (bx1 + bx2) * 0.5
    cy = (by1 + by2) * 0.5
    fw, fh = float(frame_w), float(frame_h)

    # Size the crop to the subject, then constrain it to a window of the frame.
    ch = subj_h / _clamp(fill, 0.1, 1.0)
    ch = _clamp(ch, min_frac * fh, max_frac * fh)
    cw = ch * out_aspect
    # If that is wider than the frame, cap width (keeps aspect; only happens for
    # very wide outputs — with max_frac < 1 the height stays within the frame).
    if cw > fw:
        cw, ch = fw, fw / out_aspect

    cy_adj = cy - headroom * ch
    x = _clamp(cx - cw * 0.5, 0.0, max(0.0, fw - cw))
    y = _clamp(cy_adj - ch * 0.5, 0.0, max(0.0, fh - ch))
    return (x, y, cw, ch)


@dataclass
class DigitalFramer:
    """Stateful EMA-smoothed wrapper around :func:`desired_crop`.

    Position (x, y) and size (w, h) are smoothed *separately* so detector-height
    flicker doesn't cause visible zoom breathing:

    * **Position dead-zone** (``deadzone``) — the crop centre only re-targets when
      the desired centre has moved more than ``deadzone`` of the current crop's
      width/height. Inside the band the centre is *held* (a stationary subject
      with normal bbox noise produces a still frame). Hysteresis: once the centre
      is moving it keeps following until the desired centre is back inside a
      *tighter* inner band (``deadzone * _INNER_BAND_FRAC``), so it does not
      chatter on the boundary.
    * **Size smoothing** (``size_smooth``) eases w/h at its own, slower constant,
      and a **size dead-band** (``size_deadband``) ignores size changes under a
      few percent so small height flicker leaves the zoom untouched.

    ``deadzone=0.0`` with ``size_smooth == smooth`` and ``size_deadband=0.0``
    reproduces the original uniform-EMA behaviour exactly.
    """

    out_aspect: float = 16.0 / 9.0
    fill: float = 0.70
    min_frac: float = 0.34
    max_frac: float = 0.78  # crop is at most this fraction of the frame (always crops)
    smooth: float = 0.18  # EMA weight toward the desired crop CENTRE each frame
    size_smooth: float = 0.08  # slower EMA weight for crop SIZE (w, h) → calmer zoom
    deadzone: float = 0.04  # hold centre while desired moves < this frac of crop w/h
    size_deadband: float = 0.03  # ignore size changes under this fraction
    headroom: float = 0.10
    _crop: tuple[float, float, float, float] | None = None
    _following: bool = False  # hysteresis: True once the centre is being tracked

    # Once moving, keep following until the desired centre is back inside this
    # (tighter) fraction of the dead-zone band — prevents boundary chatter.
    _INNER_BAND_FRAC: float = 0.5

    def reset(self) -> None:
        self._crop = None
        self._following = False

    def frame_for(
        self, bbox: tuple[float, float, float, float], frame_w: int, frame_h: int
    ) -> tuple[int, int, int, int]:
        """Smoothed integer crop framing *bbox*."""
        tgt = desired_crop(
            bbox,
            frame_w,
            frame_h,
            out_aspect=self.out_aspect,
            fill=self.fill,
            min_frac=self.min_frac,
            max_frac=self.max_frac,
            headroom=self.headroom,
        )
        return self._step(tgt)

    def full_frame(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """Ease the crop back toward the whole frame (no target to follow)."""
        return self._step((0.0, 0.0, float(frame_w), float(frame_h)))

    def _step(self, tgt: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        if self._crop is None:
            self._crop = tgt
            self._following = False
            x, y, w, h = self._crop
            return (int(round(x)), int(round(y)), max(1, int(round(w))), max(1, int(round(h))))

        cx, cy, cw, ch = self._crop
        tx, ty, tw, th = tgt

        # ── Size: ease ONE aspect-locked scalar, then derive the other ──────────
        # Easing w and h through INDEPENDENT dead-bands lets one axis cross while
        # the other freezes (because cw != ch), so the crop drifts off-aspect and
        # the fixed-size output resize stretches the subject. Ease HEIGHT through
        # the dead-band + slower size constant, then DERIVE width from the output
        # aspect so the crop stays locked on out_aspect every frame.
        nh = self._ease_size(ch, th)
        nw = nh * self.out_aspect

        # ── Position (x, y): dead-zone hold + hysteresis on the crop CENTRE ─────
        # Compare the desired crop CENTRE to the held centre; the band scales with
        # the current crop size so it tracks the on-screen subject size.
        held_cx, held_cy = cx + cw * 0.5, cy + ch * 0.5
        des_cx, des_cy = tx + tw * 0.5, ty + th * 0.5
        outer_x, outer_y = self.deadzone * cw, self.deadzone * ch
        inner_x = outer_x * self._INNER_BAND_FRAC
        inner_y = outer_y * self._INNER_BAND_FRAC
        moved_x, moved_y = abs(des_cx - held_cx), abs(des_cy - held_cy)

        if self._following:
            # Keep following until BOTH axes settle back inside the inner band.
            if moved_x <= inner_x and moved_y <= inner_y:
                self._following = False
        elif moved_x > outer_x or moved_y > outer_y:
            self._following = True

        if self._following:
            a = self.smooth
            nx = cx + a * (tx - cx)
            ny = cy + a * (ty - cy)
        else:
            # Held: keep the existing top-left (the centre does not move). If the
            # size eased, re-anchor x/y so the *centre* stays put rather than the
            # corner — otherwise a size change would drift the held subject.
            nx = held_cx - nw * 0.5
            ny = held_cy - nh * 0.5

        self._crop = (nx, ny, nw, nh)
        return (int(round(nx)), int(round(ny)), max(1, int(round(nw))), max(1, int(round(nh))))

    def _ease_size(self, cur: float, tgt: float) -> float:
        """Ease one crop dimension toward *tgt*: ignore sub-dead-band changes,
        otherwise EMA at the (slower) ``size_smooth`` constant."""
        if cur > 0.0 and abs(tgt - cur) <= self.size_deadband * cur:
            return cur
        return cur + self.size_smooth * (tgt - cur)
