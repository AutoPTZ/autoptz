"""Center Stage auto-framer — compute a smoothed crop that frames the target.

Driving the digital crop through the physical PTZ velocity controller did not
work: the controller's conservative auto-zoom never engaged, so the crop stayed
at the full frame (and with no zoom there is no room to pan). This computes the
crop **directly** from the target's bounding box each frame instead — centre on
the subject, size the crop so the subject fills a configured fraction of it,
match the output aspect ratio, clamp inside the frame, and EMA-smooth so it does
not jitter. That is what "Center Stage" actually does, and it always responds.
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
    headroom: float = 0.10,
) -> tuple[float, float, float, float]:
    """The crop ``(x, y, w, h)`` (pixels) that frames *bbox*.

    ``fill`` = the fraction of the crop height the subject should occupy (smaller
    fill → tighter shot). ``out_aspect`` = the output width/height (the crop keeps
    it so the resize doesn't distort). ``min_frac`` floors the crop at a fraction
    of the frame so a tiny far-away box doesn't zoom to a postage stamp.
    ``headroom`` lifts the centre slightly so the head isn't jammed at the top.
    """
    bx1, by1, bx2, by2 = (float(v) for v in bbox)
    subj_h = max(1.0, by2 - by1)
    cx = (bx1 + bx2) * 0.5
    cy = (by1 + by2) * 0.5

    fw, fh = float(frame_w), float(frame_h)
    ch = subj_h / _clamp(fill, 0.1, 1.0)
    cw = ch * out_aspect

    # Cap to the frame, keeping the output aspect.
    if cw > fw:
        cw, ch = fw, fw / out_aspect
    if ch > fh:
        ch, cw = fh, fh * out_aspect
    # Floor at min_frac of the frame height (don't over-zoom a small box).
    min_ch = max(1.0, min_frac * fh)
    if ch < min_ch:
        ch, cw = min_ch, min_ch * out_aspect
        if cw > fw:
            cw, ch = fw, fw / out_aspect

    # Lift the centre a touch for headroom, then clamp the window inside the frame.
    cy_adj = cy - headroom * ch
    x = _clamp(cx - cw * 0.5, 0.0, max(0.0, fw - cw))
    y = _clamp(cy_adj - ch * 0.5, 0.0, max(0.0, fh - ch))
    return (x, y, cw, ch)


@dataclass
class DigitalFramer:
    """Stateful EMA-smoothed wrapper around :func:`desired_crop`."""

    out_aspect: float = 16.0 / 9.0
    fill: float = 0.62
    min_frac: float = 0.34
    smooth: float = 0.15  # EMA weight toward the desired crop each frame
    headroom: float = 0.10
    _crop: tuple[float, float, float, float] | None = None

    def reset(self) -> None:
        self._crop = None

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
            headroom=self.headroom,
        )
        return self._step(tgt)

    def full_frame(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """Ease the crop back toward the whole frame (no target to follow)."""
        return self._step((0.0, 0.0, float(frame_w), float(frame_h)))

    def _step(self, tgt: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        if self._crop is None:
            self._crop = tgt
        else:
            a = self.smooth
            self._crop = tuple(c + a * (t - c) for c, t in zip(self._crop, tgt, strict=True))  # type: ignore[assignment]
        x, y, w, h = self._crop
        return (int(round(x)), int(round(y)), max(1, int(round(w))), max(1, int(round(h))))
