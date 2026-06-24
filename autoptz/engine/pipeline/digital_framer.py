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
    """Stateful EMA-smoothed wrapper around :func:`desired_crop`."""

    out_aspect: float = 16.0 / 9.0
    fill: float = 0.70
    min_frac: float = 0.34
    max_frac: float = 0.78  # crop is at most this fraction of the frame (always crops)
    smooth: float = 0.18  # EMA weight toward the desired crop each frame
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
        else:
            a = self.smooth
            self._crop = tuple(  # type: ignore[assignment]
                c + a * (t - c) for c, t in zip(self._crop, tgt, strict=True)
            )
        x, y, w, h = self._crop
        return (int(round(x)), int(round(y)), max(1, int(round(w))), max(1, int(round(h))))
