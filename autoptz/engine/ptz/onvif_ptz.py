"""ONVIF PTZ backend via onvif-zeep.

Requires: pip install onvif-zeep
  (optional — backend degrades with ImportError if not installed)

Implements:
  ContinuousMove  — velocity pan/tilt/zoom
  AbsoluteMove    — normalized [-1,1] / [0,1] position
  Stop            — halt pan/tilt and zoom independently
  GotoPreset      — by integer slot (1-based, as ONVIF spec requires)
  SetPreset       — store current position as named preset
  GetStatus       — PTZState with current position
"""

from __future__ import annotations

import logging
from typing import Any

from autoptz.engine.ptz.base import PTZBackend, PTZCaps, PTZState

log = logging.getLogger(__name__)


def _require_onvif() -> Any:
    try:
        from onvif import ONVIFCamera

        return ONVIFCamera
    except ImportError as exc:
        raise ImportError(
            "onvif-zeep is required for ONVIF PTZ control.  Install it: pip install onvif-zeep"
        ) from exc


class ONVIFPTZBackend(PTZBackend):
    """ONVIF PTZ backend.

    Args:
        host:     Camera IP or hostname.
        port:     ONVIF device service port (default 80).
        username: ONVIF authentication username.
        password: ONVIF authentication password.
        profile:  PTZ profile token.  If None, the first profile is used.
        wsdl_dir: Path to WSDL files shipped with onvif-zeep (optional).
    """

    def __init__(
        self,
        host: str,
        port: int = 80,
        username: str = "admin",
        password: str = "",
        profile: str | None = None,
        wsdl_dir: str | None = None,
    ) -> None:
        super().__init__()
        ONVIFCamera = _require_onvif()

        kwargs: dict[str, Any] = {}
        if wsdl_dir:
            kwargs["wsdl_dir"] = wsdl_dir

        cam = ONVIFCamera(host, port, username, password, **kwargs)
        self._ptz = cam.create_ptz_service()
        self._media = cam.create_media_service()

        # Pick profile
        profiles = self._media.GetProfiles()
        if not profiles:
            raise RuntimeError("ONVIF camera returned no media profiles")
        if profile is not None:
            matching = [p for p in profiles if p.token == profile]
            if not matching:
                raise ValueError(f"ONVIF profile {profile!r} not found")
            self._profile_token: str = matching[0].token
        else:
            self._profile_token = profiles[0].token

        # Probe capabilities
        try:
            caps_resp = self._ptz.GetConfigurationOptions(
                {"ConfigurationToken": self._profile_token}
            )
            has_abs = caps_resp.Spaces is not None
        except Exception:
            has_abs = False

        self.caps = PTZCaps(
            continuous_pan_tilt=True,
            continuous_zoom=True,
            absolute_pan_tilt=has_abs,
            absolute_zoom=has_abs,
            native_presets=True,
            query_position=True,
        )
        log.info("ONVIFPTZBackend ready: %s:%d profile=%r", host, port, self._profile_token)

    # ── PTZBackend interface ──────────────────────────────────────────────────

    def move_velocity(self, pan: float, tilt: float, zoom: float = 0.0) -> None:
        request = self._ptz.create_type("ContinuousMove")
        request.ProfileToken = self._profile_token
        request.Velocity = {
            "PanTilt": {"x": pan, "y": tilt},
            "Zoom": {"x": zoom},
        }
        self._ptz.ContinuousMove(request)

    def move_absolute(self, pan: float, tilt: float, zoom: float) -> None:
        request = self._ptz.create_type("AbsoluteMove")
        request.ProfileToken = self._profile_token
        request.Position = {
            "PanTilt": {"x": pan, "y": tilt},
            "Zoom": {"x": zoom},
        }
        self._ptz.AbsoluteMove(request)

    def stop(self) -> None:
        try:
            self._ptz.Stop(
                {
                    "ProfileToken": self._profile_token,
                    "PanTilt": True,
                    "Zoom": True,
                }
            )
        except Exception:
            pass

    def get_position(self) -> PTZState | None:
        try:
            resp = self._ptz.GetStatus({"ProfileToken": self._profile_token})
            pos = resp.Position
            return PTZState(
                pan=float(pos.PanTilt.x),
                tilt=float(pos.PanTilt.y),
                zoom=float(pos.Zoom.x),
            )
        except Exception:
            return None

    def goto_preset(self, idx: int) -> None:
        # ONVIF preset tokens are 1-based strings
        request = self._ptz.create_type("GotoPreset")
        request.ProfileToken = self._profile_token
        request.PresetToken = str(idx)
        self._ptz.GotoPreset(request)

    def save_preset(self, idx: int) -> None:
        request = self._ptz.create_type("SetPreset")
        request.ProfileToken = self._profile_token
        request.PresetToken = str(idx)
        request.PresetName = f"Preset {idx}"
        self._ptz.SetPreset(request)

    def home(self) -> None:
        """Recall the ONVIF home position via ``GotoHomePosition`` (best effort).

        Cameras that have no home position configured raise; we swallow it so the
        call stays a safe no-op.
        """
        try:
            request = self._ptz.create_type("GotoHomePosition")
            request.ProfileToken = self._profile_token
            self._ptz.GotoHomePosition(request)
        except Exception:
            log.debug("ONVIF GotoHomePosition unsupported/failed", exc_info=True)

    def osd_menu(self) -> None:
        """ONVIF has no standard OSD-menu operation — safe no-op."""
        log.debug("ONVIF backend has no OSD-menu command; ignoring osd_menu()")

    def close(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
        log.info("ONVIFPTZBackend closed")
