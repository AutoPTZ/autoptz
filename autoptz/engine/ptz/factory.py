"""Build a concrete :class:`PTZBackend` from a :class:`PTZConfig`.

The single entry point :func:`build_backend` maps the config's ``backend`` field
to a backend instance:

  ``visca_usb`` → :class:`ViscaUSBBackend` (serial port from ``address``)
  ``visca_ip``  → :class:`ViscaIPBackend`  (``host[:port]`` from ``address``)
  ``ndi``       → :class:`NDIPTZBackend`   (requires an NDI ``receiver``)
  ``onvif``     → :class:`ONVIFBackend`    (``host[:port]`` + creds)
  ``auto``      → probe: NDI source → NDI PTZ; else ONVIF / VISCA-IP by address.

It **never raises**.  A missing optional dependency, an unparseable address, or
an unreachable device is logged and yields ``None`` (manual PTZ then no-ops and
auto control is simply disabled).  ``None``/``""``/``auto`` with nothing probable
also returns ``None``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from autoptz.config.models import PTZConfig
    from autoptz.engine.ptz.base import PTZBackend

log = logging.getLogger(__name__)

# Default ports per transport.
_DEFAULT_VISCA_IP_PORT = 52381
_DEFAULT_ONVIF_PORT = 80


def build_backend(
    ptz: PTZConfig,
    *,
    ndi_source: Any | None = None,
    ndi_name: str | None = None,
) -> PTZBackend | None:
    """Construct the PTZ backend described by *ptz*, or ``None``.

    Args:
        ptz:        The camera's :class:`PTZConfig`.
        ndi_source: An already-connected NDI receiver (``cyndilib.Receiver``) to
                    share for NDI PTZ.
        ndi_name:   NDI source name; when given (and no ``ndi_source``), the NDI
                    backend opens its own low-bandwidth PTZ receiver to it.  This
                    is what makes NDI cameras controllable without threading the
                    video receiver through (the video adapter keeps its own).

    Returns:
        A :class:`PTZBackend` instance, or ``None`` when unconfigured / the
        device or optional dependency is unavailable.  Never raises.
    """
    if ptz is None:  # pragma: no cover - defensive
        return None

    backend = (ptz.backend or "").strip().lower()

    try:
        if backend in ("", "auto"):
            return _probe_auto(ptz, ndi_source=ndi_source, ndi_name=ndi_name)
        if backend == "visca_usb":
            return _build_visca_usb(ptz)
        if backend == "visca_ip":
            return _build_visca_ip(ptz)
        if backend == "ndi":
            return _build_ndi(ptz, ndi_source=ndi_source, ndi_name=ndi_name)
        if backend == "onvif":
            return _build_onvif(ptz)
        if backend == "digital":
            return _build_digital(ptz)
    except Exception:  # noqa: BLE001 - factory must never raise into the engine
        log.warning(
            "PTZ backend %r build failed (addr=%r); PTZ disabled.",
            backend,
            ptz.address,
            exc_info=True,
        )
        return None

    log.warning("Unknown PTZ backend %r; PTZ disabled.", backend)
    return None


# ── per-backend builders ────────────────────────────────────────────────────────


def _build_visca_usb(ptz: PTZConfig) -> PTZBackend | None:
    from autoptz.engine.ptz.visca_usb import ViscaUSBBackend

    port = (ptz.address or "").strip()
    if not port:
        log.warning("visca_usb requested but no serial port in address; PTZ disabled.")
        return None
    try:
        backend = ViscaUSBBackend(port)
    except Exception:  # noqa: BLE001 - pyserial missing / port not present
        log.warning("ViscaUSB %s unavailable; PTZ disabled.", port, exc_info=True)
        return None
    log.info("PTZ backend: VISCA-USB on %s", port)
    return backend


def _build_visca_ip(ptz: PTZConfig) -> PTZBackend | None:
    from autoptz.engine.ptz.visca_ip import ViscaIPBackend

    host, port = _split_host_port(ptz.address, _DEFAULT_VISCA_IP_PORT)
    if not host:
        log.warning("visca_ip requested but no host in address; PTZ disabled.")
        return None
    try:
        backend = ViscaIPBackend(host, port)
    except Exception:  # noqa: BLE001 - unreachable device / connection refused
        log.warning("ViscaIP %s:%d unreachable; PTZ disabled.", host, port, exc_info=True)
        return None
    log.info("PTZ backend: VISCA-IP on %s:%d", host, port)
    return backend


def _build_ndi(
    ptz: PTZConfig, *, ndi_source: Any | None, ndi_name: str | None = None
) -> PTZBackend | None:
    name = (ndi_name or "").strip()
    if ndi_source is None and not name:
        log.warning("ndi PTZ requested but no NDI source/name supplied; PTZ disabled.")
        return None
    from autoptz.engine.ptz.ndi_ptz import NDIPTZBackend

    try:
        backend = NDIPTZBackend(ndi_name=name, receiver=ndi_source)
    except Exception:  # noqa: BLE001 - cyndilib missing / source not found
        log.warning(
            "NDI PTZ unavailable (cyndilib missing or source %r not found); PTZ disabled.",
            name,
            exc_info=True,
        )
        return None
    log.info("PTZ backend: NDI PTZ (%s)", name or "shared receiver")
    return backend


def _build_onvif(ptz: PTZConfig) -> PTZBackend | None:
    from autoptz.engine.ptz.onvif_ptz import ONVIFPTZBackend

    host, port = _split_host_port(ptz.address, _DEFAULT_ONVIF_PORT)
    if not host:
        log.warning("onvif requested but no host in address; PTZ disabled.")
        return None
    username, password = _split_credentials(ptz)
    try:
        backend = ONVIFPTZBackend(host, port, username=username, password=password)
    except Exception:  # noqa: BLE001 - onvif-zeep missing / device unreachable
        log.warning("ONVIF %s:%d unavailable; PTZ disabled.", host, port, exc_info=True)
        return None
    log.info("PTZ backend: ONVIF on %s:%d", host, port)
    return backend


def _build_digital(ptz: PTZConfig) -> PTZBackend | None:
    from autoptz.engine.ptz.digital import DigitalPTZBackend

    log.info("PTZ backend: digital (Center Stage)")
    return DigitalPTZBackend()


# ── auto probe ──────────────────────────────────────────────────────────────────


def _probe_auto(
    ptz: PTZConfig, *, ndi_source: Any | None, ndi_name: str | None = None
) -> PTZBackend | None:
    """Best-effort backend discovery for ``backend="auto"``.

    Order:
      1. An NDI source/name → NDI PTZ (own or shared receiver).
      2. An address present → try ONVIF, then VISCA-IP.
      3. Nothing probable → ``None`` (manual PTZ no-ops; auto control disabled).
    """
    if ndi_source is not None or (ndi_name or "").strip():
        ndi = _build_ndi(ptz, ndi_source=ndi_source, ndi_name=ndi_name)
        if ndi is not None:
            return ndi

    addr = (ptz.address or "").strip()
    if not addr:
        log.debug("PTZ auto-probe: no NDI source and no address; PTZ disabled.")
        return None

    # Looks like a host/IP → try networked PTZ.  ONVIF first (richer), then VISCA-IP.
    onvif = _build_onvif(ptz)
    if onvif is not None:
        return onvif

    visca = _build_visca_ip(ptz)
    if visca is not None:
        return visca

    log.info("PTZ auto-probe found no reachable backend at %s; PTZ disabled.", addr)
    return None


# ── address parsing ───────────────────────────────────────────────────────────────


def _split_host_port(address: str | None, default_port: int) -> tuple[str, int]:
    """Parse ``host``/``host:port``/``scheme://host:port`` → (host, port).

    Returns ``("", default_port)`` when no host can be parsed.  Never raises.
    """
    raw = (address or "").strip()
    if not raw:
        return "", default_port
    # Strip any scheme (e.g. "onvif://", "visca://", "tcp://").
    if "://" in raw:
        raw = raw.split("://", 1)[1]
    # Drop any trailing path / query.
    raw = raw.split("/", 1)[0]
    if ":" in raw:
        host, _, port_s = raw.rpartition(":")
        try:
            return host, int(port_s)
        except ValueError:
            return raw, default_port
    return raw, default_port


def _split_credentials(ptz: PTZConfig) -> tuple[str, str]:
    """Pull ONVIF username/password off the config if present (best-effort)."""
    username = getattr(ptz, "username", None) or "admin"
    password = getattr(ptz, "password", None) or ""
    return str(username), str(password)
