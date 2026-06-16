#!/usr/bin/env python3
"""Manual ingest probe — view one source and print live stats.

Usage examples::

    # USB camera at index 0
    python tools/ingest_probe.py usb 0

    # RTSP stream
    python tools/ingest_probe.py rtsp rtsp://admin:pass@192.168.1.100/stream

    # NDI source (name from NDIDiscovery)
    python tools/ingest_probe.py ndi "LAPTOP (NDI CAMERA)"

    # Run NDI/USB discovery and list available sources
    python tools/ingest_probe.py discover [--timeout 5]

Press Ctrl+C to stop.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
log = logging.getLogger("ingest_probe")


def _run_usb(index: int, target_fps: float, duration: float) -> None:
    from autoptz.engine.pipeline.ingest import USBAdapter  # noqa: PLC0415

    log.info("Probing USB camera index=%d at %.0f fps …", index, target_fps)
    adapter = USBAdapter("probe", source=index, target_fps=target_fps)
    _run_adapter(adapter, duration)


def _run_rtsp(url: str, target_fps: float, duration: float) -> None:
    from autoptz.engine.pipeline.ingest import RTSPAdapter  # noqa: PLC0415

    log.info("Probing RTSP %r at %.0f fps …", url, target_fps)
    adapter = RTSPAdapter("probe", url=url, target_fps=target_fps)
    _run_adapter(adapter, duration)


def _run_ndi(name: str, target_fps: float, duration: float) -> None:
    from autoptz.engine.pipeline.ingest import NDIAdapter  # noqa: PLC0415

    log.info("Probing NDI source %r at %.0f fps …", name, target_fps)
    adapter = NDIAdapter("probe", ndi_name=name, target_fps=target_fps)
    _run_adapter(adapter, duration)


def _run_adapter(adapter: object, duration: float) -> None:
    from autoptz.engine.pipeline.ingest import SourceAdapter  # noqa: PLC0415

    assert isinstance(adapter, SourceAdapter)

    adapter.start()
    deadline = time.monotonic() + duration
    try:
        while time.monotonic() < deadline:
            time.sleep(1.0)
            st = adapter.status
            log.info(
                "state=%-12s  fps=%5.1f  frames=%d  error=%s",
                st.state.value,
                st.fps,
                st.frames_total,
                st.last_error or "—",
            )
    except KeyboardInterrupt:
        pass
    finally:
        adapter.stop()
        log.info("Stopped.")


def _run_discover(timeout: float) -> None:
    from autoptz.engine.discovery.ndi import NDIDiscovery  # noqa: PLC0415
    from autoptz.engine.discovery.usb import USBDiscovery  # noqa: PLC0415
    from autoptz.engine.discovery.onvif import ONVIFDiscovery  # noqa: PLC0415

    log.info("Running discovery for %.0f s — plug/unplug cameras to test …", timeout)

    usb = USBDiscovery(poll_interval=2.0)
    ndi = NDIDiscovery(poll_interval=2.0)
    onvif = ONVIFDiscovery(rescan_interval=10.0)

    def on_usb(ev: str, dev: object) -> None:
        log.info("[USB]  %-8s  %s", ev, dev)

    def on_ndi(ev: str, src: object) -> None:
        log.info("[NDI]  %-8s  %s", ev, src)

    def on_onvif(ev: str, dev: object) -> None:
        log.info("[ONVIF] %-8s  %s", ev, dev)

    usb.on_change(on_usb)  # type: ignore[arg-type]
    ndi.on_change(on_ndi)  # type: ignore[arg-type]
    onvif.on_change(on_onvif)  # type: ignore[arg-type]

    usb.start()
    ndi.start()
    onvif.start()

    try:
        time.sleep(timeout)
    except KeyboardInterrupt:
        pass
    finally:
        usb.stop()
        ndi.stop()
        onvif.stop()

    log.info("USB devices   : %s", [d.index for d in usb.devices])
    log.info("NDI sources   : %s", [s.name for s in ndi.sources])
    log.info("ONVIF devices : %s", [d.host for d in onvif.devices])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="AutoPTZ v2 — ingest adapter probe tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS (default: 30)")
    parser.add_argument("--duration", type=float, default=30.0, help="Run for N seconds (default: 30)")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_usb = sub.add_parser("usb", help="Probe a USB camera by index")
    p_usb.add_argument("index", type=int, help="Device index (0, 1, 2, …)")

    p_rtsp = sub.add_parser("rtsp", help="Probe an RTSP stream")
    p_rtsp.add_argument("url", help="RTSP URL")

    p_ndi = sub.add_parser("ndi", help="Probe an NDI source by name")
    p_ndi.add_argument("name", help='NDI source name, e.g. "LAPTOP (NDI CAMERA)"')

    p_disc = sub.add_parser("discover", help="Run all discovery services")
    p_disc.add_argument(
        "--timeout", type=float, default=30.0,
        help="How long to run (default: 30 s)"
    )

    args = parser.parse_args(argv)

    if args.cmd == "usb":
        _run_usb(args.index, args.fps, args.duration)
    elif args.cmd == "rtsp":
        _run_rtsp(args.url, args.fps, args.duration)
    elif args.cmd == "ndi":
        _run_ndi(args.name, args.fps, args.duration)
    elif args.cmd == "discover":
        _run_discover(args.timeout)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
