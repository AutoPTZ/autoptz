"""Application theme for the native Qt Widgets UI.

A single broadcast-pro palette (Wirecast/OBS lineage) rendered through the
**Fusion** style, consistent on every OS while honoring two system signals:

  * **System Light/Dark** — via ``QStyleHints.colorScheme`` (Qt 6.5+).
  * **System accent color** — captured from the platform palette's ``Highlight``
    role (brand blue ``#3b82f6`` fallback), then *toned down* for selections so
    they read clearly without glaring.

Widgets must read colors from :data:`CURRENT` (the active appearance) — never
from ``DARK`` directly — so light mode is actually light.  :class:`ThemeController`
rebinds :data:`CURRENT`/:data:`ACCENT`/:data:`SELECTION` and re-applies the global
palette + stylesheet whenever the appearance changes.

On-video HUD colors (``VIDEO_*``) and status colors are CONSTANT (drawn by
``QPainter``) so overlays stay readable on the dark video scrim in any mode.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QEvent, QObject, QPropertyAnimation, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QAbstractItemView, QComboBox, QMenu

log = logging.getLogger(__name__)

# ── themed SVG icons (chevron / checkmark) ─────────────────────────────────────
# Qt's CSS triangle (border trick) for dropdown arrows reads dated; instead we
# bake tiny stroke SVGs in the *current* palette colour and reference them from
# the stylesheet via ``image: url(...)``.  Files are written once per (name,
# colour) into a temp cache and reused, so re-applying the theme is cheap.
_ICON_DIR = Path(tempfile.gettempdir()) / "autoptz-icons"

_CHEVRON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" '
    'viewBox="0 0 16 16"><path d="M4 6.5 L8 10.5 L12 6.5" fill="none" '
    'stroke="{color}" stroke-width="1.6" stroke-linecap="round" '
    'stroke-linejoin="round"/></svg>'
)
_CHECK_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" '
    'viewBox="0 0 14 14"><path d="M2.8 7.4 L5.8 10.4 L11.2 3.8" fill="none" '
    'stroke="{color}" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round"/></svg>'
)


def _icon_url(name: str, color: str, svg_tpl: str) -> str:
    """Write a colour-baked SVG to the icon cache and return a stylesheet url().

    Returns ``""`` if the file can't be written (callers then omit the image rule
    and fall back to a plain coloured indicator), so theming never hard-fails.
    """
    try:
        _ICON_DIR.mkdir(parents=True, exist_ok=True)
        key = hashlib.md5(f"{name}:{color}".encode()).hexdigest()[:10]
        path = _ICON_DIR / f"{name}-{key}.svg"
        if not path.exists():
            path.write_text(svg_tpl.format(color=color), encoding="utf-8")
        return path.as_posix()
    except Exception:  # noqa: BLE001
        log.debug("icon write failed", exc_info=True)
        return ""


# ── brand + status constants (do NOT flip with light/dark) ─────────────────────
ACCENT_FALLBACK = "#2563eb"
ACCENT_TEXT = "#ffffff"

TRACKING = "#22c55e"
TARGET = "#4ade80"
WARNING = "#fb923c"
LOST = "#f97316"
ERROR = "#ef4444"
BBOX = "#4dd0e1"

# Semantic aliases for destructive UI (delete / remove / discard).  A single
# source of truth so every "danger" affordance reads the same — paired with the
# global ``QPushButton[danger="true"]`` rule and the DangerButton/IconButton
# components in widgets/common.py.
DANGER = ERROR
DANGER_HOVER = QColor(ERROR).lighter(112).name()
# Face-detection box + pose-skeleton overlay colours (drawn on the video scrim).
FACE_BOX = "#f59e0b"  # amber — distinct from the cyan person box / green target
POSE = "#38bdf8"  # sky-blue skeleton

VIDEO_TEXT = "#ffffff"
VIDEO_SUBTEXT = "#cbd5e1"
VIDEO_SCRIM = QColor(15, 15, 18, 204)

RADIUS = 6
RADIUS_L = 10

# Live UI-scale multiplier (rebound by ThemeController.apply); widgets and the
# stylesheet route pixel font sizes through :func:`fs` so text grows coherently.
SCALE: float = 1.0


def fs(px: float) -> int:
    """Scale a pixel font/metric by the active UI :data:`SCALE` (min 1px)."""
    return max(1, round(px * SCALE))


@dataclass(frozen=True)
class Palette:
    """Structural surface/text tokens for one appearance (dark or light)."""

    background: str
    surface: str
    surface_alt: str
    surface_hov: str
    sidebar_bg: str
    border: str
    border_hov: str
    text: str
    subtext: str
    muted: str


# Lighter than near-black so surfaces read as panels with visible separation.
DARK = Palette(
    background="#1b1b20",
    surface="#232329",
    surface_alt="#2c2c34",
    surface_hov="#383842",
    sidebar_bg="#17171b",
    border="#3c3c48",
    border_hov="#52525f",
    text="#edeef2",
    subtext="#b0b2c0",
    muted="#7c7e8e",
)
# Darker text/borders than a flat gray so panels read as distinct surfaces and
# secondary text is actually legible (the old light mode was "too gray").
LIGHT = Palette(
    background="#eceef2",
    surface="#ffffff",
    surface_alt="#e2e4ea",
    surface_hov="#d5d8e1",
    sidebar_bg="#dde0e7",
    border="#bcbecb",
    border_hov="#9a9caf",
    text="#14151c",
    subtext="#3f4150",
    muted="#6a6c7b",
)

# Live globals (rebound by ThemeController.apply); widgets read these.
CURRENT: Palette = DARK
ACCENT: QColor = QColor(ACCENT_FALLBACK)
SELECTION: str = "#2c3550"  # muted accent-tinted selection background


def _mix(c1: QColor, c2: QColor, t: float) -> QColor:
    """Blend ``c1`` toward ``c2`` by fraction ``t`` (0..1)."""
    return QColor(
        round(c1.red() + (c2.red() - c1.red()) * t),
        round(c1.green() + (c2.green() - c1.green()) * t),
        round(c1.blue() + (c2.blue() - c1.blue()) * t),
    )


def resolve_mode(app: object, mode: str) -> str:
    if mode == "light":
        return "light"
    if mode == "dark":
        return "dark"
    try:
        if app.styleHints().colorScheme() == Qt.ColorScheme.Light:  # type: ignore[attr-defined]
            return "light"
    except Exception:  # noqa: BLE001
        pass
    return "dark"


def system_accent(app: object) -> QColor:
    try:
        hi = app.palette().color(QPalette.ColorRole.Highlight)  # type: ignore[attr-defined]
        if (
            hi.isValid()
            and (max(hi.red(), hi.green(), hi.blue()) - min(hi.red(), hi.green(), hi.blue())) > 24
        ):
            return hi
    except Exception:  # noqa: BLE001
        pass
    return QColor(ACCENT_FALLBACK)


def build_qpalette(pal: Palette, accent: QColor, selection: QColor) -> QPalette:
    p = QPalette()
    bg = QColor(pal.background)
    surface = QColor(pal.surface)
    text = QColor(pal.text)

    p.setColor(QPalette.ColorRole.Window, bg)
    p.setColor(QPalette.ColorRole.WindowText, text)
    p.setColor(QPalette.ColorRole.Base, surface)
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(pal.surface_alt))
    p.setColor(QPalette.ColorRole.Text, text)
    p.setColor(QPalette.ColorRole.Button, QColor(pal.surface_alt))
    p.setColor(QPalette.ColorRole.ButtonText, text)
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(pal.surface_alt))
    p.setColor(QPalette.ColorRole.ToolTipText, text)
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor(pal.muted))
    p.setColor(QPalette.ColorRole.Mid, QColor(pal.border))
    # Toned-down selection so it doesn't glare.
    p.setColor(QPalette.ColorRole.Highlight, selection)
    p.setColor(QPalette.ColorRole.HighlightedText, text)
    p.setColor(QPalette.ColorRole.Link, accent)

    disabled = QColor(pal.muted)
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        p.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    return p


def build_stylesheet(pal: Palette, accent: QColor, selection: QColor) -> str:
    a = accent.name()
    a_hov = accent.darker(108).name()
    sel = selection.name()
    r = RADIUS

    # Colour-baked icons (fall back to the CSS triangle / a plain fill if the
    # cache can't be written).
    chevron = _icon_url("chevron", pal.subtext, _CHEVRON_SVG)
    chevron_hi = _icon_url("chevron", pal.text, _CHEVRON_SVG)
    check = _icon_url("check", "#ffffff", _CHECK_SVG)
    if chevron:
        arrow_rule = (
            f"QComboBox::down-arrow {{ image: url({chevron});"
            f" width: {fs(13)}px; height: {fs(13)}px; }}"
            f"QComboBox::down-arrow:on, QComboBox::down-arrow:hover {{"
            f" image: url({chevron_hi}); }}"
        )
    else:
        arrow_rule = (
            f"QComboBox::down-arrow {{ image: none; width: 0; height: 0;"
            f" border-left: {fs(4)}px solid transparent;"
            f" border-right: {fs(4)}px solid transparent;"
            f" border-top: {fs(5)}px solid {pal.subtext}; }}"
            f"QComboBox::down-arrow:hover {{ border-top-color: {pal.text}; }}"
        )
    check_img = f" image: url({check});" if check else ""

    return f"""
    QWidget {{ color: {pal.text}; font-size: {fs(13)}px; }}
    QMainWindow, QDialog {{ background: {pal.background}; }}
    QWidget#mainContent, QWidget#cameraWall, QWidget#cameraGridHost {{
        background: {pal.background}; }}
    QMainWindow::separator {{ background: {pal.border}; width: 4px; height: 4px; }}
    QMainWindow::separator:hover {{ background: {a}; }}

    QToolTip {{ background: {pal.surface_alt}; color: {pal.text};
        border: 1px solid {pal.border_hov}; border-radius: {r}px;
        padding: {fs(5)}px {fs(8)}px; font-size: {fs(12)}px; }}

    /* dock panels — visible frame + clear title */
    QDockWidget {{ titlebar-close-icon: none; titlebar-normal-icon: none;
        background: {pal.surface}; border: 1px solid {pal.border}; font-weight: 600; }}
    QWidget#propertiesPanel, QWidget#servicesPanel, QWidget#logsPanel,
    QWidget#cameraInfoPanel, QWidget#peoplePanel {{
        background: {pal.surface}; }}
    QDockWidget::title {{ background: {pal.sidebar_bg}; padding: {fs(8)}px {fs(12)}px;
        border-bottom: 1px solid {pal.border}; }}

    /* rounded, underlined tabs (no boxy Win95 frame) */
    QTabBar::tab {{ background: transparent; color: {pal.subtext};
        padding: {fs(7)}px {fs(16)}px; border: 1px solid transparent;
        border-top-left-radius: {r}px; border-top-right-radius: {r}px; margin-right: 2px; }}
    QTabBar::tab:selected {{ background: {pal.surface}; color: {pal.text};
        border-color: {pal.border}; border-bottom-color: {pal.surface}; }}
    QTabBar::tab:hover:!selected {{ color: {pal.text}; background: {pal.surface_alt}; }}

    QMenuBar {{ background: {pal.sidebar_bg}; padding: {fs(2)}px; }}
    QMenuBar::item {{ padding: {fs(5)}px {fs(10)}px; border-radius: {r}px; }}
    QMenuBar::item:selected {{ background: {pal.surface_hov}; }}
    QMenu {{ background: {pal.surface}; border: 1px solid {pal.border};
        border-radius: {RADIUS_L}px; padding: {fs(6)}px; }}
    QMenu::item {{ background: transparent; color: {pal.text};
        padding: {fs(7)}px {fs(28)}px {fs(7)}px {fs(14)}px; border-radius: {r}px;
        margin: 1px {fs(4)}px; }}
    QMenu::item:selected {{ background: {sel}; color: {pal.text}; }}
    QMenu::item:disabled {{ color: {pal.muted}; background: transparent; }}
    QMenu::icon {{ padding-left: {fs(6)}px; }}
    QMenu::separator {{ height: 1px; background: {pal.border}; margin: {fs(6)}px {fs(10)}px; }}
    QMenu::right-arrow {{ width: {fs(8)}px; height: {fs(8)}px; }}

    QStatusBar {{ background: {pal.sidebar_bg}; color: {pal.subtext}; }}
    QStatusBar::item {{ border: 0; }}

    QPushButton {{ background: {pal.surface_alt}; border: 1px solid {pal.border};
        border-radius: {r}px; padding: {fs(6)}px {fs(14)}px; }}
    QPushButton:hover {{ background: {pal.surface_hov}; border-color: {pal.border_hov}; }}
    QPushButton:pressed {{ background: {pal.surface}; }}
    QPushButton:disabled {{ color: {pal.muted}; border-color: {pal.border}; background: transparent; }}
    QPushButton:checked {{ background: {sel}; border-color: {a}; color: {pal.text}; }}
    QPushButton[accent="true"] {{ background: {a}; color: {ACCENT_TEXT}; border: none; font-weight: 600; }}
    QPushButton[accent="true"]:hover {{ background: {a_hov}; }}
    QPushButton[accent="true"]:disabled {{ background: {pal.surface_alt}; color: {pal.muted}; }}
    /* destructive: red text on a normal border, filling red on hover */
    QPushButton[danger="true"] {{ background: transparent; color: {ERROR}; border: 1px solid {pal.border}; }}
    QPushButton[danger="true"]:hover {{ background: {ERROR}; color: {ACCENT_TEXT}; border-color: {ERROR}; }}
    QPushButton[danger="true"]:pressed {{ background: {DANGER_HOVER}; color: {ACCENT_TEXT}; }}
    /* icon-only square button (IconButton): borderless, subtle hover fill */
    QToolButton#iconButton {{ background: transparent; border: none; border-radius: {r}px;
        padding: 0; color: {pal.subtext}; }}
    QToolButton#iconButton:hover {{ background: {pal.surface_hov}; color: {pal.text}; }}
    QToolButton#iconButton[danger="true"]:hover {{ background: {ERROR}; color: {ACCENT_TEXT}; }}

    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background: {pal.surface};
        border: 1px solid {pal.border}; border-radius: {r}px;
        padding: {fs(5)}px {fs(8)}px; min-height: {fs(20)}px;
        selection-background-color: {sel}; selection-color: {pal.text}; }}
    QComboBox {{ padding-right: {fs(4)}px; }}
    QComboBox:editable {{ background: {pal.surface}; }}
    QComboBox:on {{ border-color: {a}; }}
    QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {pal.border_hov}; }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {a}; }}
    QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right;
        border: none; border-left: 1px solid {pal.border};
        border-top-right-radius: {r}px; border-bottom-right-radius: {r}px;
        width: {fs(22)}px; }}
    {arrow_rule}
    QComboBox QAbstractItemView {{ background: {pal.surface};
        border: 1px solid {pal.border_hov}; border-radius: {r}px; padding: {fs(4)}px;
        selection-background-color: {sel}; selection-color: {pal.text}; outline: none; }}
    QComboBox QAbstractItemView::item {{ min-height: {fs(24)}px;
        padding: {fs(3)}px {fs(8)}px; border-radius: {r}px; border: none; }}
    QComboBox QAbstractItemView::item:hover {{ background: {pal.surface_hov}; }}
    QComboBox QAbstractItemView::item:selected {{ background: {sel}; color: {pal.text}; }}

    QSlider::groove:horizontal {{ height: {fs(4)}px; background: {pal.surface_alt}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {a}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ background: {pal.text}; width: {fs(14)}px; height: {fs(14)}px;
        margin: -{fs(6)}px 0; border-radius: {fs(7)}px; }}
    QSlider::handle:horizontal:hover {{ background: {a}; }}

    QScrollArea, QAbstractScrollArea {{ background: {pal.surface}; border: none; }}
    QScrollArea > QWidget > QWidget, QAbstractScrollArea > QWidget > QWidget {{
        background: {pal.surface}; }}

    QScrollBar:vertical {{ background: transparent; width: {fs(11)}px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {pal.border_hov}; border-radius: {fs(5)}px; min-height: {fs(28)}px; }}
    QScrollBar::handle:vertical:hover {{ background: {pal.muted}; }}
    QScrollBar:horizontal {{ background: transparent; height: {fs(11)}px; margin: 0; }}
    QScrollBar::handle:horizontal {{ background: {pal.border_hov}; border-radius: {fs(5)}px; min-width: {fs(28)}px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}

    QHeaderView::section {{ background: {pal.sidebar_bg}; color: {pal.subtext};
        border: 0; border-bottom: 1px solid {pal.border}; padding: {fs(5)}px {fs(8)}px; }}
    QTableView, QListView, QTreeView {{ background: {pal.surface};
        alternate-background-color: {pal.surface_alt}; border: 1px solid {pal.border};
        gridline-color: {pal.border}; selection-background-color: {sel};
        selection-color: {pal.text}; outline: none; }}
    QTableView::viewport, QListView::viewport, QTreeView::viewport {{
        background: {pal.surface}; }}
    QCheckBox {{ color: {pal.text}; spacing: {fs(7)}px; }}
    QCheckBox::indicator {{ width: {fs(16)}px; height: {fs(16)}px;
        border-radius: {fs(4)}px; border: 1px solid {pal.border_hov};
        background: {pal.surface}; }}
    QCheckBox::indicator:hover {{ border-color: {a}; }}
    QCheckBox::indicator:checked {{ background: {a}; border-color: {a};{check_img} }}
    QCheckBox::indicator:disabled {{ border-color: {pal.border}; background: {pal.surface_alt}; }}

    /* main toolbar + per-dock title bar with the lock toggle (main_window.py) */
    QToolBar#mainToolbar {{ background: {pal.sidebar_bg}; border: none;
        border-bottom: 1px solid {pal.border}; padding: {fs(3)}px; spacing: {fs(4)}px; }}
    QToolBar#mainToolbar QToolButton {{ background: transparent; border: 1px solid transparent;
        border-radius: {r}px; padding: {fs(4)}px {fs(10)}px; }}
    QToolBar#mainToolbar QToolButton:hover {{ background: {pal.surface_hov}; }}
    QToolBar#mainToolbar QToolButton:checked {{ background: {sel}; border-color: {a}; }}
    QWidget#dockTitleBar {{ background: {pal.sidebar_bg};
        border-bottom: 1px solid {pal.border}; }}

    /* collapsible section card + header (common.py CollapsibleGroup) */
    QWidget#collGroup {{ background: {pal.surface}; border: 1px solid {pal.border};
        border-radius: {r}px; }}
    QToolButton#collGroupHeader {{ border: none; border-bottom: 1px solid {pal.border};
        background: {pal.sidebar_bg}; text-align: left;
        border-top-left-radius: {r}px; border-top-right-radius: {r}px;
        padding: {fs(7)}px {fs(10)}px; font-size: {fs(10)}px;
        font-weight: 700; letter-spacing: 1px; color: {pal.subtext}; }}
    QToolButton#collGroupHeader:hover {{ color: {pal.text}; }}

    /* palette-driven shared widgets (common.py) — track light/dark for free */
    QLabel#sectionCaption {{ color: {pal.subtext}; font-size: {fs(10)}px;
        font-weight: 700; letter-spacing: 1px; }}
    QFrame#hline {{ background: {pal.border}; border: none; }}
    QLabel#helpBadge {{ color: {pal.muted}; background: {pal.surface_alt};
        border: 1px solid {pal.border}; border-radius: {fs(8)}px;
        font-size: {fs(10)}px; font-weight: 700; }}
    QLabel#helpBadge:hover {{ color: {ACCENT_TEXT}; background: {a}; border-color: {a}; }}

    /* AutoPTZ Mark HUD — chart card, control bar, details panel (mark_window.py) */
    QWidget#markChart {{ background: {pal.surface}; border: 1px solid {pal.border};
        border-radius: {r}px; }}
    QFrame#chartCard {{ background: {pal.surface}; border: 1px solid {pal.border};
        border-radius: {r}px; }}
    QWidget#markControlPanel {{ background: {pal.surface_alt};
        border-top: 1px solid {pal.border}; }}
    QLabel#markVerdict {{ color: {pal.text}; font-size: {fs(14)}px; font-weight: 600; }}
    QLabel#detailsHeader, QLabel#chartTitle {{ background: {pal.sidebar_bg};
        color: {pal.subtext}; padding: {fs(6)}px {fs(10)}px; font-size: {fs(10)}px;
        font-weight: 700; letter-spacing: 1px; }}
    """


def _is_popup_window(win: object) -> bool:
    try:
        return (
            win.windowFlags() & Qt.WindowType.WindowType_Mask  # type: ignore[attr-defined]
        ) == Qt.WindowType.Popup
    except Exception:  # noqa: BLE001
        return False


def _round_popup_window(win: object, margin: int) -> None:
    """Make a popup's top-level window translucent so its rounded fill shows.

    Qt paints a popup's *window* as an opaque rectangle, so a rounded
    ``border``/``border-radius`` from the stylesheet draws over square corners
    (rounded border, square fill).  The fix is the standard Qt recipe: give the
    window a translucent background and a frameless, shadowless flag set, and
    add a small contents margin so the stylesheet's rounded surface paints
    *inside* the transparent window — leaving the corners genuinely empty.
    """
    if not _is_popup_window(win):
        return
    try:
        win.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)  # type: ignore[attr-defined]
        win.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)  # type: ignore[attr-defined]
        win.setWindowFlag(Qt.WindowType.NoDropShadowWindowHint, True)  # type: ignore[attr-defined]
        win.setContentsMargins(margin, margin, margin, margin)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        log.debug("round popup window failed", exc_info=True)


def _fade_popup_window(win: object) -> None:
    """Short, non-blocking fade-in for popup windows."""
    try:
        anim = QPropertyAnimation(win, b"windowOpacity", win)  # type: ignore[arg-type]
        anim.setDuration(90)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        win._autoptz_popup_fade_anim = anim
        win.setWindowOpacity(0.0)  # type: ignore[attr-defined]
        anim.start()
    except Exception:  # noqa: BLE001
        log.debug("popup fade failed", exc_info=True)


class _PopupRounder(QObject):
    """App-wide filter that rounds QMenu + QComboBox popup backgrounds.

    Popups are created lazily and re-created across shows, so a one-shot fix
    can't reach them; this filter applies the translucent-window recipe (see
    :func:`_round_popup_window`) the moment each popup is polished/shown.  Kept
    here so the rounded-popup behaviour lives entirely in the theme module.
    """

    def eventFilter(self, obj: object, event: object) -> bool:  # noqa: N802
        et = event.type()
        if et in (QEvent.Type.Show, QEvent.Type.Polish):
            if isinstance(obj, QMenu):
                _round_popup_window(obj, 0)
                if et == QEvent.Type.Show:
                    _fade_popup_window(obj)
            elif isinstance(obj, QComboBox):
                view = obj.view()
                win = view.window() if view is not None else None
                if win is not None and win is not obj and _is_popup_window(win):
                    # The container window owns the square corners; round it and
                    # let the QAbstractItemView stylesheet draw the rounded fill.
                    _round_popup_window(win, 0)
            elif isinstance(obj, QAbstractItemView):
                win = obj.window()
                if win is not None and _is_popup_window(win):
                    _round_popup_window(win, 0)
                    if et == QEvent.Type.Show:
                        _fade_popup_window(win)
        return False


class ThemeController(QObject):
    """Applies the theme to a ``QApplication`` and keeps it in sync."""

    # Discrete UI-scale steps offered in the View menu / cycled by shortcuts.
    SCALE_STEPS = (0.9, 1.0, 1.1, 1.25, 1.5)

    def __init__(self, app: object, client: object) -> None:
        super().__init__()
        self._app = app
        self._client = client
        self._accent = system_accent(app)
        self._mode = "dark"
        self._scale = 1.0
        try:
            self._mode = str(client.themeMode)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        try:
            self._scale = float(client.getSetting("ui_scale", 1.0)) or 1.0  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        try:
            self._base_point = float(app.font().pointSizeF())  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            self._base_point = 13.0
        if self._base_point <= 0:
            self._base_point = 13.0
        try:
            app.setStyle("Fusion")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            log.debug("Fusion style unavailable", exc_info=True)
        # Round QMenu / combo-popup backgrounds (translucent-window recipe) so
        # rounded borders no longer sit over square fills.  Held as an attribute
        # so the filter isn't garbage-collected.
        self._popup_rounder = _PopupRounder(self)
        try:
            app.installEventFilter(self._popup_rounder)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            log.debug("popup rounder install failed", exc_info=True)
        try:
            app.styleHints().colorSchemeChanged.connect(lambda _s: self.apply())  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        try:
            client.themeChanged.connect(self._on_mode_changed)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        self.apply()

    def _on_mode_changed(self, mode: str) -> None:
        self._mode = mode or "dark"
        self.apply()

    @property
    def scale(self) -> float:
        return self._scale

    def set_scale(self, factor: float) -> None:
        """Set the UI scale (clamped), persist it, and re-apply the theme."""
        self._scale = max(0.5, min(3.0, float(factor)))
        try:
            self._client.setSetting("ui_scale", self._scale)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        self.apply()

    def nudge_scale(self, direction: int) -> None:
        """Step to the next/prev discrete scale (``+1``/``-1``); ``0`` resets to 100%."""
        if direction == 0:
            self.set_scale(1.0)
            return
        steps = self.SCALE_STEPS
        # Snap to the nearest defined step, then move one notch.
        idx = min(range(len(steps)), key=lambda i: abs(steps[i] - self._scale))
        idx = max(0, min(len(steps) - 1, idx + (1 if direction > 0 else -1)))
        self.set_scale(steps[idx])

    def apply(self) -> None:
        global CURRENT, ACCENT, SELECTION, SCALE
        appearance = resolve_mode(self._app, self._mode)
        pal = DARK if appearance == "dark" else LIGHT
        # Tone the accent into a calm selection color over the surface.
        selection = _mix(self._accent, QColor(pal.surface), 0.62)
        CURRENT = pal
        ACCENT = self._accent
        SELECTION = selection.name()
        SCALE = self._scale
        try:
            font = self._app.font()  # type: ignore[attr-defined]
            font.setPointSizeF(self._base_point * self._scale)
            self._app.setFont(font)  # type: ignore[attr-defined]
            self._app.setPalette(build_qpalette(pal, self._accent, selection))  # type: ignore[attr-defined]
            self._app.setStyleSheet(build_stylesheet(pal, self._accent, selection))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            log.debug("Failed to apply theme", exc_info=True)
