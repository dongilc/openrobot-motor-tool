"""
Shared plot styling — VESC-Tool compatible dark theme.
Colors extracted from vedderb/vesc_tool utility.cpp mAppColors.
"""

import pyqtgraph as pg
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtCore import Qt


# ── VESC-Tool color palette ─────────────────────────────────────────
GRAPH_COLORS = [
    "#4D7FC4",  # plot_graph1  (77,127,196)  blue
    "#C83434",  # plot_graph2  (200,52,52)   red
    "#7FC87F",  # plot_graph3  (127,200,127) green
    "#CE7D2C",  # plot_graph4  (206,125,44)  orange
    "#D2D27F",  # plot_graph5  (210,210,127) yellow
    "#4FCBCB",  # plot_graph6  (79,203,203)  cyan
    "#9D7FD2",  # plot_graph7  (157,127,210) purple
    "#81D4FA",  # plot_graph8  (129,212,250) light blue
    "#B4B4B4",  # plot_graph9  (180,180,180) grey
    "#DB628B",  # plot_graph10 (219,98,139)  pink
    "#FAFAC8",  # plot_graph11 (250,250,200) pale yellow
]

BG_LIGHTEST  = "#505050"   # (80,80,80)
BG_LIGHT     = "#424242"   # (66,66,66)
BG_NORMAL    = "#303030"   # (48,48,48)
BG_DARK      = "#272727"   # (39,39,39)

TEXT_NORMAL   = "#B4B4B4"  # (180,180,180)
TEXT_LIGHT    = "#DCDCDC"  # (220,220,220)
TEXT_DISABLED = "#7F7F7F"  # (127,127,127)

ACCENT_LIGHT = "#81D4FA"   # (129,212,250)
ACCENT_DARK  = "#477589"   # (71,117,137)

# Aliases for backward compatibility
COLORS = {
    "blue":    GRAPH_COLORS[0],
    "red":     GRAPH_COLORS[1],
    "green":   GRAPH_COLORS[2],
    "orange":  GRAPH_COLORS[3],
    "yellow":  GRAPH_COLORS[4],
    "cyan":    GRAPH_COLORS[5],
    "purple":  GRAPH_COLORS[6],
    "lblue":   GRAPH_COLORS[7],
    "grey":    GRAPH_COLORS[8],
    "pink":    GRAPH_COLORS[9],
    "pyellow": GRAPH_COLORS[10],
    "white":   TEXT_LIGHT,
    "dimmed":  TEXT_DISABLED,
}

PLOT_BG      = BG_DARK
GRID_COLOR   = BG_LIGHTEST
AXIS_COLOR   = TEXT_NORMAL
BORDER_COLOR = BG_LIGHTEST

_AXIS_FONT = None

def _get_axis_font():
    global _AXIS_FONT
    if _AXIS_FONT is None:
        _AXIS_FONT = QFont("Segoe UI", 9)
    return _AXIS_FONT


def apply_global_theme():
    """Call once at app startup."""
    # OpenGL disabled — causes framebuffer context corruption on
    # widget resize/collapse that cascades into PCAN USB failures.
    # Software rendering with downsampling + clipToView is fast enough.
    pg.setConfigOptions(
        antialias=False,
        useOpenGL=False,
        enableExperimental=False,
        background=QColor(BG_NORMAL),
        foreground=QColor(TEXT_NORMAL),
    )


def style_plot(pw: pg.PlotWidget, title: str = "",
               left_label: str = "", left_unit: str = "",
               bottom_label: str = "Time", bottom_unit: str = "s",
               show_grid: bool = True):
    """Apply VESC-Tool style to a PlotWidget. Lightweight — no heavy effects."""
    plot = pw.getPlotItem()

    # Background
    pw.setBackground(QColor(BG_DARK))

    # Title
    if title:
        plot.setTitle(title, color=TEXT_LIGHT, size="10pt")

    # Axis styling — minimal
    font = _get_axis_font()
    for axis_name in ["bottom", "left", "right"]:
        axis = plot.getAxis(axis_name)
        axis.setPen(pg.mkPen(BG_LIGHTEST, width=1))
        axis.setTextPen(pg.mkPen(TEXT_NORMAL))
        axis.setTickFont(font)
        axis.setStyle(tickLength=-6, autoExpandTextSpace=True)

    if left_label:
        plot.setLabel("left", left_label, left_unit,
                       **{"color": TEXT_NORMAL, "font-size": "9pt"})
    if bottom_label:
        plot.setLabel("bottom", bottom_label, bottom_unit,
                       **{"color": TEXT_NORMAL, "font-size": "9pt"})

    # Grid — subtle dotted lines like VESC-Tool
    if show_grid:
        plot.showGrid(x=True, y=True, alpha=0.12)

    # Performance: clip to view + downsampling
    pw.setClipToView(True)
    pw.setDownsampling(auto=True, mode='peak')

    # Thin border
    pw.setStyleSheet(
        f"border: 1px solid {BG_LIGHTEST}; border-radius: 2px;"
    )

    return plot


def make_pen(color_key: str, width: float = 1.5) -> pg.mkPen:
    """Create a pen from palette key or color index."""
    if isinstance(color_key, int):
        c = GRAPH_COLORS[color_key % len(GRAPH_COLORS)]
    else:
        c = COLORS.get(color_key, color_key)
    return pg.mkPen(QColor(c), width=width)


def graph_pen(index: int, width: float = 1.5) -> pg.mkPen:
    """Create a pen by graph color index (0-10), matching VESC-Tool ordering."""
    c = GRAPH_COLORS[index % len(GRAPH_COLORS)]
    return pg.mkPen(QColor(c), width=width)


def make_fill_brush(color_key: str, alpha: int = 40):
    """Create a semi-transparent fill brush."""
    if isinstance(color_key, int):
        c = QColor(GRAPH_COLORS[color_key % len(GRAPH_COLORS)])
    else:
        c = QColor(COLORS.get(color_key, color_key))
    c.setAlpha(alpha)
    return c


def make_gradient_brush(*args, **kwargs):
    """Removed for performance — returns None."""
    return None


class Crosshair:
    """Lightweight crosshair — only for static/non-streaming plots."""

    def __init__(self, pw: pg.PlotWidget):
        self._pw = pw
        self._vline = pg.InfiniteLine(angle=90, movable=False,
                                       pen=pg.mkPen(TEXT_DISABLED, width=1,
                                                     style=Qt.PenStyle.DashLine))
        self._hline = pg.InfiniteLine(angle=0, movable=False,
                                       pen=pg.mkPen(TEXT_DISABLED, width=1,
                                                     style=Qt.PenStyle.DashLine))
        pw.addItem(self._vline, ignoreBounds=True)
        pw.addItem(self._hline, ignoreBounds=True)

        self._label = pg.TextItem(color=TEXT_LIGHT, anchor=(0, 1))
        self._label.setFont(QFont("Consolas", 8))
        pw.addItem(self._label, ignoreBounds=True)

        pw.scene().sigMouseMoved.connect(self._on_move)

    def _on_move(self, pos):
        vb = self._pw.getPlotItem().vb
        if self._pw.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            self._vline.setPos(x)
            self._hline.setPos(y)
            self._label.setText(f"  x={x:.2f}  y={y:.2f}")
            self._label.setPos(x, y)


def style_legend(pw: pg.PlotWidget):
    """Add and style a legend — VESC-Tool style, with click-to-toggle."""
    legend = pw.addLegend(offset=(10, 10), labelTextColor=TEXT_LIGHT)
    legend.setBrush(QColor(0, 0, 0, 100))
    legend.setPen(pg.mkPen(BG_LIGHTEST))
    legend.setScale(0.8)

    # Monkey-patch addItem so every legend entry becomes click-to-toggle
    _orig_addItem = legend.addItem

    def _patched_addItem(item, name):
        _orig_addItem(item, name)
        if legend.items:
            sample, label = legend.items[-1]
            _install_legend_toggle(item, sample, label)

    legend.addItem = _patched_addItem
    return legend


def _install_legend_toggle(plot_item, sample, label):
    """Make a single legend entry click-to-toggle its curve."""
    DIM = 0.3

    def _on_click(ev):
        vis = plot_item.isVisible()
        plot_item.setVisible(not vis)
        opacity = DIM if vis else 1.0
        sample.setOpacity(opacity)
        label.setOpacity(opacity)

    sample.mouseClickEvent = _on_click
    label.mouseClickEvent = _on_click


def set_curve_visible(legend, curve, visible: bool):
    """Programmatically set curve visibility AND sync legend opacity."""
    curve.setVisible(visible)
    for sample, label in legend.items:
        if sample.item is curve:
            sample.setOpacity(1.0 if visible else 0.3)
            label.setOpacity(1.0 if visible else 0.3)
            break
