import pyqtgraph as pg
from PyQt6.QtCore import QObject, QEvent, Qt


CURSOR_COLOR = "#00e5ff"
_READOUT_BG = pg.mkBrush(0, 0, 0, 190)
_READOUT_BORDER = pg.mkPen(CURSOR_COLOR, width=1)


def _format_time(ns: float) -> str:
    ns = abs(ns)
    if ns >= 10_000_000_000:
        return f"{ns / 1_000_000_000:.2f}s"
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.4g}ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.4g}µs"
    return f"{ns:.4g}ns"


def _format_freq(hz: float) -> str:
    if hz >= 1_000_000:
        return f"{hz / 1_000_000:.4g}MHz"
    if hz >= 1_000:
        return f"{hz / 1_000:.4g}kHz"
    return f"{hz:.4g}Hz"


def _format_volt(v: float) -> str:
    v = abs(v)
    if v < 1.0:
        return f"{v * 1000:.4g}mV"
    return f"{v:.4g}V"


class CursorOverlay(QObject):
    """Measurement-cursor overlay for a pyqtgraph PlotWidget.

    When enabled, a crosshair tracks the mouse as a visual aid. Click-dragging
    on the plot defines a measurement region; the overlay reports the duration
    (Δt), the voltage span (ΔV) and the frequency (1/Δt) of that region.

    X is interpreted in buffer-sample units (matching the plot's X axis) and is
    converted to time via ``ns_per_sample``. Y is interpreted directly in volts.
    """

    def __init__(self, plot_widget):
        super().__init__(plot_widget)
        self._plot_widget = plot_widget
        self._vb = plot_widget.getPlotItem().getViewBox()
        self._enabled = False
        self._ns_per_sample = 0.0

        self._dragging = False
        self._x0 = self._y0 = 0.0
        self._x1 = self._y1 = 0.0
        self._has_selection = False

        pen = pg.mkPen(CURSOR_COLOR, width=1)
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        sel_pen = pg.mkPen(CURSOR_COLOR, width=1, style=Qt.PenStyle.DashLine)
        self._sel_v0 = pg.InfiniteLine(angle=90, movable=False, pen=sel_pen)
        self._sel_v1 = pg.InfiniteLine(angle=90, movable=False, pen=sel_pen)
        self._sel_h0 = pg.InfiniteLine(angle=0, movable=False, pen=sel_pen)
        self._sel_h1 = pg.InfiniteLine(angle=0, movable=False, pen=sel_pen)
        self._region = pg.LinearRegionItem(
            orientation="vertical", movable=False, brush=pg.mkBrush(0, 229, 255, 28)
        )
        self._region.setZValue(5)
        self._readout = pg.TextItem(
            anchor=(0, 1), color="#ffffff", fill=_READOUT_BG, border=_READOUT_BORDER
        )
        self._readout.setZValue(200)

        for item in (self._region, self._sel_v0, self._sel_v1, self._sel_h0,
                     self._sel_h1, self._vline, self._hline, self._readout):
            item.setZValue(max(item.zValue(), 150))
            item.setVisible(False)
            plot_widget.addItem(item, ignoreBounds=True)
        self._vline.setZValue(160)
        self._hline.setZValue(160)
        self._readout.setZValue(200)

        plot_widget.viewport().installEventFilter(self)

    def set_timebase(self, ns_per_sample: float) -> None:
        self._ns_per_sample = max(0.0, ns_per_sample)
        if self._has_selection:
            self._update_readout()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        if not enabled:
            self._dragging = False
            self._hide_all()

    def is_enabled(self) -> bool:
        return self._enabled

    def _hide_all(self) -> None:
        self._has_selection = False
        for item in (self._region, self._sel_v0, self._sel_v1, self._sel_h0,
                     self._sel_h1, self._vline, self._hline, self._readout):
            item.setVisible(False)

    def _scene_to_view(self, pos):
        scene_pt = self._plot_widget.mapToScene(pos.toPoint())
        return self._vb.mapSceneToView(scene_pt)

    def eventFilter(self, obj, event):
        if not self._enabled:
            return False
        etype = event.type()

        if etype == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            pt = self._scene_to_view(event.position())
            self._dragging = True
            self._x0, self._y0 = pt.x(), pt.y()
            self._x1, self._y1 = pt.x(), pt.y()
            self._has_selection = True
            self._show_selection()
            self._update_crosshair(pt.x(), pt.y())
            self._update_readout()
            return True

        if etype == QEvent.Type.MouseMove:
            pt = self._scene_to_view(event.position())
            self._update_crosshair(pt.x(), pt.y())
            self._vline.setVisible(True)
            self._hline.setVisible(True)
            if self._dragging:
                self._x1, self._y1 = pt.x(), pt.y()
                self._show_selection()
                self._update_readout()
                return True
            return False

        if etype == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            if self._dragging:
                self._dragging = False
                pt = self._scene_to_view(event.position())
                self._x1, self._y1 = pt.x(), pt.y()
                self._show_selection()
                self._update_readout()
                return True
            return False

        if etype == QEvent.Type.Leave:
            self._vline.setVisible(False)
            self._hline.setVisible(False)

        return False

    def _update_crosshair(self, x, y):
        self._vline.setValue(x)
        self._hline.setValue(y)

    def _show_selection(self):
        lo, hi = sorted((self._x0, self._x1))
        self._region.setRegion((lo, hi))
        self._sel_v0.setValue(self._x0)
        self._sel_v1.setValue(self._x1)
        self._sel_h0.setValue(self._y0)
        self._sel_h1.setValue(self._y1)
        for item in (self._region, self._sel_v0, self._sel_v1,
                     self._sel_h0, self._sel_h1, self._readout):
            item.setVisible(True)

    def _update_readout(self):
        dt_ns = abs(self._x1 - self._x0) * self._ns_per_sample
        dv = abs(self._y1 - self._y0)
        if dt_ns > 0:
            freq_str = _format_freq(1e9 / dt_ns)
        else:
            freq_str = "—"
        html = (
            "<div style='font-size:11px; font-family:monospace; padding:2px;'>"
            f"<span style='color:{CURSOR_COLOR};'>Time:</span> {_format_time(dt_ns)}<br>"
            f"<span style='color:{CURSOR_COLOR};'>Freq:</span> {freq_str}<br>"
            f"<span style='color:{CURSOR_COLOR};'>Volt:</span> {_format_volt(dv)}"
            "</div>"
        )
        self._readout.setHtml(html)
        self._readout.setPos(min(self._x0, self._x1), max(self._y0, self._y1))
