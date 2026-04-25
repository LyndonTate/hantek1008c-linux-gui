from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPolygon, QFont


class ChannelMarginWidget(QWidget):
    """Narrow strip to the left of the plot containing draggable per-channel
    offset handles.  Each handle is a small colored right-pointing triangle
    labelled with its channel number; dragging it vertically shifts that
    channel's trace without touching the hardware trigger."""

    channel_dragged = pyqtSignal(int, float)  # ch_id, new_offset_volts

    _HIT_RADIUS = 12   # pixels from handle centre that counts as a grab

    def __init__(self, plot_widget, parent=None):
        super().__init__(parent)
        self._plot = plot_widget
        self._channels = {}   # {ch_id: [offset_volts, color_str]}
        self._yrange = 4.0
        self._dragging = None
        self._drag_y_start = 0
        self._drag_offset_start = 0.0
        self.setFixedWidth(20)
        self.setStyleSheet("background-color: #000000;")
        self.setMouseTracking(True)

    # ------------------------------------------------------------------ public

    def set_channels(self, channels: dict, yrange: float):
        """channels: {ch_id: (offset_volts, color_str)}"""
        self._channels = {k: list(v) for k, v in channels.items()}
        self._yrange = yrange
        self.update()

    def update_offset(self, ch_id: int, offset: float):
        if ch_id in self._channels:
            self._channels[ch_id][0] = offset
            self.update()

    def set_yrange(self, yrange: float):
        self._yrange = yrange
        self.update()

    # --------------------------------------------------------- coordinate math

    def _vb_top_and_height(self):
        """Return (top_px, height_px) of the ViewBox in this widget's coords."""
        vb = self._plot.getPlotItem().getViewBox()
        sr = vb.sceneBoundingRect()
        pw = self._plot
        tl = self.mapFromGlobal(pw.mapToGlobal(pw.mapFromScene(sr.topLeft())))
        br = self.mapFromGlobal(pw.mapToGlobal(pw.mapFromScene(sr.bottomRight())))
        return tl.y(), max(1, br.y() - tl.y())

    def _offset_to_y(self, offset):
        top, h = self._vb_top_and_height()
        frac = 1.0 - (offset + self._yrange) / (2.0 * self._yrange)
        return int(top + frac * h)

    def _y_to_offset(self, y):
        top, h = self._vb_top_and_height()
        frac = (y - top) / h
        return self._yrange * (1.0 - 2.0 * frac)

    # ----------------------------------------------------------------- painting

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        font = QFont()
        font.setPixelSize(8)
        font.setBold(True)
        p.setFont(font)

        for ch_id, (offset, color_str) in self._channels.items():
            cy = self._offset_to_y(offset)
            color = QColor(color_str)
            # right-pointing triangle: tip at right edge, base on left
            tri = QPolygon([
                QPoint(w - 1, cy),
                QPoint(0,     cy - 6),
                QPoint(0,     cy + 6),
            ])
            p.setBrush(QBrush(color))
            p.setPen(QPen(color.darker(160), 1))
            p.drawPolygon(tri)
            # channel number centred in the triangle body
            p.setPen(QPen(QColor("#000000")))
            p.drawText(QRect(0, cy - 6, w - 4, 12),
                       Qt.AlignmentFlag.AlignCenter, str(ch_id + 1))

    # --------------------------------------------------------- mouse handling

    def _channel_at_y(self, y):
        best, best_dist = None, self._HIT_RADIUS
        # Iterate in reverse paint order so the topmost (last-painted) handle
        # wins when two handles overlap at the same distance.
        for ch_id, (offset, _) in reversed(list(self._channels.items())):
            d = abs(self._offset_to_y(offset) - y)
            if d < best_dist:
                best_dist, best = d, ch_id
        return best

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            ch = self._channel_at_y(event.pos().y())
            if ch is not None:
                self._dragging = ch
                self._drag_y_start = event.pos().y()
                self._drag_offset_start = self._channels[ch][0]
                self.setCursor(Qt.CursorShape.SizeVerCursor)
                event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging is not None:
            dy = event.pos().y() - self._drag_y_start
            _, h = self._vb_top_and_height()
            delta = -dy * (2.0 * self._yrange) / h
            new_offset = self._drag_offset_start + delta
            self._channels[self._dragging][0] = new_offset
            self.update()
            self.channel_dragged.emit(self._dragging, new_offset)
            event.accept()
        else:
            # change cursor on hover near a handle
            ch = self._channel_at_y(event.pos().y())
            self.setCursor(Qt.CursorShape.SizeVerCursor if ch is not None
                           else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = None
            event.accept()
