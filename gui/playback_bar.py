from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSlider, QLabel
from PyQt6.QtCore import Qt, pyqtSignal


def _fmt_ns(ns):
    s = max(0, int(ns) // 1_000_000_000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


_BTN = (
    "background-color: #2a2a2a; color: #dddddd; border: 1px solid #444444;"
    "padding: 3px 8px; font-size: 11px;"
)
_BTN_ON = (
    "background-color: #ff9900; color: #000000; border: none;"
    "padding: 3px 8px; font-size: 11px; font-weight: bold;"
)


class PlaybackBar(QWidget):
    play_clicked = pyqtSignal()
    live_clicked = pyqtSignal()
    seek_ns = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._duration_ns = 0
        self._playing = False
        self.setFixedHeight(36)
        self.setStyleSheet("background-color: #111111;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        self._play_btn = QPushButton("Pause")
        self._play_btn.setFixedWidth(64)
        self._play_btn.setStyleSheet(_BTN_ON)
        self._play_btn.clicked.connect(self.play_clicked.emit)
        layout.addWidget(self._play_btn)

        self._live_btn = QPushButton("Live")
        self._live_btn.setFixedWidth(52)
        self._live_btn.setStyleSheet(_BTN)
        self._live_btn.clicked.connect(self.live_clicked.emit)
        layout.addWidget(self._live_btn)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1)
        self._slider.setStyleSheet(
            "QSlider::groove:horizontal { height: 6px; background: #333333; border-radius: 3px; }"
            "QSlider::handle:horizontal { width: 12px; height: 12px; margin: -4px 0;"
            " background: #ff9900; border-radius: 6px; }"
            "QSlider::sub-page:horizontal { background: #886600; border-radius: 3px; }"
        )
        self._slider.sliderMoved.connect(self._on_moved)
        self._slider.sliderReleased.connect(self._on_released)
        layout.addWidget(self._slider, stretch=1)

        self._time = QLabel("00:00 / 00:00")
        self._time.setStyleSheet("color: #cccccc; font-size: 11px; font-family: monospace;")
        self._time.setFixedWidth(110)
        layout.addWidget(self._time)

    def set_duration_ns(self, ns):
        self._duration_ns = max(0, int(ns))
        ms = max(1, self._duration_ns // 1_000_000)
        self._slider.blockSignals(True)
        self._slider.setRange(0, ms)
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._set_time_label(0)

    def set_playing(self, on):
        self._playing = on
        self._play_btn.setText("Pause" if on else "Play")
        self._play_btn.setStyleSheet(_BTN_ON if on else _BTN)

    def set_position_ns(self, ns):
        if self._slider.isSliderDown():
            return
        ns = max(0, min(int(ns), self._duration_ns))
        ms = ns // 1_000_000
        self._slider.blockSignals(True)
        self._slider.setValue(ms)
        self._slider.blockSignals(False)
        self._set_time_label(ns)

    def _on_moved(self, ms):
        ns = ms * 1_000_000
        self._set_time_label(ns)
        self.seek_ns.emit(ns)

    def _on_released(self):
        self.seek_ns.emit(self._slider.value() * 1_000_000)

    def _set_time_label(self, ns):
        self._time.setText(f"{_fmt_ns(ns)} / {_fmt_ns(self._duration_ns)}")
