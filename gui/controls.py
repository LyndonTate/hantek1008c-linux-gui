from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QScrollArea, QFrame,
)
from PyQt6.QtCore import pyqtSignal, Qt, QSize, QRectF
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QPen, QBrush

from gui.measurements import MEASURE_TYPES, MEASURE_GROUPS

NS_PER_DIV_VALUES = [
    1, 2, 5, 10, 20, 50, 100, 200, 500,
    1_000, 2_000, 5_000, 10_000, 20_000, 50_000,
    100_000, 200_000, 500_000,
    1_000_000, 2_000_000, 5_000_000,
    10_000_000, 20_000_000, 50_000_000,
    100_000_000, 200_000_000,
    500_000_000, 1_000_000_000,  # roll mode (500ms, 1s)
    2_000_000_000, 5_000_000_000, 10_000_000_000, 20_000_000_000,  # roll mode (2s..20s)
]

VSCALE_OPTIONS = [
    (0.01, "10mV"), (0.02, "20mV"), (0.05, "50mV"),
    (0.1, "100mV"), (0.2, "200mV"), (0.5, "500mV"),
    (1.0, "1V"), (2.0, "2V"), (5.0, "5V"),
]

CHANNEL_COLORS = [
    "#00ff7f",  # CH1
    "#ffff00",  # CH2
    "#00bfff",  # CH3
    "#ff6600",  # CH4
    "#ff00ff",  # CH5
    "#ff4444",  # CH6
    "#aaaaaa",  # CH7
    "#ffffff",  # CH8
]


def fmt_ns(ns):
    if ns < 1_000:
        return f"{ns}ns"
    elif ns < 1_000_000:
        v = ns / 1_000
        return f"{v:g}µs"
    elif ns < 1_000_000_000:
        v = ns / 1_000_000
        return f"{v:g}ms"
    else:
        v = ns / 1_000_000_000
        return f"{v:g}s"


_COMBO_STYLE = """
    QComboBox {
        background-color: #2a2a2a;
        color: #dddddd;
        border: 1px solid #444444;
        padding: 2px 4px;
        font-size: 12px;
    }
    QComboBox QAbstractItemView {
        background-color: #2a2a2a;
        color: #dddddd;
        selection-background-color: #444444;
    }
"""


def _btn_style(color, is_on):
    if is_on:
        return (f"background-color: {color}; color: #000000; border: none; "
                "padding: 2px 6px; font-size: 11px; font-weight: bold;")
    return ("background-color: #2a2a2a; color: #555555; border: 1px solid #444444; "
            "padding: 2px 6px; font-size: 11px;")


def _trig_btn_style(color, is_selected, ch_is_active):
    if not ch_is_active:
        return ("background-color: #1a1a1a; color: #333333; border: 1px solid #2a2a2a; "
                "padding: 2px 3px; font-size: 11px;")
    if is_selected:
        return (f"background-color: {color}; color: #000000; border: none; "
                "padding: 2px 3px; font-size: 11px; font-weight: bold;")
    return ("background-color: #2a2a2a; color: #555555; border: 1px solid #444444; "
            "padding: 2px 3px; font-size: 11px;")


def _mode_btn_style(is_selected):
    if is_selected:
        return ("background-color: #ff9900; color: #000000; border: none; "
                "padding: 3px 6px; font-size: 11px; font-weight: bold;")
    return ("background-color: #2a2a2a; color: #888888; border: 1px solid #444444; "
            "padding: 3px 6px; font-size: 11px;")


def _auto_type_style(any_on):
    if any_on:
        return ("background-color: #2a2a2a; color: #dddddd; border: 1px solid #888888; "
                "padding: 1px 4px; font-size: 10px; font-weight: bold;")
    return ("background-color: #2a2a2a; color: #888888; border: 1px solid #444444; "
            "padding: 1px 4px; font-size: 10px;")


def _make_mic_icon(muted, hot):
    s = 64
    pm = QPixmap(s, s)
    pm.fill(QColor(0, 0, 0, 0))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    ink = QColor("#111111") if hot else QColor("#dddddd")
    pen = QPen(ink, 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(QBrush(ink))
    p.drawRoundedRect(QRectF(24, 8, 16, 28), 8, 8)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawArc(QRectF(16, 16, 32, 32), 0, -180 * 16)
    p.drawLine(32, 48, 32, 54)
    p.drawLine(20, 54, 44, 54)
    if muted:
        p.setPen(QPen(QColor("#ff5555"), 7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(12, 52, 52, 12)
    p.end()
    return QIcon(pm)


def _auto_dot_style(color, is_on, ch_active):
    if not ch_active:
        return ("background-color: #151515; color: #2a2a2a; border: 1px solid #222222; "
                "border-radius: 2px; padding: 0px; font-size: 10px;")
    if is_on:
        return (f"background-color: {color}; color: #000000; border: none; "
                "border-radius: 2px; padding: 0px; font-size: 10px; font-weight: bold;")
    return ("background-color: #2a2a2a; color: #666666; border: 1px solid #444444; "
            "border-radius: 2px; padding: 0px; font-size: 10px;")


class ControlsPanel(QWidget):
    time_div_changed = pyqtSignal(int)       # ns_per_div
    channel_toggled = pyqtSignal(int, bool)  # ch_idx, is_on
    vscale_changed = pyqtSignal(int, float)  # ch_idx, vscale
    trigger_channel_changed = pyqtSignal(int)  # ch_idx
    trigger_slope_changed = pyqtSignal(str)  # "rising" | "falling"
    acq_mode_changed = pyqtSignal(str)       # "auto" | "normal" | "single"
    cursor_toggled = pyqtSignal(bool)        # measurement cursor on/off
    auto_measure_changed = pyqtSignal()
    record_clicked = pyqtSignal()
    open_clicked = pyqtSignal()
    mic_muted_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)
        self.setStyleSheet("background-color: #1a1a1a; color: #dddddd;")

        self._active = {i: (i == 0) for i in range(8)}
        self._vscales = {i: 1.0 for i in range(8)}
        self._trigger_ch = 0
        self._trigger_slope = "rising"
        # "auto" force-fires after a timeout (free-running scroll when no edge
        # matches); "normal" holds the display until a matching edge arrives.
        self._acq_mode = "auto"
        self._cursor_on = False
        self._auto_on = {(ch, mid): False for ch in range(8) for mid, _, _ in MEASURE_TYPES}
        self._live = True
        self._recording = False
        self._mic_available = True
        self._mic_capture_ready = False
        self._mic_icon_live = _make_mic_icon(False, True)
        self._mic_icon_muted = _make_mic_icon(True, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(6)

        rec_row = QWidget()
        rec_row.setStyleSheet("background-color: transparent;")
        rec_layout = QHBoxLayout(rec_row)
        rec_layout.setContentsMargins(0, 0, 0, 0)
        rec_layout.setSpacing(4)
        self._record_btn = QPushButton("Record")
        self._record_btn.setToolTip("Record live traces to a local .hsrec file")
        self._record_btn.clicked.connect(self.record_clicked.emit)
        self._mic_btn = QPushButton()
        self._mic_btn.setCheckable(True)
        self._mic_btn.setChecked(False)
        self._mic_btn.setFixedWidth(32)
        self._mic_btn.setIconSize(QSize(16, 16))
        self._mic_btn.clicked.connect(self._on_mic_clicked)
        self._open_btn = QPushButton("Open")
        self._open_btn.setToolTip("Open a recording for playback")
        self._open_btn.clicked.connect(self.open_clicked.emit)
        rec_layout.addWidget(self._record_btn)
        rec_layout.addWidget(self._mic_btn)
        rec_layout.addWidget(self._open_btn)
        layout.addWidget(rec_row)
        self._update_record_btn_style()
        self._update_mic_btn_style()

        # Time/Div
        lbl = QLabel("Time / Div")
        lbl.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(lbl)

        self._time_combo = QComboBox()
        self._time_combo.setStyleSheet(_COMBO_STYLE)
        default_idx = 0
        for i, ns in enumerate(NS_PER_DIV_VALUES):
            self._time_combo.addItem(fmt_ns(ns), ns)
            if ns == 500_000:
                default_idx = i
        self._time_combo.setCurrentIndex(default_idx)
        self._time_combo.currentIndexChanged.connect(self._on_time_div)
        layout.addWidget(self._time_combo)

        sep = QLabel()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #333333; margin-top: 4px; margin-bottom: 2px;")
        layout.addWidget(sep)

        lbl_trig = QLabel("Trigger Mode")
        lbl_trig.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(lbl_trig)

        mode_row = QWidget()
        mode_row.setStyleSheet("background-color: transparent;")
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(4)
        self._auto_btn = QPushButton("Auto")
        self._auto_btn.setToolTip("Free-run / scroll when no trigger edge matches")
        self._normal_btn = QPushButton("Normal")
        self._normal_btn.setToolTip("Hold the display until a trigger edge matches")
        self._single_btn = QPushButton("Single")
        self._single_btn.setToolTip("Arm and wait for one trigger edge, then freeze on it")
        for btn, mode in ((self._auto_btn, "auto"), (self._normal_btn, "normal"),
                          (self._single_btn, "single")):
            btn.clicked.connect(lambda _, m=mode: self._on_acq_mode(m))
            mode_layout.addWidget(btn)
        layout.addWidget(mode_row)
        self._update_mode_btn_styles()

        lbl_edge = QLabel("Trigger Edge")
        lbl_edge.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(lbl_edge)

        edge_row = QWidget()
        edge_row.setStyleSheet("background-color: transparent;")
        edge_layout = QHBoxLayout(edge_row)
        edge_layout.setContentsMargins(0, 0, 0, 0)
        edge_layout.setSpacing(4)
        self._rising_btn = QPushButton("Rising ⤴")
        self._rising_btn.setToolTip("Trigger on the rising / leading edge of the waveform")
        self._falling_btn = QPushButton("Falling ⤵")
        self._falling_btn.setToolTip("Trigger on the falling / trailing edge of the waveform")
        for btn, slope in ((self._rising_btn, "rising"), (self._falling_btn, "falling")):
            btn.clicked.connect(lambda _, s=slope: self._on_trigger_slope(s))
            edge_layout.addWidget(btn)
        layout.addWidget(edge_row)
        self._update_slope_btn_styles()

        sep_cur = QLabel()
        sep_cur.setFixedHeight(1)
        sep_cur.setStyleSheet("background-color: #333333; margin-top: 4px; margin-bottom: 2px;")
        layout.addWidget(sep_cur)

        lbl_cur = QLabel("Measurement")
        lbl_cur.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(lbl_cur)

        self._cursor_btn = QPushButton("Cursor")
        self._cursor_btn.setToolTip("Crosshair + click-drag to measure Δt, ΔV and frequency")
        self._cursor_btn.clicked.connect(self._on_cursor_toggle)
        layout.addWidget(self._cursor_btn)
        self._update_cursor_btn_style()

        lbl_auto = QLabel("Auto")
        lbl_auto.setStyleSheet("color: #666666; font-size: 10px; margin-top: 4px;")
        lbl_auto.setToolTip(
            "One row per measure. Dots = channels (click one, or the label for all active). "
            "Live values show on the plot overlay."
        )
        layout.addWidget(lbl_auto)

        self._auto_list_host = QWidget()
        self._auto_list_host.setStyleSheet("background-color: transparent;")
        self._auto_list = QVBoxLayout(self._auto_list_host)
        self._auto_list.setContentsMargins(0, 0, 0, 0)
        self._auto_list.setSpacing(2)
        self._auto_type_btns = {}
        self._auto_dot_btns = {}
        self._build_auto_list()
        auto_scroll = QScrollArea()
        auto_scroll.setWidgetResizable(True)
        auto_scroll.setFrameShape(QFrame.Shape.NoFrame)
        auto_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        auto_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        auto_scroll.setMinimumHeight(180)
        auto_scroll.setMaximumHeight(260)
        auto_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:vertical { width: 6px; background: #1a1a1a; }"
            "QScrollBar::handle:vertical { background: #444444; border-radius: 3px; }"
        )
        auto_scroll.setWidget(self._auto_list_host)
        layout.addWidget(auto_scroll)

        sep2 = QLabel()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet("background-color: #333333; margin-top: 4px; margin-bottom: 2px;")
        layout.addWidget(sep2)

        lbl2 = QLabel("Channels")
        lbl2.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(lbl2)

        self._vscale_combos = []
        self._toggle_btns = []
        self._trig_btns = []

        for i in range(8):
            layout.addWidget(self._make_channel_row(i))

        layout.addStretch()
        self._update_trig_btn_styles()
        self._refresh_auto_list()

    def _make_channel_row(self, ch_idx):
        color = CHANNEL_COLORS[ch_idx]
        is_on = self._active[ch_idx]

        widget = QWidget()
        widget.setStyleSheet("background-color: transparent;")
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        lbl = QLabel(f"CH{ch_idx + 1}")
        lbl.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: bold; min-width: 28px;")
        row.addWidget(lbl)

        combo = QComboBox()
        combo.setStyleSheet(_COMBO_STYLE)
        combo.setFixedWidth(70)
        for vscale, text in VSCALE_OPTIONS:
            combo.addItem(text, vscale)
        combo.setCurrentIndex(6)  # default 1V
        combo.currentIndexChanged.connect(lambda _, idx=ch_idx: self._on_vscale(idx))
        row.addWidget(combo)
        self._vscale_combos.append(combo)

        btn = QPushButton("ON" if is_on else "OFF")
        btn.setFixedWidth(38)
        btn.setStyleSheet(_btn_style(color, is_on))
        btn.clicked.connect(lambda _, idx=ch_idx: self._on_toggle(idx))
        row.addWidget(btn)
        self._toggle_btns.append(btn)

        trig_btn = QPushButton("T")
        trig_btn.setFixedWidth(22)
        trig_btn.setStyleSheet(_trig_btn_style(color, ch_idx == self._trigger_ch, is_on))
        trig_btn.setToolTip(f"Set CH{ch_idx + 1} as trigger source")
        trig_btn.clicked.connect(lambda _, idx=ch_idx: self._on_trigger(idx))
        row.addWidget(trig_btn)
        self._trig_btns.append(trig_btn)

        return widget

    def _build_auto_list(self):
        for _gid, group_label, items in MEASURE_GROUPS:
            hdr = QLabel(group_label)
            hdr.setStyleSheet(
                "color: #666666; font-size: 9px; font-weight: bold; "
                "margin-top: 4px; margin-bottom: 1px;"
            )
            self._auto_list.addWidget(hdr)
            for mid, short, full in items:
                row_w = QWidget()
                row_w.setStyleSheet("background-color: transparent;")
                row = QHBoxLayout(row_w)
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(2)

                type_btn = QPushButton(short)
                type_btn.setFixedWidth(44)
                type_btn.setFixedHeight(18)
                type_btn.setToolTip(f"{full} — toggle for all active channels")
                type_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                type_btn.clicked.connect(lambda _, m=mid: self._on_auto_type(m))
                row.addWidget(type_btn)
                self._auto_type_btns[mid] = type_btn

                for ch in range(8):
                    dot = QPushButton(str(ch + 1))
                    dot.setFixedSize(18, 18)
                    dot.setToolTip(f"CH{ch + 1} {full}")
                    dot.setCursor(Qt.CursorShape.PointingHandCursor)
                    dot.clicked.connect(lambda _, c=ch, m=mid: self._on_auto_dot(c, m))
                    row.addWidget(dot)
                    self._auto_dot_btns[(ch, mid)] = dot

                row.addStretch()
                self._auto_list.addWidget(row_w)
        self._auto_list.addStretch()

    def _on_auto_dot(self, ch, mid):
        if not self._active[ch]:
            return
        key = (ch, mid)
        self._auto_on[key] = not self._auto_on[key]
        self._refresh_auto_list()
        self.auto_measure_changed.emit()

    def _on_auto_type(self, mid):
        active = [ch for ch in range(8) if self._active[ch]]
        if not active:
            return
        all_on = all(self._auto_on[(ch, mid)] for ch in active)
        new_state = not all_on
        for ch in active:
            self._auto_on[(ch, mid)] = new_state
        self._refresh_auto_list()
        self.auto_measure_changed.emit()

    def _refresh_auto_list(self):
        for mid, _, _ in MEASURE_TYPES:
            active = [ch for ch in range(8) if self._active[ch]]
            any_on = any(self._auto_on[(ch, mid)] for ch in active) if active else False
            self._auto_type_btns[mid].setStyleSheet(_auto_type_style(any_on))
            for ch in range(8):
                ch_active = self._active[ch]
                is_on = self._auto_on[(ch, mid)] and ch_active
                dot = self._auto_dot_btns[(ch, mid)]
                dot.setEnabled(ch_active)
                dot.setStyleSheet(_auto_dot_style(CHANNEL_COLORS[ch], is_on, ch_active))

    def get_auto_measure_selection(self):
        return {
            (ch, mid)
            for ch in range(8)
            if self._active[ch]
            for mid, _, _ in MEASURE_TYPES
            if self._auto_on[(ch, mid)]
        }

    def _on_time_div(self, _):
        self.time_div_changed.emit(self._time_combo.currentData())

    def _on_vscale(self, ch_idx):
        vscale = self._vscale_combos[ch_idx].currentData()
        self._vscales[ch_idx] = vscale
        self.vscale_changed.emit(ch_idx, vscale)

    def _on_toggle(self, ch_idx):
        new_state = not self._active[ch_idx]
        if not new_state and sum(self._active.values()) <= 1:
            return  # don't allow all channels off
        self._active[ch_idx] = new_state
        color = CHANNEL_COLORS[ch_idx]
        btn = self._toggle_btns[ch_idx]
        btn.setText("ON" if new_state else "OFF")
        btn.setStyleSheet(_btn_style(color, new_state))
        # If turning off the trigger channel, silently move trigger to first active.
        # channel_toggled already causes a restart that reads get_trigger_channel(), so
        # no separate trigger_channel_changed emission is needed here.
        if not new_state and ch_idx == self._trigger_ch:
            first_active = next(i for i in range(8) if self._active[i])
            self._set_trigger_channel(first_active)
        else:
            self._update_trig_btn_styles()
        if not new_state:
            for mid, _, _ in MEASURE_TYPES:
                self._auto_on[(ch_idx, mid)] = False
        self._refresh_auto_list()
        self.channel_toggled.emit(ch_idx, new_state)
        self.auto_measure_changed.emit()

    def _on_trigger(self, ch_idx):
        if not self._active[ch_idx]:
            return
        if ch_idx == self._trigger_ch:
            return
        self._set_trigger_channel(ch_idx)
        self.trigger_channel_changed.emit(ch_idx)

    def _on_trigger_slope(self, slope):
        if slope == self._trigger_slope:
            return
        self._trigger_slope = slope
        self._update_slope_btn_styles()
        self.trigger_slope_changed.emit(slope)

    def _on_acq_mode(self, mode):
        if mode == self._acq_mode:
            return
        self._acq_mode = mode
        self._update_mode_btn_styles()
        self.acq_mode_changed.emit(mode)

    def _on_cursor_toggle(self):
        self._cursor_on = not self._cursor_on
        self._update_cursor_btn_style()
        self.cursor_toggled.emit(self._cursor_on)

    def _update_cursor_btn_style(self):
        self._cursor_btn.setStyleSheet(_mode_btn_style(self._cursor_on))

    def is_cursor_enabled(self):
        return self._cursor_on

    def clear_mode_selection(self):
        """Deselect all mode buttons — used when a single-shot capture stops."""
        self._acq_mode = "stopped"
        self._update_mode_btn_styles()

    def _update_mode_btn_styles(self):
        self._auto_btn.setStyleSheet(_mode_btn_style(self._acq_mode == "auto"))
        self._normal_btn.setStyleSheet(_mode_btn_style(self._acq_mode == "normal"))
        self._single_btn.setStyleSheet(_mode_btn_style(self._acq_mode == "single"))

    def _update_slope_btn_styles(self):
        self._rising_btn.setStyleSheet(_mode_btn_style(self._trigger_slope == "rising"))
        self._falling_btn.setStyleSheet(_mode_btn_style(self._trigger_slope == "falling"))

    def _set_trigger_channel(self, ch_idx):
        self._trigger_ch = ch_idx
        self._update_trig_btn_styles()

    def _update_trig_btn_styles(self):
        for i, btn in enumerate(self._trig_btns):
            is_selected = (i == self._trigger_ch)
            btn.setStyleSheet(_trig_btn_style(CHANNEL_COLORS[i], is_selected, self._active[i]))

    def get_trigger_channel(self):
        return self._trigger_ch

    def get_trigger_slope(self):
        return self._trigger_slope

    def get_acq_mode(self):
        return self._acq_mode

    def get_ns_per_div(self):
        return self._time_combo.currentData()

    def get_active_channels(self):
        return [i for i in range(8) if self._active[i]]

    def get_vscales(self):
        return dict(self._vscales)

    def get_snapshot(self):
        return {
            "ns_per_div": int(self.get_ns_per_div()),
            "active": [bool(self._active[i]) for i in range(8)],
            "vscales": [float(self._vscales[i]) for i in range(8)],
            "trigger_ch": int(self._trigger_ch),
            "trigger_slope": self._trigger_slope,
            "acq_mode": self._acq_mode,
        }

    def apply_snapshot(self, snap):
        self._time_combo.blockSignals(True)
        idx = self._time_combo.findData(snap["ns_per_div"])
        if idx >= 0:
            self._time_combo.setCurrentIndex(idx)
        self._time_combo.blockSignals(False)

        self._acq_mode = snap["acq_mode"]
        self._update_mode_btn_styles()

        self._trigger_slope = snap["trigger_slope"]
        self._update_slope_btn_styles()

        self._trigger_ch = int(snap["trigger_ch"])

        for ch in range(8):
            self._vscales[ch] = float(snap["vscales"][ch])
            combo = self._vscale_combos[ch]
            combo.blockSignals(True)
            vi = combo.findData(self._vscales[ch])
            if vi >= 0:
                combo.setCurrentIndex(vi)
            combo.blockSignals(False)

            on = bool(snap["active"][ch])
            self._active[ch] = on
            btn = self._toggle_btns[ch]
            btn.setText("ON" if on else "OFF")
            btn.setStyleSheet(_btn_style(CHANNEL_COLORS[ch], on))

        self._update_trig_btn_styles()
        self._refresh_auto_list()

    def set_live_enabled(self, on):
        self._live = on
        self._time_combo.setEnabled(on)
        for btn in (self._auto_btn, self._normal_btn, self._single_btn,
                    self._rising_btn, self._falling_btn):
            btn.setEnabled(on)
        for w in self._vscale_combos + self._toggle_btns + self._trig_btns:
            w.setEnabled(on)
        self._record_btn.setEnabled(on)
        self._sync_mic_enabled()
        self._open_btn.setEnabled(True if not on else not self._recording)
        if on:
            self._update_trig_btn_styles()

    def set_recording(self, on, path=""):
        self._recording = on
        self._open_btn.setEnabled(self._live and not on)
        if not on:
            self._mic_capture_ready = False
            self._mic_btn.blockSignals(True)
            self._mic_btn.setChecked(False)
            self._mic_btn.blockSignals(False)
        if on:
            self._record_btn.setToolTip(path)
        else:
            self._record_btn.setToolTip("Record live traces to a local .hsrec file")
        self._sync_mic_enabled()
        self._update_record_btn_style()
        self._update_mic_btn_style()

    def set_mic_capture_ready(self, on):
        self._mic_capture_ready = bool(on) and self._recording
        self._mic_btn.blockSignals(True)
        self._mic_btn.setChecked(False)
        self._mic_btn.blockSignals(False)
        self._sync_mic_enabled()
        self._update_mic_btn_style()

    def set_mic_available(self, on):
        self._mic_available = bool(on)
        self._mic_btn.setVisible(self._mic_available)
        self._sync_mic_enabled()

    def _sync_mic_enabled(self):
        self._mic_btn.setEnabled(
            self._live and self._recording and self._mic_available and self._mic_capture_ready
        )

    def _on_mic_clicked(self):
        self._update_mic_btn_style()
        self.mic_muted_changed.emit(not self._mic_btn.isChecked())

    def _update_mic_btn_style(self):
        live = self._mic_btn.isChecked()
        self._mic_btn.setIcon(self._mic_icon_live if live else self._mic_icon_muted)
        self._mic_btn.setStyleSheet(_mode_btn_style(live))
        if live:
            self._mic_btn.setToolTip("Mute microphone")
        elif self._recording and not self._mic_capture_ready:
            self._mic_btn.setToolTip("Microphone unavailable")
        elif not self._recording:
            self._mic_btn.setToolTip("Unmute after starting a recording")
        else:
            self._mic_btn.setToolTip("Unmute microphone")

    def _update_record_btn_style(self):
        if self._recording:
            self._record_btn.setText("Stop")
            self._record_btn.setStyleSheet(
                "background-color: #ff5555; color: #000000; border: none; "
                "padding: 3px 6px; font-size: 11px; font-weight: bold;"
            )
        else:
            self._record_btn.setText("Record")
            self._record_btn.setStyleSheet(_mode_btn_style(False))
