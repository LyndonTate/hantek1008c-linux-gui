from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
from PyQt6.QtCore import pyqtSignal

NS_PER_DIV_VALUES = [
    1, 2, 5, 10, 20, 50, 100, 200, 500,
    1_000, 2_000, 5_000, 10_000, 20_000, 50_000,
    100_000, 200_000, 500_000,
    1_000_000, 2_000_000, 5_000_000,
    10_000_000, 20_000_000, 50_000_000,
    100_000_000, 200_000_000,
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
    else:
        v = ns / 1_000_000
        return f"{v:g}ms"


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


class ControlsPanel(QWidget):
    time_div_changed = pyqtSignal(int)       # ns_per_div
    channel_toggled = pyqtSignal(int, bool)  # ch_idx, is_on
    vscale_changed = pyqtSignal(int, float)  # ch_idx, vscale
    trigger_channel_changed = pyqtSignal(int)  # ch_idx
    acq_mode_changed = pyqtSignal(str)       # "auto" | "normal" | "single"
    cursor_toggled = pyqtSignal(bool)        # measurement cursor on/off

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)
        self.setStyleSheet("background-color: #1a1a1a; color: #dddddd;")

        self._active = {i: (i == 0) for i in range(8)}
        self._vscales = {i: 1.0 for i in range(8)}
        self._trigger_ch = 0
        # "auto" force-fires after a timeout (free-running scroll when no edge
        # matches); "normal" holds the display until a matching edge arrives.
        self._acq_mode = "auto"
        self._cursor_on = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(6)

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
        self.channel_toggled.emit(ch_idx, new_state)

    def _on_trigger(self, ch_idx):
        if not self._active[ch_idx]:
            return
        if ch_idx == self._trigger_ch:
            return
        self._set_trigger_channel(ch_idx)
        self.trigger_channel_changed.emit(ch_idx)

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

    def _set_trigger_channel(self, ch_idx):
        self._trigger_ch = ch_idx
        self._update_trig_btn_styles()

    def _update_trig_btn_styles(self):
        for i, btn in enumerate(self._trig_btns):
            is_selected = (i == self._trigger_ch)
            btn.setStyleSheet(_trig_btn_style(CHANNEL_COLORS[i], is_selected, self._active[i]))

    def get_trigger_channel(self):
        return self._trigger_ch

    def get_acq_mode(self):
        return self._acq_mode

    def get_ns_per_div(self):
        return self._time_combo.currentData()

    def get_active_channels(self):
        return [i for i in range(8) if self._active[i]]

    def get_vscales(self):
        return dict(self._vscales)
