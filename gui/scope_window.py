import sys
import logging
import numpy as np
import pyqtgraph as pg

log = logging.getLogger(__name__)
from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QApplication
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt, QTimer

from gui.controls import ControlsPanel, CHANNEL_COLORS, NS_PER_DIV_VALUES
from gui.acquisition import AcquisitionThread
from gui.channel_margin import ChannelMarginWidget
from gui.cursor_overlay import CursorOverlay
from vendor.hantek1008 import Hantek1008

TIME_DIVS = 10
VOLT_DIVS = 8
GRID_COLOR = "#2a2a2a"
ADC_MIN_NS_PER_SAMPLE = 416  # fast-fixed max sample rate (~2.4 MSa/s)
# In slow-burst mode (ns/div ≥ 200µs) the device clocks its 4000-short buffer at
# a fixed rate of ~1250 ns/sample regardless of the ac-payload's B_SUM field.
# Empirically: at 500µs/div the 4000-sample buffer spans exactly 5000µs (10
# divs), confirming the rate. At 200µs/div the same buffer still spans 5000µs
# (= 25 divs at 200µs), so the GUI must display only 1600 of the 4000 samples
# to render the labeled 10-div window. B_SUM (which only changes for 200µs in
# Windows captures, to 1402) doesn't actually move the rate — the device clamps
# to its slow-burst clock floor.
SLOW_BURST_NS_PER_SAMPLE = 1250
# At ns_per_div <= 100us the device runs in "fast-fixed" mode where the ADC
# samples at its max rate (~416 ns/sample) regardless of the requested ns/div.
# At ≤100us the requested rate exceeds 416ns so the buffer fits inside one
# screen and the 416 clamp matches both the device and the display formula.
FAST_FIXED_NS_PER_DIV_MAX = 100_000

# When the display window contains few samples (fast-fixed mode at low ns/div),
# the polyline between widely-spaced points becomes visibly jagged AND the raw
# data carries real ±30-40 ADC-unit zigzag at the per-sample timescale (visible
# as wobble on rising edges). The Windows vendor app clearly low-pass filters
# its display, so we do the same: a tiny Gaussian smooth knocks down the
# alternating-sample noise, then a Catmull-Rom cubic spline upsample removes
# the polyline-corner artifacts. Total drawn points stay well under 1000 with
# no measurable perf hit on pyqtgraph.
_INTERP_THRESHOLD = 600      # apply smoothing+interpolation only when display_samples < this
                             # (covers fast-fixed time-bases up to ~20us/div = 480 samples)
_INTERP_FACTOR = 10          # output 10 points per input segment
# 5-tap Gaussian (sigma≈1.0); sums to 1.0
_SMOOTH_KERNEL = np.array([0.06136, 0.24477, 0.38774, 0.24477, 0.06136])


# AutoSet quality gates
_AUTOSET_MIN_AMPLITUDE = 4  # ADC counts peak-to-peak (~5 % of 12-bit range)
_AUTOSET_MIN_CYCLES = 5         # need this many complete cycles in the buffer

def _reject_spikes(y: np.ndarray) -> np.ndarray:
    """3-point median filter: eliminates single-sample ADC spikes for display.

    Has zero effect on well-sampled signals (median of 3 consecutive points of
    a smooth waveform equals the middle point), but replaces isolated outliers
    with the average of their neighbours.
    """
    if len(y) < 3:
        return y
    out = y.copy()
    out[1:-1] = np.median(np.stack([y[:-2], y[1:-1], y[2:]]), axis=0)
    return out


def _estimate_period_ns(samples: np.ndarray, ns_per_sample: float):
    """Estimate signal period in ns via rising zero-crossing intervals.

    Returns None when the signal doesn't look periodic (too small, too few
    cycles, or crossing intervals too irregular / noisy).
    """
    peak_to_peak = float(samples.max()) - float(samples.min())
    if peak_to_peak < _AUTOSET_MIN_AMPLITUDE:
        return None

    mid = (float(samples.max()) + float(samples.min())) / 2
    above = samples > mid
    crossings = np.where(np.diff(above.astype(np.int8)) == 1)[0]
    if len(crossings) < _AUTOSET_MIN_CYCLES + 1:
        return None

    intervals = np.diff(crossings).astype(float)
    mean_iv = float(np.mean(intervals))
    if mean_iv == 0:
        return None

    return float(np.median(intervals)) * ns_per_sample


def _smooth_and_upsample(y: np.ndarray, factor: int = _INTERP_FACTOR) -> np.ndarray:
    """Low-pass smooth then Catmull-Rom upsample, numpy-only.

    Returns ``(len(y) - 1) * factor + 1`` points spanning the same X range as
    the input. Endpoints are reflected to keep the smoothing kernel from
    pulling the curve toward zero at the boundaries.
    """
    n = len(y)
    if n < 2 or factor <= 1:
        return y.astype(float, copy=False)
    yf = y.astype(float, copy=False)
    if n >= len(_SMOOTH_KERNEL):
        # Reflect-pad to preserve endpoint values during convolution
        pad = len(_SMOOTH_KERNEL) // 2
        padded = np.concatenate((yf[pad:0:-1], yf, yf[-2:-pad - 2:-1]))
        yf = np.convolve(padded, _SMOOTH_KERNEL, mode="valid")
    p = np.concatenate(([yf[0]], yf, [yf[-1]]))  # length n+2 for tangents
    t = np.linspace(0.0, 1.0, factor, endpoint=False)
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    out = np.empty((n - 1) * factor + 1, dtype=float)
    for i in range(n - 1):
        m1 = (p[i + 2] - p[i]) * 0.5
        m2 = (p[i + 3] - p[i + 1]) * 0.5
        out[i * factor:(i + 1) * factor] = (
            h00 * p[i + 1] + h10 * m1 + h01 * p[i + 2] + h11 * m2
        )
    out[-1] = yf[-1]
    return out


# Maps user-facing display V/div to the 3 hardware gain settings
_HW_VSCALE_BREAKPOINTS = [(0.05, 0.02), (0.5, 0.125)]

def _hw_vscale_for(display_vscale: float) -> float:
    """Return the hardware gain (0.02 / 0.125 / 1.0) for a display V/div value."""
    for threshold, hw in _HW_VSCALE_BREAKPOINTS:
        if display_vscale <= threshold:
            return hw
    return 1.0


def _pad_channels_to_pairs(channels: list) -> list:
    """Ensure active channels always come in hardware pairs (0,1), (2,3), (4,5), (6,7).

    The Hantek 1008C firmware crashes if an odd-numbered channel count other than 1
    is sent.  A single active channel is allowed (Windows sends count=1 directly),
    so only pad when there are 3, 5, or 7 active channels.
    """
    if len(channels) <= 1:
        return sorted(channels)
    padded = set(channels)
    for ch in list(padded):
        padded.add(ch ^ 1)  # partner: 0↔1, 2↔3, 4↔5, 6↔7
    return sorted(padded)


def _yrange_for(vscales):
    return max(vscales) * (VOLT_DIVS / 2) if vscales else 4.0


class TimeAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ns_per_div = 0
        self._samples_per_div = 1

    def set_timebase(self, ns_per_div: int, samples_per_div: float):
        self._ns_per_div = ns_per_div
        self._samples_per_div = max(samples_per_div, 1.0)
        # Force the axis to recompute ticks/labels. Switching between two roll
        # time-bases (e.g. 1s -> 500ms) keeps the same sample X-range, so
        # pyqtgraph won't otherwise invalidate its cached tick picture.
        self.picture = None
        self.update()

    def tickValues(self, minVal, maxVal, size):
        # Place major ticks exactly on the division grid lines (multiples of
        # samples_per_div) so the labels line up with the vertical grid.
        if self._samples_per_div <= 0:
            return super().tickValues(minVal, maxVal, size)
        step = self._samples_per_div
        first = int(np.ceil(minVal / step))
        last = int(np.floor(maxVal / step))
        ticks = [k * step for k in range(first, last + 1)]
        return [(step, ticks)]

    def tickStrings(self, values, scale, spacing):
        if self._samples_per_div <= 0:
            return super().tickStrings(values, scale, spacing)
        ns_per_sample = self._ns_per_div / self._samples_per_div
        labels = []
        for value in values:
            time_ns = value * ns_per_sample
            labels.append(self._format_time(time_ns))
        return labels

    @staticmethod
    def _format_time(ns: float) -> str:
        if ns >= 1_000_000_000:
            return f"{ns / 1_000_000_000:.3g}s"
        if ns >= 1_000_000:
            return f"{ns / 1_000_000:.3g}ms"
        if ns >= 1_000:
            return f"{ns / 1_000:.3g}µs"
        return f"{ns:.0f}ns"


class ScopeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hanscope")
        self.setMinimumSize(1280, 720)
        self.resize(1280, 720)

        self._display_samples = 0
        self._samples_per_div = 1.0
        self._initialized = False
        self._restarting = False         # True while switching timebases; discards stale frames
        self._restart_in_progress = False  # re-entrancy guard for _restart_acquisition
        self._pending_restart = False      # a restart was requested while one was in progress
        self._autoset_searching = False  # True while iterating through timebases looking for signal
        self._autoset_changing_timebase = False  # suppress search cancel during autoset-driven restarts
        self._autoset_current_idx = 0   # current index into NS_PER_DIV_VALUES during search
        self._channel_data = {}   # {ch_idx: {'curve': PlotDataItem}}
        self._channel_offsets = {}       # {ch_idx: float} display-only Y offset in volts
        self._h_grid_lines = []
        self._v_grid_lines = []
        self._acq = None
        self._device = None              # kept alive across restarts to avoid firmware reset
        self._trigger_level_volts = 0.0  # current trigger level in volts
        self._zero_offsets = {}          # {vscale: [per-channel float]} from device calibration
        self._frame_size = 0
        self._last_frame_np = {}         # {ch_id: np.ndarray} cached for drag redraws
        self._last_frame_triggered = False  # was the displayed frame a real trigger?

        # Roll mode (500ms, 1s/div): samples stream in and are drawn into a
        # fixed-width ring buffer whose write head sweeps left->right, then wraps
        # and overwrites in place — mirroring the vendor app's rolling trace.
        self._roll_mode = False
        self._roll_buf = {}              # {ch_id: np.ndarray} ring buffer of volts
        self._roll_head = 0              # next write index into the ring buffers

        self._setup_ui()
        self._start_acquisition()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._time_axis = TimeAxisItem(orientation='bottom')
        self._plot_widget = pg.PlotWidget(axisItems={'bottom': self._time_axis})

        self._ch_margin = ChannelMarginWidget(self._plot_widget)
        self._ch_margin.channel_dragged.connect(self._on_channel_dragged)
        layout.addWidget(self._ch_margin)
        layout.addWidget(self._plot_widget, stretch=1)

        # Right panel: status bar on top, controls below
        right = QWidget()
        right.setFixedWidth(220)
        right.setStyleSheet("background-color: #1a1a1a;")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self._status_label = QLabel("● Connecting")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setFixedHeight(24)
        self._status_label.setStyleSheet(
            "color: #ffaa00; font-size: 11px; background-color: #111111;"
            "border-bottom: 1px solid #333333;"
        )
        right_layout.addWidget(self._status_label)

        self._controls = ControlsPanel()
        right_layout.addWidget(self._controls, stretch=1)

        layout.addWidget(right)

        self._setup_plot()
        self._setup_trigger_marker()
        self._setup_h_trigger_marker()

        self._cursor_overlay = CursorOverlay(self._plot_widget)

        self._controls.time_div_changed.connect(self._on_time_div_changed)
        self._controls.channel_toggled.connect(self._on_channel_toggled)
        self._controls.vscale_changed.connect(self._on_vscale_changed)
        self._controls.trigger_channel_changed.connect(self._on_trigger_channel_changed)
        self._controls.trigger_slope_changed.connect(self._on_trigger_slope_changed)
        self._controls.acq_mode_changed.connect(self._on_acq_mode_changed)
        self._controls.cursor_toggled.connect(self._on_cursor_toggled)
        self._controls.autoset_requested.connect(self._on_autoset)

        self._on_acq_mode_changed(self._controls.get_acq_mode())

    def _setup_plot(self):
        self._plot_widget.setBackground("#000000")
        pi = self._plot_widget.getPlotItem()
        pi.showAxis("left")
        pi.showAxis("bottom")
        pi.setLabel("left", "Voltage", units="V", labelStyle={"color": "#dddddd", "font-size": "11px"})
        pi.setLabel("bottom", "Time", labelStyle={"color": "#dddddd", "font-size": "11px"})
        left_axis = pi.getAxis("left")
        bottom_axis = pi.getAxis("bottom")
        left_axis.setWidth(45)
        bottom_axis.setHeight(45)
        left_axis.setPen(pg.mkPen("#888888"))
        left_axis.setTextPen("#ffffff")
        left_axis.setStyle(tickTextOffset=8, tickFont=QFont("Arial", 10))
        bottom_axis.setPen(pg.mkPen("#888888"))
        bottom_axis.setTextPen("#ffffff")
        bottom_axis.setStyle(tickTextOffset=8, tickFont=QFont("Arial", 10))
        self._time_axis.enableAutoSIPrefix(False)
        pi.setMenuEnabled(False)
        self._plot_widget.setMouseEnabled(x=False, y=False)
        self._update_yrange()

    def _update_yrange(self):
        vscales_dict = self._controls.get_vscales()
        active = self._controls.get_active_channels()
        active_vscales = [vscales_dict[ch] for ch in active]
        yrange = _yrange_for(active_vscales)
        self._plot_widget.setYRange(-yrange, yrange, padding=0.05)
        self._rebuild_h_grid(yrange)
        self._ch_margin.set_yrange(yrange)

    def _rebuild_h_grid(self, yrange):
        pi = self._plot_widget.getPlotItem()
        grid_pen = pg.mkPen(GRID_COLOR, width=1)
        for line in self._h_grid_lines:
            pi.removeItem(line)
        self._h_grid_lines.clear()
        step = 2 * yrange / VOLT_DIVS
        for i in range(1, VOLT_DIVS):
            v = -yrange + i * step
            line = pg.InfiniteLine(pos=v, angle=0, pen=grid_pen, movable=False)
            pi.addItem(line)
            self._h_grid_lines.append(line)

    def _setup_trigger_marker(self):
        self._trigger_marker = pg.InfiniteLine(
            pos=self._trigger_level_volts,
            angle=0,
            movable=True,
            pen=pg.mkPen("#ff9900", width=1, style=pg.QtCore.Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen("#ff9900", width=12),
            label="T1",
            labelOpts={"color": "#ff9900", "movable": False, "position": 0.97},
        )
        self._trigger_marker.setZValue(100)
        self._trigger_marker.sigPositionChangeFinished.connect(self._on_trigger_marker_moved)
        self._plot_widget.addItem(self._trigger_marker)

    def _update_trigger_marker_label(self):
        trig_ch = self._controls.get_trigger_channel()
        self._trigger_marker.label.setFormat(f"T{trig_ch + 1}")

    def _setup_h_trigger_marker(self):
        self._h_trigger_marker = pg.InfiniteLine(
            pos=0,
            angle=90,
            movable=True,
            pen=pg.mkPen("#ff9900", width=1, style=pg.QtCore.Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen("#ff9900", width=12),
            label="T",
            labelOpts={"color": "#ff9900", "movable": False, "position": 0.05},
        )
        self._h_trigger_marker.setZValue(100)
        self._h_trigger_marker.sigPositionChanged.connect(self._on_h_trigger_moved)
        self._plot_widget.addItem(self._h_trigger_marker)

    def _on_h_trigger_moved(self):
        pos = int(self._h_trigger_marker.value())
        self._acq.queue_hw_trigger_pre_samples(pos)
        # In fast-fixed mode the marker is realised in software (see _redraw),
        # so repaint immediately for snappy feedback.
        if self._last_frame_np:
            self._redraw()

    def _redraw(self):
        ns = self._controls.get_ns_per_div()
        if not self._last_frame_triggered:
            # Free-run / single-preview: no trigger event to align to, just show
            # the start of the buffer so the trace slides freely.
            start, end = 0, self._display_samples
        elif self._frame_size > self._display_samples:
            # The captured buffer is larger than the labeled time window — true
            # in fast-fixed mode at low ns/div AND in slow-burst at 200µs (where
            # the 4000-sample buffer's 5000µs span exceeds the 2000µs label).
            # The hardware always centers the trigger event in its 4000-sample
            # buffer when we send a centered ac payload (A=4000, B1=B2), so
            # slide the display window so the trigger event lines up with the
            # user's H-trigger marker on screen.
            hw_trigger_idx = self._frame_size // 2
            h_marker = int(self._h_trigger_marker.value())
            start = hw_trigger_idx - h_marker
            start = max(0, min(start, self._frame_size - self._display_samples))
            end = start + self._display_samples
        else:
            start, end = 0, self._display_samples
        for ch_id, samples in self._last_frame_np.items():
            if ch_id not in self._channel_data:
                continue
            window = _reject_spikes(samples[start:end])
            if len(window) < _INTERP_THRESHOLD and len(window) >= 2:
                y = _smooth_and_upsample(window)
                x = np.linspace(0, len(window) - 1, len(y))
                self._channel_data[ch_id]["curve"].setData(x, y)
            else:
                self._channel_data[ch_id]["curve"].setData(window)

    def _volts_to_adc(self, volts, display_vscale, channel_id):
        hw_vscale = _hw_vscale_for(display_vscale)
        zero_offset = 2048
        if hw_vscale in self._zero_offsets:
            offsets = self._zero_offsets[hw_vscale]
            if channel_id < len(offsets):
                zero_offset = offsets[channel_id]
        adc = int(volts / (0.01 * hw_vscale) + zero_offset)
        return max(0, min(4095, adc))

    def _on_trigger_marker_moved(self):
        vscales_dict = self._controls.get_vscales()
        trig_ch = self._controls.get_trigger_channel()
        trig_offset = self._channel_offsets.get(trig_ch, 0.0)
        # Trigger fires at the real wire voltage, independent of the display offset
        self._trigger_level_volts = self._trigger_marker.value() - trig_offset
        trig_adc = self._volts_to_adc(self._trigger_level_volts, vscales_dict[trig_ch], trig_ch)
        self._acq.queue_trigger_level(trig_adc)

    def _on_channel_dragged(self, ch_id, new_offset):
        self._channel_offsets[ch_id] = new_offset
        if ch_id in self._channel_data:
            self._channel_data[ch_id]["curve"].setPos(0, new_offset)
        trig_ch = self._controls.get_trigger_channel()
        if ch_id == trig_ch:
            # Keep trigger marker visually aligned with the moving trace
            self._trigger_marker.blockSignals(True)
            self._trigger_marker.setValue(self._trigger_level_volts + new_offset)
            self._trigger_marker.blockSignals(False)

    def _compute_display_geometry(self, frame_size, ns_per_div):
        """Return (display_samples, samples_per_div) for the current timebase.

        samples_per_div is the float number of buffer samples that span exactly
        one labeled time-division at the device's actual sample period. The X
        axis grid uses this for its step so each grid square always represents
        ns_per_div of real time.

        display_samples is how many samples we render. When the buffer can hold
        the full 10-div window we render 10×samples_per_div. When it can't
        (fast-fixed mode at ≥200us, where the 4000-sample buffer covers
        ~1664us at the fixed 416 ns/sample rate), we render only the integer
        number of full divs that fit, so the labeled width stays accurate and
        no partial sample-stretching is needed.
        """
        if Hantek1008.is_roll_mode_ns_per_div(ns_per_div):
            # Roll mode: the sample rate is fixed by the time-base, independent
            # of any hardware buffer size. With fewer than 8 active channels the
            # device actually streams faster than the nominal table rate (the
            # single-channel factor is ~4.56x), so we scale by the hardware
            # channel count to keep the time axis and sweep speed correct.
            hw_active = _pad_channels_to_pairs(self._controls.get_active_channels())
            n_hw = max(1, len(hw_active))
            sampling_rate = Hantek1008.effective_roll_sampling_rate(ns_per_div, n_hw)
            samples_per_div = sampling_rate * ns_per_div / 1e9
            display_samples = max(1, int(round(TIME_DIVS * samples_per_div)))
            return display_samples, samples_per_div

        if ns_per_div <= FAST_FIXED_NS_PER_DIV_MAX:
            actual_ns_per_sample = ADC_MIN_NS_PER_SAMPLE
        else:
            requested_ns_per_sample = (ns_per_div * TIME_DIVS) / frame_size
            # Slow-burst can't sample faster than SLOW_BURST_NS_PER_SAMPLE — at
            # 200µs the requested 500ns/sample exceeds the device's 1250ns/sample
            # floor, so we display fewer samples than frame_size to keep the
            # labeled width accurate.
            actual_ns_per_sample = max(requested_ns_per_sample, SLOW_BURST_NS_PER_SAMPLE)
        samples_per_div = ns_per_div / actual_ns_per_sample
        full_divs = min(TIME_DIVS, int(frame_size // samples_per_div))
        display_samples = max(1, int(full_divs * samples_per_div))
        return display_samples, samples_per_div

    def _compute_display_samples(self, frame_size, ns_per_div):
        return self._compute_display_geometry(frame_size, ns_per_div)[0]

    def _init_buffer(self, frame_size):
        ns_per_div = self._controls.get_ns_per_div()
        old_display_samples = self._display_samples
        self._display_samples, samples_per_div = self._compute_display_geometry(frame_size, ns_per_div)
        self._samples_per_div = samples_per_div
        self._time_axis.set_timebase(ns_per_div, samples_per_div)
        self._cursor_overlay.set_timebase(ns_per_div / samples_per_div if samples_per_div else 0.0)
        log.info("_init_buffer: ns/div=%d frame_size=%d -> display_samples=%d samples_per_div=%.2f (was display=%d)",
                 ns_per_div, frame_size, self._display_samples, samples_per_div, old_display_samples)
        is_first_init = self._frame_size == 0
        self._frame_size = frame_size
        self._last_frame_np = {}

        if is_first_init:
            new_h_pos = self._display_samples // 2
        else:
            # Preserve fractional screen position across timebase changes
            old_marker = int(self._h_trigger_marker.value())
            fraction = old_marker / max(1, old_display_samples - 1)
            new_h_pos = max(0, min(int(round(fraction * (self._display_samples - 1))), self._display_samples - 1))

        self._h_trigger_marker.blockSignals(True)
        self._h_trigger_marker.setBounds((0, self._display_samples - 1))
        self._h_trigger_marker.setValue(new_h_pos)
        self._h_trigger_marker.blockSignals(False)
        self._acq.queue_hw_trigger_pre_samples(new_h_pos)

        self._plot_widget.setXRange(0, self._display_samples - 1, padding=0)

        pi = self._plot_widget.getPlotItem()
        grid_pen = pg.mkPen(GRID_COLOR, width=1)

        for line in self._v_grid_lines:
            pi.removeItem(line)
        self._v_grid_lines.clear()
        x_step = samples_per_div
        for i in range(1, TIME_DIVS):
            pos = i * x_step
            if pos >= self._display_samples:
                break
            line = pg.InfiniteLine(pos=pos, angle=90, pen=grid_pen, movable=False)
            pi.addItem(line)
            self._v_grid_lines.append(line)

        for info in self._channel_data.values():
            self._plot_widget.removeItem(info["curve"])
        self._channel_data.clear()

        active = self._controls.get_active_channels()
        for ch in active:
            color = CHANNEL_COLORS[ch]
            offset = self._channel_offsets.get(ch, 0.0)
            curve = self._plot_widget.plot(pen=pg.mkPen(color, width=1))
            curve.setPos(0, offset)
            self._channel_data[ch] = {"curve": curve}

        # Update margin handles: one per active channel
        vscales_dict = self._controls.get_vscales()
        active_vscales = [vscales_dict[ch] for ch in active]
        yrange = _yrange_for(active_vscales)
        margin_channels = {ch: (self._channel_offsets.get(ch, 0.0), CHANNEL_COLORS[ch])
                           for ch in active}
        self._ch_margin.set_channels(margin_channels, yrange)

        # Align trigger marker with current trigger channel offset
        trig_ch = self._controls.get_trigger_channel()
        trig_offset = self._channel_offsets.get(trig_ch, 0.0)
        self._trigger_marker.blockSignals(True)
        self._trigger_marker.setValue(self._trigger_level_volts + trig_offset)
        self._trigger_marker.blockSignals(False)

    def _start_acquisition(self):
        self._restarting = False          # now safe to accept frames from new thread
        active = self._controls.get_active_channels()
        hw_active = _pad_channels_to_pairs(active)  # hardware requires channels in pairs
        vscales_dict = self._controls.get_vscales()
        ns = self._controls.get_ns_per_div()

        # Driver requires vertical_scale_factor to be a float or list of 8 (all channels)
        # Map display V/div → hardware gain for each channel
        vscales_hw = [_hw_vscale_for(vscales_dict[ch]) for ch in range(8)]

        trig_ch = self._controls.get_trigger_channel()
        trig_adc = self._volts_to_adc(self._trigger_level_volts, vscales_dict[trig_ch], trig_ch)

        self._initialized = False
        self._roll_mode = Hantek1008.is_roll_mode_ns_per_div(ns)
        # Trigger / horizontal-trigger markers are meaningless in roll mode
        # (the device free-runs and streams), so hide them while rolling.
        self._trigger_marker.setVisible(not self._roll_mode)
        self._h_trigger_marker.setVisible(not self._roll_mode)
        self._roll_buf = {}
        self._roll_head = 0

        # The hardware buffer holds 4000 shorts total; with N channels interleaved
        # each channel gets 4000/N samples per burst.
        frame_size_per_ch = 4000 // len(hw_active)
        new_display_samples = self._compute_display_samples(frame_size_per_ch, ns)
        if self._display_samples > 0 and new_display_samples > 1:
            fraction = int(self._h_trigger_marker.value()) / max(1, self._display_samples - 1)
            initial_pre = max(0, min(int(round(fraction * (new_display_samples - 1))), new_display_samples - 1))
        else:
            initial_pre = new_display_samples // 2

        self._acq = AcquisitionThread(
            ns_per_div=ns,
            active_channels=hw_active,
            vscales=vscales_hw,
            trigger_channel=trig_ch,
            trigger_slope=self._controls.get_trigger_slope(),
            trigger_level=trig_adc,
            initial_pre_samples=initial_pre,
            device=self._device,
            capture_mode=self._controls.get_acq_mode(),
        )
        self._acq.new_frame.connect(self.on_new_frame)
        self._acq.roll_chunk.connect(self.on_roll_chunk)
        self._acq.device_ready.connect(self._on_device_ready)
        self._acq.error.connect(self._on_error)
        self._acq.start()

    def _restart_acquisition(self):
        if self._restart_in_progress:
            self._pending_restart = True
            return
        self._restart_in_progress = True
        self._pending_restart = False
        try:
            if self._acq is not None:
                self._acq.stop()
                self._restarting = True
                self._set_status("● Updating", "#ffaa00")
                QApplication.processEvents()
                self._acq.wait()
                QApplication.processEvents()
            self._start_acquisition()
            if self._controls.get_acq_mode() == "stopped":
                self._set_status("● Stopped", "#ff5555")
        finally:
            self._restart_in_progress = False
            if self._pending_restart:
                QTimer.singleShot(0, self._restart_acquisition)

    def _set_status(self, text, color):
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            f"color: {color}; font-size: 11px; background-color: #111111;"
            "border-bottom: 1px solid #333333;"
        )

    def on_new_frame(self, data, triggered):
        if self._restarting:
            return                        # discard stale frames from the dying thread
        mode = self._controls.get_acq_mode()
        if mode == "stopped":
            return                        # single-shot captured; display is frozen

        expected = set(self._controls.get_active_channels())
        # data may include silent partner channels (hardware pair padding); filter them out
        data = {k: v for k, v in data.items() if k in expected}
        if set(data.keys()) != expected:
            return

        frame_size = len(next(iter(data.values())))

        if not self._initialized:
            self._init_buffer(frame_size)
            self._initialized = True
            self._set_status("● Live", "#44cc44")

        self._last_frame_triggered = triggered
        self._last_frame_np = {
            ch_id: np.asarray(samples_list, dtype=np.float32)
            for ch_id, samples_list in data.items()
        }
        self._redraw()
        if self._autoset_searching:
            QTimer.singleShot(0, self._autoset_step)

    def _begin_autoset(self):
        self._autoset_searching = True
        self._controls.start_autoset_animation()

    def _end_autoset(self):
        self._autoset_searching = False
        self._controls.stop_autoset_animation()

    def _on_autoset(self):
        if not self._last_frame_np or self._frame_size == 0:
            return
        ns_per_div = self._controls.get_ns_per_div()
        try:
            self._autoset_current_idx = NS_PER_DIV_VALUES.index(ns_per_div)
        except ValueError:
            self._autoset_current_idx = 0
        self._begin_autoset()
        self._autoset_step()

    def _autoset_step(self):
        ns_per_div = self._controls.get_ns_per_div()
        if ns_per_div <= FAST_FIXED_NS_PER_DIV_MAX:
            ns_per_sample = ADC_MIN_NS_PER_SAMPLE
        else:
            ns_per_sample = max(
                (ns_per_div * TIME_DIVS) / self._frame_size,
                SLOW_BURST_NS_PER_SAMPLE,
            )
        ref_ch = self._controls.get_autoset_ref_channel()
        if ref_ch is not None:
            candidates = {k: v for k, v in self._last_frame_np.items() if k == ref_ch}
        else:
            candidates = self._last_frame_np
        periods = [
            p for p in (
                _estimate_period_ns(samples, ns_per_sample)
                for samples in candidates.values()
            )
            if p is not None
        ]
        if periods:
            period_ns = float(np.median(periods))
            target_ns_per_div = period_ns * 5 / TIME_DIVS  # show ~5 full cycles
            best = min(NS_PER_DIV_VALUES, key=lambda v: abs(v - target_ns_per_div))
            log.info("AutoSet: found period=%.1f ns at %d ns/div -> best=%d ns/div",
                     period_ns, ns_per_div, best)
            self._end_autoset()
            self._autoset_changing_timebase = True
            self._controls.set_ns_per_div(best)
            self._autoset_changing_timebase = False
        else:
            next_idx = self._autoset_current_idx + 1
            if next_idx >= len(NS_PER_DIV_VALUES):
                log.info("AutoSet: no signal found at max timebase, giving up")
                self._end_autoset()
            else:
                next_ns = NS_PER_DIV_VALUES[next_idx]
                log.info("AutoSet: no signal at %d ns/div, trying %d ns/div", ns_per_div, next_ns)
                self._autoset_current_idx = next_idx
                self._autoset_changing_timebase = True
                self._controls.set_ns_per_div(next_ns)
                self._autoset_changing_timebase = False

    def _init_roll_display(self):
        """Set up the rolling-trace display: time-base, grid, curves and the
        per-channel ring buffers that the sweeping write head fills."""
        ns_per_div = self._controls.get_ns_per_div()
        self._display_samples, samples_per_div = self._compute_display_geometry(0, ns_per_div)
        self._samples_per_div = samples_per_div
        self._frame_size = self._display_samples
        self._time_axis.set_timebase(ns_per_div, samples_per_div)
        self._cursor_overlay.set_timebase(ns_per_div / samples_per_div if samples_per_div else 0.0)
        log.info("_init_roll_display: ns/div=%d -> display_samples=%d samples_per_div=%.2f",
                 ns_per_div, self._display_samples, samples_per_div)

        self._plot_widget.setXRange(0, self._display_samples - 1, padding=0)

        pi = self._plot_widget.getPlotItem()
        grid_pen = pg.mkPen(GRID_COLOR, width=1)
        for line in self._v_grid_lines:
            pi.removeItem(line)
        self._v_grid_lines.clear()
        for i in range(1, TIME_DIVS):
            pos = i * samples_per_div
            if pos >= self._display_samples:
                break
            line = pg.InfiniteLine(pos=pos, angle=90, pen=grid_pen, movable=False)
            pi.addItem(line)
            self._v_grid_lines.append(line)

        for info in self._channel_data.values():
            self._plot_widget.removeItem(info["curve"])
        self._channel_data.clear()
        self._roll_buf = {}
        self._roll_head = 0

        active = self._controls.get_active_channels()
        for ch in active:
            color = CHANNEL_COLORS[ch]
            offset = self._channel_offsets.get(ch, 0.0)
            # connect="finite" so the unfilled (NaN) part of the buffer and the
            # sweeping head gap render as gaps rather than spurious lines.
            curve = self._plot_widget.plot(pen=pg.mkPen(color, width=1), connect="finite")
            curve.setPos(0, offset)
            self._channel_data[ch] = {"curve": curve}
            self._roll_buf[ch] = np.full(self._display_samples, np.nan, dtype=np.float32)

        vscales_dict = self._controls.get_vscales()
        active_vscales = [vscales_dict[ch] for ch in active]
        yrange = _yrange_for(active_vscales)
        margin_channels = {ch: (self._channel_offsets.get(ch, 0.0), CHANNEL_COLORS[ch])
                           for ch in active}
        self._ch_margin.set_channels(margin_channels, yrange)

    def on_roll_chunk(self, data):
        if self._restarting or not self._roll_mode:
            return

        expected = set(self._controls.get_active_channels())
        # filter out silent partner channels from hardware pair padding
        data = {k: v for k, v in data.items() if k in expected}
        if set(data.keys()) != expected:
            return

        if not self._initialized:
            self._init_roll_display()
            self._initialized = True
            self._set_status("● Roll", "#44cc44")

        n = self._display_samples
        # All active channels advance together, so the chunk length is the same
        # for each; use the shortest to stay in lock-step.
        count = min(len(v) for v in data.values())
        if count == 0:
            return

        head = self._roll_head
        for ch, samples in data.items():
            buf = self._roll_buf.get(ch)
            if buf is None:
                continue
            arr = np.asarray(samples[:count], dtype=np.float32)
            idx = (head + np.arange(count)) % n
            buf[idx] = arr

        self._roll_head = (head + count) % n
        # Blank a couple of samples just ahead of the write head so the sweep
        # point reads as a moving gap, like the vendor app.
        gap = (self._roll_head + np.arange(2)) % n
        for buf in self._roll_buf.values():
            buf[gap] = np.nan

        self._redraw_roll()

    def _redraw_roll(self):
        for ch, buf in self._roll_buf.items():
            info = self._channel_data.get(ch)
            if info is not None:
                info["curve"].setData(buf, connect="finite")

    def _on_time_div_changed(self, ns):
        if not self._autoset_changing_timebase:
            self._end_autoset()  # user manually changed timebase; cancel any search
        log.info("===== time/div -> %d ns/div (autoset=%s) =====", ns, self._autoset_searching)
        self._restart_acquisition()

    def _on_channel_toggled(self, ch_idx, is_on):
        if is_on and ch_idx not in self._channel_offsets:
            # Stagger first-time default: each channel half a grid div lower than the previous
            vscale = self._controls.get_vscales()[ch_idx]
            self._channel_offsets[ch_idx] = -ch_idx * 0.5 * vscale
        self._update_yrange()
        self._update_trigger_marker_label()   # trigger may have auto-moved
        self._restart_acquisition()

    def _on_trigger_channel_changed(self, ch_idx):
        self._update_trigger_marker_label()
        self._restart_acquisition()

    def _on_trigger_slope_changed(self, slope):
        self._restart_acquisition()

    def _on_acq_mode_changed(self, mode):
        if not self._roll_mode:
            # Trigger markers stay hidden while rolling (no trigger applies).
            self._trigger_marker.setVisible(True)
            self._h_trigger_marker.setVisible(True)
        if self._acq is not None:
            self._acq.set_capture_mode(mode)
        if self._roll_mode:
            return                            # acquisition mode has no effect in roll mode
        if mode == "single":
            self._set_status("● Armed", "#ffaa00")
        elif self._initialized:
            self._set_status("● Live", "#44cc44")

    def _on_vscale_changed(self, ch_idx, vscale):
        self._update_yrange()
        self._restart_acquisition()

    def _on_cursor_toggled(self, enabled):
        self._cursor_overlay.set_enabled(enabled)

    def _on_device_ready(self, zero_offsets):
        self._zero_offsets = zero_offsets
        # Guard: _acq._device is set to None in the thread's finally block when it
        # exits.  A device_ready signal queued before exit and processed after it
        # (during QApplication.processEvents() in _restart_acquisition) must not
        # overwrite self._device with None, which would force an unnecessary full
        # USB reconnect on the next restart and trigger EBUSY.
        dev = self._acq._device
        if dev is not None:
            self._device = dev

    def _on_error(self, msg):
        self._end_autoset()
        print(f"Device error: {msg}", file=sys.stderr)
        if self._device is not None:
            # Do NOT call close() here — it sends a USB reset which causes the
            # device to briefly disappear (ENODEV) and then re-enumerate, making
            # the next connect() fail with EBUSY.  Just drop the reference and
            # force GC so pyusb releases the interface claim before the reconnect.
            self._device = None
            import gc
            gc.collect()
        disconnected = "19" in msg or "No such device" in msg
        label, color = ("● Disconnected", "#ff6666") if disconnected else ("● Error", "#ff6666")
        self._status_label.setText(label)
        self._status_label.setStyleSheet(
            f"color: {color}; font-size: 11px; background-color: #111111;"
            "border-bottom: 1px solid #333333;"
        )
        if disconnected:
            # Retry automatically — the device re-enumerates after a brief pause.
            QTimer.singleShot(2000, self._restart_acquisition)

    def closeEvent(self, event):
        if self._acq is not None:
            self._acq.stop()
            self._acq.wait()
        if self._device is not None:
            self._device.close()
            self._device = None
        super().closeEvent(event)

