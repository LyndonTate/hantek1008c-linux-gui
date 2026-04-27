import sys
import logging
import numpy as np
import pyqtgraph as pg

log = logging.getLogger(__name__)
from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QApplication
from PyQt6.QtCore import Qt

from gui.controls import ControlsPanel, CHANNEL_COLORS
from gui.acquisition import AcquisitionThread
from gui.channel_margin import ChannelMarginWidget

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


class ScopeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hanscope")
        self.setMinimumSize(1280, 720)
        self.resize(1280, 720)

        self._display_samples = 0
        self._initialized = False
        self._restarting = False         # True while switching timebases; discards stale frames
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

        self._setup_ui()
        self._start_acquisition()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._plot_widget = pg.PlotWidget()

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

        self._controls.time_div_changed.connect(self._on_time_div_changed)
        self._controls.channel_toggled.connect(self._on_channel_toggled)
        self._controls.vscale_changed.connect(self._on_vscale_changed)
        self._controls.trigger_channel_changed.connect(self._on_trigger_channel_changed)

    def _setup_plot(self):
        self._plot_widget.setBackground("#000000")
        pi = self._plot_widget.getPlotItem()
        pi.hideAxis("left")
        pi.hideAxis("bottom")
        pi.setMenuEnabled(False)
        self._plot_widget.setMouseEnabled(x=False, y=False)
        self._update_yrange()

    def _update_yrange(self):
        vscales_dict = self._controls.get_vscales()
        active = self._controls.get_active_channels()
        active_vscales = [vscales_dict[ch] for ch in active]
        yrange = _yrange_for(active_vscales)
        self._plot_widget.setYRange(-yrange, yrange, padding=0)
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
        if self._frame_size > self._display_samples:
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
            self._channel_data[ch_id]["curve"].setData(samples[start:end])

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
            trigger_slope="rising",
            trigger_level=trig_adc,
            initial_pre_samples=initial_pre,
            device=self._device,
        )
        self._acq.new_frame.connect(self.on_new_frame)
        self._acq.device_ready.connect(self._on_device_ready)
        self._acq.error.connect(self._on_error)
        self._acq.start()

    def _restart_acquisition(self):
        if self._acq is not None:
            self._acq.stop()
            self._restarting = True       # discard any frames that arrive from here on
            self._status_label.setText("● Updating")
            self._status_label.setStyleSheet(
                "color: #ffaa00; font-size: 11px; background-color: #111111;"
                "border-bottom: 1px solid #333333;"
            )
            QApplication.processEvents()  # paint the status indicator immediately
            self._acq.wait()
            QApplication.processEvents()  # drain signals queued while wait() was blocking
        self._start_acquisition()

    def on_new_frame(self, data):
        if self._restarting:
            return                        # discard stale frames from the dying thread

        expected = set(self._controls.get_active_channels())
        # data may include silent partner channels (hardware pair padding); filter them out
        data = {k: v for k, v in data.items() if k in expected}
        if set(data.keys()) != expected:
            return

        frame_size = len(next(iter(data.values())))

        if not self._initialized:
            self._init_buffer(frame_size)
            self._initialized = True
            self._status_label.setText("● Live")
            self._status_label.setStyleSheet(
                "color: #44cc44; font-size: 11px; background-color: #111111;"
                "border-bottom: 1px solid #333333;"
            )

        self._last_frame_np = {
            ch_id: np.asarray(samples_list, dtype=np.float32)
            for ch_id, samples_list in data.items()
        }
        self._redraw()

    def _on_time_div_changed(self, ns):
        log.info("===== USER changed time/div -> %d ns/div =====", ns)
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

    def _on_vscale_changed(self, ch_idx, vscale):
        self._update_yrange()
        self._restart_acquisition()

    def _on_device_ready(self, zero_offsets):
        self._zero_offsets = zero_offsets
        self._device = self._acq._device  # keep a reference for reconfiguration on next restart

    def _on_error(self, msg):
        print(f"Device error: {msg}", file=sys.stderr)
        # Invalidate the cached device so the next restart does a full connect/init.
        self._device = None

    def closeEvent(self, event):
        if self._acq is not None:
            self._acq.stop()
            self._acq.wait()
        if self._device is not None:
            self._device.close()
            self._device = None
        super().closeEvent(event)

