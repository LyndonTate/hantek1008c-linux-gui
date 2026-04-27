import sys
sys.path.insert(0, ".")

from PyQt6.QtCore import QThread, pyqtSignal
from vendor.hantek1008 import Hantek1008


class AcquisitionThread(QThread):
    new_frame = pyqtSignal(dict)
    error = pyqtSignal(str)
    device_ready = pyqtSignal(dict)  # emits zero_offsets {vscale: [per-channel floats]}

    def __init__(self, ns_per_div=500_000, active_channels=None, vscales=None,
                 trigger_channel=0, trigger_slope="rising", trigger_level=2048,
                 initial_pre_samples=2016, device=None, free_run=False, parent=None):
        super().__init__(parent)
        self._ns_per_div = ns_per_div
        self._active_channels = active_channels or [0]
        self._vscales = vscales if vscales is not None else 1.0
        self._trigger_channel = trigger_channel
        self._trigger_slope = trigger_slope
        self._trigger_level = trigger_level
        self._initial_pre_samples = initial_pre_samples
        self._free_run = free_run
        self._running = False
        self._stop_requested = False  # set by stop() to abort before entering the burst loop
        self._device = None
        # If an already-initialised device is supplied, we reuse it (no connect/init).
        self._existing_device = device

    def run(self):
        if self._existing_device is not None:
            # Reuse an existing device — just reconfigure it, no USB reset.
            device = self._existing_device
            self._existing_device = None
            try:
                device.reconfigure(
                    active_channels=self._active_channels,
                    vscales=self._vscales,
                    ns_per_div=self._ns_per_div,
                    trigger_channel=self._trigger_channel,
                    trigger_slope=self._trigger_slope,
                    trigger_level=self._trigger_level,
                    pre_samples=self._initial_pre_samples,
                )
                self._device = device
                self.device_ready.emit(device.get_zero_offsets())
            except Exception as e:
                self.error.emit(str(e))
                return
        else:
            try:
                device = Hantek1008(
                    ns_per_div=self._ns_per_div,
                    vertical_scale_factor=self._vscales,
                    active_channels=self._active_channels,
                    trigger_channel=self._trigger_channel,
                    trigger_slope=self._trigger_slope,
                    trigger_level=self._trigger_level,
                )
                device.queue_hw_trigger_pre_samples(self._initial_pre_samples)
                device.connect()
                device.init()
                self._device = device
                self.device_ready.emit(device.get_zero_offsets())
            except Exception as e:
                self.error.emit(str(e))
                return

        device.set_free_run(self._free_run)

        self._running = True
        if self._stop_requested:
            self._device = None
            return
        try:
            while self._running:
                try:
                    data = device.request_samples_burst_mode()
                except RuntimeError:
                    # trigger level out of range — device timed out waiting for edge, retry
                    continue
                except Exception as e:
                    raise
                # Guard against empty frames (can occur in free-run mode if device
                # returns before a capture is ready)
                if not data or any(len(v) == 0 for v in data.values()):
                    continue
                self.new_frame.emit(data)
        finally:
            # ScopeWindow owns the device lifetime; we never close it here.
            self._device = None

    def queue_hw_trigger_pre_samples(self, pre_samples: int) -> None:
        if self._device is not None:
            self._device.queue_hw_trigger_pre_samples(pre_samples)

    def queue_trigger_level(self, level: int) -> None:
        if self._device is not None:
            self._device.queue_trigger_level(level)

    def set_free_run(self, enabled: bool) -> None:
        self._free_run = enabled
        if self._device is not None:
            self._device.set_free_run(enabled)

    def stop(self):
        self._stop_requested = True
        self._running = False
