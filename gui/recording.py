import json
import queue
import struct
import threading
import time
import bisect

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6 import sip

try:
    from PyQt6.QtMultimedia import QAudioSource, QAudioSink, QAudioFormat, QMediaDevices
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

MAGIC = b"HSREC001"
INDEX_MAGIC = b"HSIDX001"
VERSION = 1
HEADER_SIZE = 64
EV_CONFIG = 1
EV_FRAME = 2
EV_ROLL = 3
EV_AUDIO = 4
_KNOWN = (EV_CONFIG, EV_FRAME, EV_ROLL, EV_AUDIO)
AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
_AUDIO_HDR = struct.Struct("<IBBH")

_EVENT_HDR = struct.Struct("<IBBHQ")
_KF = struct.Struct("<QQII")
INDEX_ENV_MAGIC = b"HSENV001"
_ENV = struct.Struct("<QH")


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    return obj


def _pack_header(index_offset, start_wall_ns):
    buf = bytearray(HEADER_SIZE)
    buf[0:8] = MAGIC
    struct.pack_into("<H", buf, 8, VERSION)
    struct.pack_into("<QQ", buf, 12, index_offset, start_wall_ns)
    return buf


def _pack_channels(data, triggered):
    items = sorted((int(ch), samples) for ch, samples in data.items())
    trig = 255 if triggered is None else (1 if triggered else 0)
    parts = [struct.pack("<BB", trig, len(items))]
    for ch, samples in items:
        arr = np.ascontiguousarray(np.asarray(samples, dtype=np.float32))
        parts.append(struct.pack("<BI", ch, int(arr.size)))
        parts.append(arr.tobytes())
    return b"".join(parts)


def _unpack_channels(payload):
    trig, nch = struct.unpack_from("<BB", payload, 0)
    off = 2
    data = {}
    for _ in range(nch):
        ch, n = struct.unpack_from("<BI", payload, off)
        off += 5
        data[int(ch)] = np.frombuffer(payload, dtype=np.float32, count=n, offset=off).copy()
        off += n * 4
    triggered = None if trig == 255 else bool(trig)
    return data, triggered


def _pack_audio(pcm, rate, channels):
    return _AUDIO_HDR.pack(int(rate), int(channels), 1, 0) + pcm


def _unpack_audio(payload):
    rate, channels, _fmt, _res = _AUDIO_HDR.unpack_from(payload, 0)
    return payload[_AUDIO_HDR.size:], int(rate), int(channels)


def _pcm_peak(pcm):
    n = len(pcm) // 2
    if n <= 0:
        return 0
    samples = np.frombuffer(pcm, dtype="<i2", count=n)
    return int(np.max(np.abs(samples.astype(np.int32))))


def _audio_format():
    fmt = QAudioFormat()
    fmt.setSampleRate(AUDIO_RATE)
    fmt.setChannelCount(AUDIO_CHANNELS)
    fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
    return fmt


class MicRecorder(QObject):
    def __init__(self, recorder, parent=None):
        super().__init__(parent)
        self._rec = recorder
        self._src = None
        self._io = None
        self._buf = bytearray()
        self._rate = AUDIO_RATE
        self._channels = AUDIO_CHANNELS
        self._chunk = AUDIO_RATE * 2 * AUDIO_CHANNELS // 10
        self._muted = True

    def set_muted(self, on):
        self._muted = bool(on)
        if self._muted:
            self._buf.clear()

    def start(self):
        if not HAS_AUDIO:
            return False
        dev = QMediaDevices.defaultAudioInput()
        if dev.isNull():
            return False
        fmt = _audio_format()
        self._src = QAudioSource(dev, fmt, self)
        self._io = self._src.start()
        if self._io is None:
            self._src = None
            return False
        actual = self._src.format()
        self._rate = actual.sampleRate() or AUDIO_RATE
        self._channels = actual.channelCount() or AUDIO_CHANNELS
        bytes_ps = max(1, self._rate * self._channels * 2)
        self._chunk = max(bytes_ps // 10, 320)
        self._io.readyRead.connect(self._on_ready)
        return True

    def _on_ready(self):
        if self._io is None:
            return
        data = bytes(self._io.readAll())
        if self._muted:
            self._buf.clear()
            return
        self._buf.extend(data)
        while len(self._buf) >= self._chunk:
            piece = bytes(self._buf[:self._chunk])
            del self._buf[:self._chunk]
            self._rec.write_audio(piece, self._rate, self._channels)

    def stop(self):
        if self._buf and self._rec is not None and not self._muted:
            self._rec.write_audio(bytes(self._buf), self._rate, self._channels)
        self._buf.clear()
        if self._src is not None:
            self._src.stop()
            self._src.deleteLater()
            self._src = None
            self._io = None


def _qt_alive(obj):
    return obj is not None and not sip.isdeleted(obj)


class AudioPlayer(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._sink = None
        self._io = None
        self._rate = AUDIO_RATE
        self._channels = AUDIO_CHANNELS
        self._playing = True

    def _restart_io(self):
        if not _qt_alive(self._sink):
            self._sink = None
            self._io = None
            return False
        self._io = self._sink.start()
        if not _qt_alive(self._io):
            self._io = None
            return False
        if not self._playing:
            self._sink.suspend()
        return True

    def _ensure(self, rate, channels):
        if not HAS_AUDIO:
            return False
        if _qt_alive(self._sink) and self._rate == rate and self._channels == channels:
            if _qt_alive(self._io):
                return True
            return self._restart_io()
        self.stop()
        dev = QMediaDevices.defaultAudioOutput()
        if dev.isNull():
            return False
        fmt = QAudioFormat()
        fmt.setSampleRate(rate)
        fmt.setChannelCount(channels)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        self._sink = QAudioSink(dev, fmt, self)
        self._sink.setBufferSize(rate * channels * 2)
        self._rate = rate
        self._channels = channels
        return self._restart_io()

    def write(self, pcm, rate, channels):
        if not self._playing:
            return
        if not self._ensure(rate, channels):
            return
        try:
            self._io.write(pcm)
        except RuntimeError:
            self._io = None
            if self._ensure(rate, channels):
                self._io.write(pcm)

    def reset(self):
        if not _qt_alive(self._sink):
            self._sink = None
            self._io = None
            return
        self._io = None
        self._sink.reset()
        self._restart_io()

    def set_playing(self, on):
        self._playing = on
        if not _qt_alive(self._sink):
            return
        if on:
            self._sink.resume()
        else:
            self._sink.suspend()

    def stop(self):
        self._io = None
        if _qt_alive(self._sink):
            self._sink.stop()
            self._sink.deleteLater()
        self._sink = None


def _concat_trim(parts, need):
    if not parts:
        return {}
    channels = set()
    for p in parts:
        channels.update(p.keys())
    out = {}
    for ch in channels:
        segs = [p[ch] for p in parts if ch in p]
        cat = np.concatenate(segs) if len(segs) > 1 else segs[0]
        if need > 0 and len(cat) > need:
            cat = cat[-need:]
        out[ch] = cat
    return out


class Recorder(QObject):
    error = pyqtSignal(str)

    def __init__(self, path, snapshot, parent=None):
        super().__init__(parent)
        self._path = path
        self._q = queue.Queue()
        self._t0 = time.monotonic_ns()
        self._start_wall = time.time_ns()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.write_config(snapshot)

    def _now(self):
        return time.monotonic_ns() - self._t0

    def write_config(self, snap):
        payload = json.dumps(_clean(snap), separators=(",", ":")).encode()
        self._q.put((EV_CONFIG, self._now(), payload))

    def write_frame(self, data, triggered):
        self._q.put((EV_FRAME, self._now(), _pack_channels(data, triggered)))

    def write_roll(self, data):
        self._q.put((EV_ROLL, self._now(), _pack_channels(data, None)))

    def write_audio(self, pcm, rate, channels):
        n = len(pcm)
        if n < 2:
            return
        bytes_ps = max(1, int(rate) * int(channels) * 2)
        dur_ns = n * 1_000_000_000 // bytes_ps
        t_ns = max(0, self._now() - dur_ns)
        self._q.put((EV_AUDIO, t_ns, _pack_audio(pcm, rate, channels)))

    def close(self):
        self._q.put(None)
        self._thread.join(timeout=30)

    def _run(self):
        configs = []
        keyframes = []
        envelope = []
        duration = 0
        try:
            with open(self._path, "wb") as f:
                f.write(_pack_header(0, self._start_wall))
                f.flush()
                while True:
                    item = self._q.get()
                    if item is None:
                        break
                    typ, t_ns, payload = item
                    offset = f.tell()
                    f.write(_EVENT_HDR.pack(len(payload), typ, 0, 0, t_ns))
                    f.write(payload)
                    f.flush()
                    if typ == EV_CONFIG:
                        configs.append((t_ns, json.loads(payload.decode())))
                    elif typ == EV_AUDIO:
                        pcm, _rate, _ch = _unpack_audio(payload)
                        envelope.append((t_ns, _pcm_peak(pcm)))
                    cfg_idx = len(configs) - 1
                    if cfg_idx >= 0:
                        keyframes.append((t_ns, offset, cfg_idx, typ))
                    duration = t_ns
                index_offset = f.tell()
                f.write(INDEX_MAGIC)
                f.write(struct.pack("<Q", duration))
                f.write(struct.pack("<I", len(configs)))
                for t_ns, snap in configs:
                    raw = json.dumps(snap, separators=(",", ":")).encode()
                    f.write(struct.pack("<QI", t_ns, len(raw)))
                    f.write(raw)
                f.write(struct.pack("<I", len(keyframes)))
                for t_ns, offset, cfg_idx, typ in keyframes:
                    f.write(_KF.pack(t_ns, offset, cfg_idx, typ))
                f.write(INDEX_ENV_MAGIC)
                f.write(struct.pack("<I", len(envelope)))
                for t_ns, peak in envelope:
                    f.write(_ENV.pack(t_ns, min(peak, 65535)))
                f.flush()
                f.seek(0)
                f.write(_pack_header(index_offset, self._start_wall))
                f.flush()
        except Exception as e:
            self.error.emit(str(e))


class RecordingReader:
    def __init__(self, path):
        self._f = open(path, "rb")
        hdr = self._f.read(HEADER_SIZE)
        if len(hdr) < HEADER_SIZE or hdr[0:8] != MAGIC:
            self._f.close()
            raise ValueError("not a hanscope recording")
        version = struct.unpack_from("<H", hdr, 8)[0]
        if version != VERSION:
            self._f.close()
            raise ValueError(f"unsupported recording version {version}")
        self._index_offset, self.start_wall_ns = struct.unpack_from("<QQ", hdr, 12)
        self.configs = []
        self.keyframes = []
        self.envelope = []
        self.duration_ns = 0
        if self._index_offset and self._load_index():
            pass
        else:
            self._rebuild_index()
        self._kf_times = [k[0] for k in self.keyframes]
        self._f.seek(HEADER_SIZE)

    def _load_index(self):
        try:
            self._f.seek(self._index_offset)
            mag = self._f.read(8)
            if mag != INDEX_MAGIC:
                return False
            self.duration_ns = struct.unpack("<Q", self._f.read(8))[0]
            n_cfg = struct.unpack("<I", self._f.read(4))[0]
            configs = []
            for _ in range(n_cfg):
                t_ns, n = struct.unpack("<QI", self._f.read(12))
                snap = json.loads(self._f.read(n).decode())
                configs.append(snap)
            n_kf = struct.unpack("<I", self._f.read(4))[0]
            keyframes = []
            for _ in range(n_kf):
                t_ns, offset, cfg_idx, typ = _KF.unpack(self._f.read(_KF.size))
                keyframes.append((t_ns, offset, cfg_idx, typ))
            self.configs = configs
            self.keyframes = keyframes
            mag = self._f.read(8)
            if mag == INDEX_ENV_MAGIC:
                n_env = struct.unpack("<I", self._f.read(4))[0]
                env = []
                for _ in range(n_env):
                    t_ns, peak = _ENV.unpack(self._f.read(_ENV.size))
                    env.append((t_ns, peak))
                self.envelope = env
            else:
                self._scan_envelope()
            return True
        except Exception:
            return False

    def _scan_envelope(self):
        env = []
        for t_ns, offset, _cfg, typ in self.keyframes:
            if typ != EV_AUDIO:
                continue
            ev = self.peek_event(offset)
            if ev is None:
                continue
            pcm, _rate, _ch = _unpack_audio(ev[2])
            env.append((t_ns, _pcm_peak(pcm)))
        self.envelope = env

    def _rebuild_index(self):
        self._f.seek(HEADER_SIZE)
        configs = []
        keyframes = []
        envelope = []
        duration = 0
        while True:
            offset = self._f.tell()
            ev = self.read_event()
            if ev is None:
                break
            typ, t_ns, payload = ev
            if typ == EV_CONFIG:
                configs.append(json.loads(payload.decode()))
            elif typ == EV_AUDIO:
                pcm, _rate, _ch = _unpack_audio(payload)
                envelope.append((t_ns, _pcm_peak(pcm)))
            cfg_idx = len(configs) - 1
            if cfg_idx >= 0:
                keyframes.append((t_ns, offset, cfg_idx, typ))
            duration = t_ns
        self.configs = configs
        self.keyframes = keyframes
        self.envelope = envelope
        self.duration_ns = duration

    def read_event(self):
        while True:
            hdr = self._f.read(_EVENT_HDR.size)
            if len(hdr) < _EVENT_HDR.size:
                return None
            payload_len, typ, _flags, _res, t_ns = _EVENT_HDR.unpack(hdr)
            payload = self._f.read(payload_len)
            if len(payload) < payload_len:
                return None
            if typ not in _KNOWN:
                continue
            return typ, t_ns, payload

    def seek_offset(self, offset):
        self._f.seek(offset)

    def peek_event(self, offset):
        cur = self._f.tell()
        self._f.seek(offset)
        ev = self.read_event()
        self._f.seek(cur)
        return ev

    def close(self):
        self._f.close()


class PlaybackThread(QThread):
    new_frame = pyqtSignal(dict, bool)
    roll_chunk = pyqtSignal(dict)
    apply_config = pyqtSignal(dict)
    audio_chunk = pyqtSignal(object, int, int)
    audio_reset = pyqtSignal()
    position = pyqtSignal(object)
    ended = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self._reader = RecordingReader(path)
        self.duration_ns = self._reader.duration_ns
        self.envelope = self._reader.envelope
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._playing = True
        self._stop = False
        self._seek_to = None
        self._playhead = 0
        self._origin = time.monotonic_ns()

    def is_playing(self):
        with self._lock:
            return self._playing

    def play(self):
        with self._lock:
            if self.duration_ns and self._playhead >= self.duration_ns:
                self._seek_to = 0
                self._playhead = 0
            self._playing = True
            self._origin = time.monotonic_ns() - self._playhead
        self._wake.set()

    def pause(self):
        with self._lock:
            if self._playing:
                self._playhead = max(0, time.monotonic_ns() - self._origin)
            self._playing = False
        self._wake.set()

    def seek(self, ns):
        ns = max(0, min(int(ns), self.duration_ns))
        with self._lock:
            self._seek_to = ns
        self._wake.set()

    def stop(self):
        self._stop = True
        self._wake.set()

    def run(self):
        pending = None
        try:
            if self._reader.configs:
                self.apply_config.emit(self._reader.configs[0])
            while not self._stop:
                with self._lock:
                    seek = self._seek_to
                    self._seek_to = None
                    playing = self._playing
                if seek is not None:
                    pending = self._apply_seek(seek)
                    with self._lock:
                        self._playhead = seek
                        if self._playing:
                            self._origin = time.monotonic_ns() - seek
                    self.position.emit(seek)
                    continue
                if not playing:
                    self._wake.clear()
                    self._wake.wait(0.05)
                    continue
                if pending is None:
                    pending = self._reader.read_event()
                if pending is None:
                    with self._lock:
                        self._playing = False
                        self._playhead = self.duration_ns
                    self.position.emit(self.duration_ns)
                    self.ended.emit()
                    continue
                _typ, t_ns, _payload = pending
                with self._lock:
                    origin = self._origin
                deadline = origin + t_ns
                interrupted = False
                while True:
                    if self._stop:
                        return
                    with self._lock:
                        if self._seek_to is not None or not self._playing:
                            interrupted = True
                            break
                    now = time.monotonic_ns()
                    remaining = deadline - now
                    if remaining <= 0:
                        break
                    self.position.emit(now - origin)
                    self._wake.clear()
                    self._wake.wait(min(remaining / 1e9, 0.05))
                if interrupted:
                    continue
                self._dispatch(pending)
                with self._lock:
                    self._playhead = t_ns
                self.position.emit(t_ns)
                pending = None
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self._reader.close()

    def _last_of(self, i, types):
        kfs = self._reader.keyframes
        while i >= 0:
            if kfs[i][3] in types:
                return i
            i -= 1
        return -1

    def _apply_seek(self, target):
        kfs = self._reader.keyframes
        times = self._reader._kf_times
        if not kfs:
            return None
        i = bisect.bisect_right(times, target) - 1
        if i < 0:
            i = 0
        _t_ns, offset, cfg_idx, typ = kfs[i]
        snap = self._reader.configs[cfg_idx]
        self.apply_config.emit(snap)
        self.audio_reset.emit()
        vis_i = self._last_of(i, (EV_FRAME, EV_ROLL))
        consumed_extra = False
        if vis_i >= 0 and kfs[vis_i][3] == EV_ROLL:
            preload = self._roll_preload(vis_i, snap)
            if preload:
                self.roll_chunk.emit(preload)
        elif vis_i >= 0:
            ev = self._reader.peek_event(kfs[vis_i][1])
            if ev is not None:
                self._dispatch(ev)
        elif typ == EV_CONFIG:
            self._reader.seek_offset(offset)
            self._reader.read_event()
            nxt = self._reader.read_event()
            if nxt is not None and nxt[0] in (EV_FRAME, EV_ROLL):
                self._dispatch(nxt)
                consumed_extra = True
        if self.is_playing():
            self._emit_audio_tail(i, target)
        self._reader.seek_offset(offset)
        self._reader.read_event()
        nxt = self._reader.read_event()
        if consumed_extra and nxt is not None:
            nxt = self._reader.read_event()
        return nxt

    def _emit_audio_tail(self, i, target):
        a = self._last_of(i, (EV_AUDIO,))
        if a < 0:
            return
        ev = self._reader.peek_event(self._reader.keyframes[a][1])
        if ev is None:
            return
        pcm, rate, channels = _unpack_audio(ev[2])
        t0 = ev[1]
        bytes_ps = max(1, rate * channels * 2)
        skip = int((target - t0) * bytes_ps / 1_000_000_000)
        skip = max(0, skip - (skip % (channels * 2)))
        if skip < len(pcm):
            self.audio_chunk.emit(pcm[skip:], rate, channels)

    def _roll_preload(self, i, snap):
        need = int(snap.get("display_samples") or 0)
        cfg_idx = self._reader.keyframes[i][2]
        parts = []
        got = 0
        j = i
        while j >= 0:
            _t, off, cidx, typ = self._reader.keyframes[j]
            if typ != EV_ROLL or cidx != cfg_idx:
                break
            ev = self._reader.peek_event(off)
            if ev is None:
                break
            data, _trig = _unpack_channels(ev[2])
            parts.append(data)
            if data:
                got += len(next(iter(data.values())))
            if not need or got >= need:
                break
            j -= 1
        parts.reverse()
        return _concat_trim(parts, need)

    def _dispatch(self, ev):
        typ, _t_ns, payload = ev
        if typ == EV_CONFIG:
            self.apply_config.emit(json.loads(payload.decode()))
        elif typ == EV_FRAME:
            data, triggered = _unpack_channels(payload)
            self.new_frame.emit(data, bool(triggered))
        elif typ == EV_ROLL:
            data, _trig = _unpack_channels(payload)
            self.roll_chunk.emit(data)
        elif typ == EV_AUDIO:
            pcm, rate, channels = _unpack_audio(payload)
            self.audio_chunk.emit(pcm, rate, channels)
