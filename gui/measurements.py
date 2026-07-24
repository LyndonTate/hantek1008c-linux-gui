import numpy as np

MEASURE_GROUPS = [
    ("horizontal", "Horizontal", [
        ("period", "Period", "Period"),
        ("freq", "Freq", "Frequency"),
        ("rise", "Rise", "Rise Time"),
        ("fall", "Fall", "Fall Time"),
        ("duty_pos", "+Duty", "+ Duty Cycle"),
        ("duty_neg", "−Duty", "− Duty Cycle"),
        ("pw_pos", "+PW", "+ Pulse Width"),
        ("pw_neg", "−PW", "− Pulse Width"),
    ]),
    ("vertical", "Vertical", [
        ("vmax", "Max", "Maximum"),
        ("vmin", "Min", "Minimum"),
        ("vpp", "PkPk", "Peak to Peak"),
        ("vtop", "Top", "Top"),
        ("vbase", "Base", "Base"),
        ("vmiddle", "Mid", "Middle"),
        ("vrms", "RMS", "RMS"),
        ("vamp", "Amp", "Amplitude"),
        ("vmean", "Mean", "Mean"),
        ("vcycmean", "CMean", "Cycle Mean"),
        ("vos_pos", "+OS", "Positive Overshoot"),
        ("vos_neg", "−OS", "Negative Overshoot"),
    ]),
]

MEASURE_TYPES = [
    (mid, short, full)
    for _, _, items in MEASURE_GROUPS
    for mid, short, full in items
]


def _finite_samples(samples):
    y = np.asarray(samples, dtype=np.float64)
    if y.size == 0:
        return None
    finite = np.isfinite(y)
    if not finite.any():
        return None
    if finite.all():
        return y
    mask = finite.astype(np.int8)
    padded = np.concatenate(([0], mask, [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    if starts.size == 0:
        return None
    lengths = ends - starts
    i = int(np.argmax(lengths))
    return y[starts[i]:ends[i]]


def format_freq(hz):
    if hz is None or not np.isfinite(hz) or hz <= 0:
        return "—"
    if hz >= 1_000_000:
        return f"{hz / 1_000_000:.4g}MHz"
    if hz >= 1_000:
        return f"{hz / 1_000:.4g}kHz"
    return f"{hz:.4g}Hz"


def format_time(ns):
    if ns is None or not np.isfinite(ns) or ns <= 0:
        return "—"
    if ns >= 1_000_000_000:
        return f"{ns / 1_000_000_000:.4g}s"
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.4g}ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.4g}µs"
    return f"{ns:.4g}ns"


def format_volt(v):
    if v is None or not np.isfinite(v):
        return "—"
    av = abs(v)
    if av < 1.0:
        return f"{v * 1000:.4g}mV"
    return f"{v:.4g}V"


def format_percent(p):
    if p is None or not np.isfinite(p):
        return "—"
    return f"{p:.4g}%"


def _hysteresis_levels(y):
    y_min = float(y.min())
    y_max = float(y.max())
    amp = y_max - y_min
    if amp < 1e-6:
        return None
    mid = 0.5 * (y_min + y_max)
    hyst = 0.1 * amp
    return mid - hyst, mid + hyst


def _rising_edges(y, low, high):
    edges = []
    armed = False
    for i, v in enumerate(y):
        if not armed:
            if v < low:
                armed = True
        elif v > high:
            edges.append(i)
            armed = False
    return edges


def _falling_edges(y, low, high):
    edges = []
    armed = False
    for i, v in enumerate(y):
        if not armed:
            if v > high:
                armed = True
        elif v < low:
            edges.append(i)
            armed = False
    return edges


def _mean_period_samples(samples):
    y = _finite_samples(samples)
    if y is None or y.size < 4:
        return None
    levels = _hysteresis_levels(y)
    if levels is None:
        return None
    low, high = levels
    edges = _rising_edges(y, low, high)
    if len(edges) < 2:
        return None
    periods = np.diff(np.asarray(edges, dtype=np.float64))
    mean_period = float(periods.mean())
    if mean_period <= 0:
        return None
    return mean_period


def _mean_pulse_widths_samples(samples):
    y = _finite_samples(samples)
    if y is None or y.size < 4:
        return None
    levels = _hysteresis_levels(y)
    if levels is None:
        return None
    low, high = levels
    rises = _rising_edges(y, low, high)
    falls = _falling_edges(y, low, high)
    if len(rises) < 2 or not falls:
        return None
    fi = 0
    high_ws = []
    low_ws = []
    for ri in range(len(rises) - 1):
        r0 = rises[ri]
        r1 = rises[ri + 1]
        while fi < len(falls) and falls[fi] <= r0:
            fi += 1
        if fi >= len(falls) or falls[fi] >= r1:
            continue
        period = r1 - r0
        if period <= 0:
            continue
        high_w = falls[fi] - r0
        high_ws.append(high_w)
        low_ws.append(period - high_w)
    if not high_ws:
        return None
    return float(np.mean(high_ws)), float(np.mean(low_ws))


def _mean_duty_pos(samples):
    widths = _mean_pulse_widths_samples(samples)
    if widths is None:
        return None
    high_w, low_w = widths
    period = high_w + low_w
    if period <= 0:
        return None
    return 100.0 * high_w / period


def measure_frequency(samples, ns_per_sample):
    if ns_per_sample is None or ns_per_sample <= 0:
        return None
    mean_period = _mean_period_samples(samples)
    if mean_period is None:
        return None
    return 1e9 / (mean_period * ns_per_sample)


def measure_period(samples, ns_per_sample):
    if ns_per_sample is None or ns_per_sample <= 0:
        return None
    mean_period = _mean_period_samples(samples)
    if mean_period is None:
        return None
    return mean_period * ns_per_sample


def measure_duty_pos(samples, ns_per_sample):
    return _mean_duty_pos(samples)


def measure_duty_neg(samples, ns_per_sample):
    d = _mean_duty_pos(samples)
    if d is None:
        return None
    return 100.0 - d


def measure_pw_pos(samples, ns_per_sample):
    if ns_per_sample is None or ns_per_sample <= 0:
        return None
    widths = _mean_pulse_widths_samples(samples)
    if widths is None:
        return None
    return widths[0] * ns_per_sample


def measure_pw_neg(samples, ns_per_sample):
    if ns_per_sample is None or ns_per_sample <= 0:
        return None
    widths = _mean_pulse_widths_samples(samples)
    if widths is None:
        return None
    return widths[1] * ns_per_sample


def measure_vpp(samples, ns_per_sample):
    y = _finite_samples(samples)
    if y is None or y.size < 1:
        return None
    return float(y.max() - y.min())


def measure_vmax(samples, ns_per_sample):
    y = _finite_samples(samples)
    if y is None or y.size < 1:
        return None
    return float(y.max())


def measure_vmin(samples, ns_per_sample):
    y = _finite_samples(samples)
    if y is None or y.size < 1:
        return None
    return float(y.min())


def measure_vmean(samples, ns_per_sample):
    y = _finite_samples(samples)
    if y is None or y.size < 1:
        return None
    return float(y.mean())


def measure_vrms(samples, ns_per_sample):
    y = _finite_samples(samples)
    if y is None or y.size < 1:
        return None
    return float(np.sqrt(np.mean(y * y)))


def _top_base(samples):
    y = _finite_samples(samples)
    if y is None or y.size < 2:
        return None
    y_min = float(y.min())
    y_max = float(y.max())
    if y_max - y_min < 1e-6:
        return y_min, y_max
    n_bins = int(min(256, max(32, y.size // 16)))
    counts, edges = np.histogram(y, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mid = 0.5 * (y_min + y_max)
    low_idx = np.where(centers <= mid)[0]
    high_idx = np.where(centers > mid)[0]
    if low_idx.size == 0 or high_idx.size == 0:
        return y_min, y_max
    base = float(centers[low_idx[np.argmax(counts[low_idx])]])
    top = float(centers[high_idx[np.argmax(counts[high_idx])]])
    if top < base:
        base, top = top, base
    return base, top


def measure_vtop(samples, ns_per_sample):
    tb = _top_base(samples)
    if tb is None:
        return None
    return tb[1]


def measure_vbase(samples, ns_per_sample):
    tb = _top_base(samples)
    if tb is None:
        return None
    return tb[0]


def measure_vamp(samples, ns_per_sample):
    tb = _top_base(samples)
    if tb is None:
        return None
    return tb[1] - tb[0]


def measure_vmiddle(samples, ns_per_sample):
    tb = _top_base(samples)
    if tb is None:
        return None
    return 0.5 * (tb[0] + tb[1])


def _interp_cross(y0, y1, level):
    dy = y1 - y0
    if dy == 0:
        return 0.0
    return (level - y0) / dy


def _edge_times_10_90(samples, rising):
    y = _finite_samples(samples)
    if y is None or y.size < 4:
        return None
    tb = _top_base(samples)
    if tb is None:
        return None
    base, top = tb
    amp = top - base
    if amp < 1e-6:
        return None
    lo = base + 0.1 * amp
    hi = base + 0.9 * amp
    n = y.size
    times = []
    i = 0
    while i < n - 1:
        t_first = None
        while i < n - 1:
            a, b = y[i], y[i + 1]
            if rising:
                if a <= lo < b or (a < lo and b >= lo):
                    t_first = i + _interp_cross(a, b, lo)
                    i += 1
                    break
            else:
                if a >= hi > b or (a > hi and b <= hi):
                    t_first = i + _interp_cross(a, b, hi)
                    i += 1
                    break
            i += 1
        if t_first is None:
            break
        t_second = None
        j = int(t_first)
        while j < n - 1:
            a, b = y[j], y[j + 1]
            if rising:
                if a <= hi < b or (a < hi and b >= hi):
                    t_second = j + _interp_cross(a, b, hi)
                    break
                if b < lo:
                    break
            else:
                if a >= lo > b or (a > lo and b <= lo):
                    t_second = j + _interp_cross(a, b, lo)
                    break
                if b > hi:
                    break
            j += 1
        if t_second is not None and t_second > t_first:
            times.append(t_second - t_first)
            i = j + 1
    if not times:
        return None
    return float(np.mean(times))


def measure_rise(samples, ns_per_sample):
    if ns_per_sample is None or ns_per_sample <= 0:
        return None
    dt = _edge_times_10_90(samples, True)
    if dt is None:
        return None
    return dt * ns_per_sample


def measure_fall(samples, ns_per_sample):
    if ns_per_sample is None or ns_per_sample <= 0:
        return None
    dt = _edge_times_10_90(samples, False)
    if dt is None:
        return None
    return dt * ns_per_sample


def measure_vcycmean(samples, ns_per_sample):
    y = _finite_samples(samples)
    if y is None or y.size < 4:
        return None
    levels = _hysteresis_levels(y)
    if levels is None:
        return None
    low, high = levels
    edges = _rising_edges(y, low, high)
    if len(edges) < 2:
        return None
    start = edges[0]
    end = edges[-1]
    if end <= start:
        return None
    return float(y[start:end].mean())


def measure_vos_pos(samples, ns_per_sample):
    y = _finite_samples(samples)
    if y is None or y.size < 2:
        return None
    tb = _top_base(samples)
    if tb is None:
        return None
    base, top = tb
    amp = top - base
    if amp < 1e-6:
        return None
    return 100.0 * (float(y.max()) - top) / amp


def measure_vos_neg(samples, ns_per_sample):
    y = _finite_samples(samples)
    if y is None or y.size < 2:
        return None
    tb = _top_base(samples)
    if tb is None:
        return None
    base, top = tb
    amp = top - base
    if amp < 1e-6:
        return None
    return 100.0 * (base - float(y.min())) / amp


_MEASURERS = {
    "freq": measure_frequency,
    "period": measure_period,
    "duty_pos": measure_duty_pos,
    "duty_neg": measure_duty_neg,
    "pw_pos": measure_pw_pos,
    "pw_neg": measure_pw_neg,
    "rise": measure_rise,
    "fall": measure_fall,
    "vcycmean": measure_vcycmean,
    "vos_pos": measure_vos_pos,
    "vos_neg": measure_vos_neg,
    "vpp": measure_vpp,
    "vmax": measure_vmax,
    "vmin": measure_vmin,
    "vmean": measure_vmean,
    "vrms": measure_vrms,
    "vtop": measure_vtop,
    "vbase": measure_vbase,
    "vamp": measure_vamp,
    "vmiddle": measure_vmiddle,
}


_FORMATTERS = {
    "freq": format_freq,
    "period": format_time,
    "duty_pos": format_percent,
    "duty_neg": format_percent,
    "pw_pos": format_time,
    "pw_neg": format_time,
    "rise": format_time,
    "fall": format_time,
    "vpp": format_volt,
    "vmax": format_volt,
    "vmin": format_volt,
    "vmean": format_volt,
    "vrms": format_volt,
    "vtop": format_volt,
    "vbase": format_volt,
    "vamp": format_volt,
    "vmiddle": format_volt,
    "vcycmean": format_volt,
    "vos_pos": format_percent,
    "vos_neg": format_percent,
}


def measure_value(measure_id, samples, ns_per_sample):
    fn = _MEASURERS.get(measure_id)
    if fn is None:
        return None
    return fn(samples, ns_per_sample)


def format_measure(measure_id, value):
    fn = _FORMATTERS.get(measure_id)
    if fn is None:
        return "—" if value is None else f"{value:.4g}"
    return fn(value)
