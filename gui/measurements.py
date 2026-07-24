import numpy as np

MEASURE_GROUPS = [
    ("horizontal", "Horizontal", [
        ("period", "Period", "Period"),
        ("freq", "Freq", "Frequency"),
        # ("rise", "Rise", "Rise Time"),
        # ("fall", "Fall", "Fall Time"),
        # ("duty_pos", "+Duty", "+ Duty Cycle"),
        # ("duty_neg", "−Duty", "− Duty Cycle"),
        # ("pw_pos", "+PW", "+ Pulse Width"),
        # ("pw_neg", "−PW", "− Pulse Width"),
    ]),
    ("vertical", "Vertical", [
        ("vmax", "Max", "Maximum"),
        ("vmin", "Min", "Minimum"),
        ("vpp", "PkPk", "Peak to Peak"),
        # ("vtop", "Top", "Top"),
        # ("vbase", "Base", "Base"),
        # ("vmiddle", "Mid", "Middle"),
        ("vrms", "RMS", "RMS"),
        # ("vamp", "Amp", "Amplitude"),
        ("vmean", "Mean", "Mean"),
        # ("vcycmean", "CMean", "Cycle Mean"),
        # ("vos_pos", "+OS", "Positive Overshoot"),
        # ("vos_neg", "−OS", "Negative Overshoot"),
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


def _mean_period_samples(samples):
    y = _finite_samples(samples)
    if y is None or y.size < 4:
        return None
    y_min = float(y.min())
    y_max = float(y.max())
    amp = y_max - y_min
    if amp < 1e-6:
        return None
    mid = 0.5 * (y_min + y_max)
    hyst = 0.1 * amp
    low = mid - hyst
    high = mid + hyst
    edges = []
    armed = False
    for i, v in enumerate(y):
        if not armed:
            if v < low:
                armed = True
        elif v > high:
            edges.append(i)
            armed = False
    if len(edges) < 2:
        return None
    periods = np.diff(np.asarray(edges, dtype=np.float64))
    mean_period = float(periods.mean())
    if mean_period <= 0:
        return None
    return mean_period


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


_MEASURERS = {
    "freq": measure_frequency,
    "period": measure_period,
    "vpp": measure_vpp,
    "vmax": measure_vmax,
    "vmin": measure_vmin,
    "vmean": measure_vmean,
    "vrms": measure_vrms,
}


_FORMATTERS = {
    "freq": format_freq,
    "period": format_time,
    "vpp": format_volt,
    "vmax": format_volt,
    "vmin": format_volt,
    "vmean": format_volt,
    "vrms": format_volt,
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
