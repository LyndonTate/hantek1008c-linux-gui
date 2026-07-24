import numpy as np

MEASURE_GROUPS = [
    ("horizontal", "Horizontal", [
        # ("period", "Period", "Period"),
        ("freq", "Freq", "Frequency"),
        # ("rise", "Rise", "Rise Time"),
        # ("fall", "Fall", "Fall Time"),
        # ("duty_pos", "+Duty", "+ Duty Cycle"),
        # ("duty_neg", "−Duty", "− Duty Cycle"),
        # ("pw_pos", "+PW", "+ Pulse Width"),
        # ("pw_neg", "−PW", "− Pulse Width"),
    ]),
    # ("vertical", "Vertical", [
    #     ("vmax", "Max", "Maximum"),
    #     ("vmin", "Min", "Minimum"),
    #     ("vpp", "PkPk", "Peak to Peak"),
    #     ("vtop", "Top", "Top"),
    #     ("vbase", "Base", "Base"),
    #     ("vmiddle", "Mid", "Middle"),
    #     ("vrms", "RMS", "RMS"),
    #     ("vamp", "Amp", "Amplitude"),
    #     ("vmean", "Mean", "Mean"),
    #     ("vcycmean", "CMean", "Cycle Mean"),
    #     ("vos_pos", "+OS", "Positive Overshoot"),
    #     ("vos_neg", "−OS", "Negative Overshoot"),
    # ]),
]

MEASURE_TYPES = [
    (mid, short, full)
    for _, _, items in MEASURE_GROUPS
    for mid, short, full in items
]


def format_freq(hz):
    if hz is None or not np.isfinite(hz) or hz <= 0:
        return "—"
    if hz >= 1_000_000:
        return f"{hz / 1_000_000:.4g}MHz"
    if hz >= 1_000:
        return f"{hz / 1_000:.4g}kHz"
    return f"{hz:.4g}Hz"


def measure_frequency(samples, ns_per_sample):
    if ns_per_sample is None or ns_per_sample <= 0:
        return None
    y = np.asarray(samples, dtype=np.float64)
    if y.size == 0:
        return None
    finite = np.isfinite(y)
    if not finite.any():
        return None
    if not finite.all():
        mask = finite.astype(np.int8)
        padded = np.concatenate(([0], mask, [0]))
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        if starts.size == 0:
            return None
        lengths = ends - starts
        i = int(np.argmax(lengths))
        y = y[starts[i]:ends[i]]
    if y.size < 4:
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
    return 1e9 / (mean_period * ns_per_sample)


_MEASURERS = {
    "freq": measure_frequency,
}


_FORMATTERS = {
    "freq": format_freq,
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
