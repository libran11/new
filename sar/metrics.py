"""Sidelobe measurements using explicit or first-minimum mainlobe bounds."""
from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks
from .processing import _vector


@dataclass(frozen=True)
class SidelobeMetrics:
    pslr_db: float
    islr_db: float
    peak_index: int
    mainlobe_bounds: tuple[int, int]


def sidelobe_metrics(response, mainlobe_bounds=None):
    """Measure PSLR and ISLR as sidelobe/mainlobe power ratios in dB.

    Bounds are inclusive. By default the nearest sampled local minimum on
    each side of the global peak defines the mainlobe, including both minima.
    All other samples form the sidelobes. Supply bounds for ambiguous/noisy
    responses. Metrics describe the entire provided observation window.
    """
    power = np.abs(_vector(response, "response"))**2
    peak = int(np.argmax(power))
    if power[peak] == 0:
        raise ValueError("response must have nonzero energy")
    if mainlobe_bounds is None:
        minima = find_peaks(-power)[0]
        left, right = minima[minima < peak], minima[minima > peak]
        if not left.size or not right.size:
            raise ValueError("cannot locate mainlobe minima; supply mainlobe_bounds")
        lo, hi = int(left[-1]), int(right[0])
    else:
        lo, hi = mainlobe_bounds
        if not isinstance(lo, (int, np.integer)) or not isinstance(hi, (int, np.integer)):
            raise ValueError("mainlobe bounds must be integer indices")
        if not 0 <= lo <= peak <= hi < power.size:
            raise ValueError("mainlobe bounds must contain the peak and lie within response")
    side = np.concatenate((power[:lo], power[hi + 1:]))
    if not side.size:
        raise ValueError("at least one sidelobe sample is required")
    def db(ratio):
        return float(10 * np.log10(ratio)) if ratio > 0 else float("-inf")
    return SidelobeMetrics(db(side.max() / power[peak]),
                           db(side.sum() / power[lo:hi + 1].sum()), peak, (lo, hi))
