"""LFM waveform generation and linear matched filtering."""
import numpy as np
from scipy.signal import fftconvolve


def _vector(values, name):
    values = np.asarray(values, dtype=complex)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a nonempty, finite 1-D array")
    return values


def lfm_chirp(duration, bandwidth, sample_rate):
    """Return (time, unit-amplitude complex chirp), centered about t=0.

    Positive bandwidth produces an up-chirp sweeping approximately -B/2 to B/2.
    duration * sample_rate must be an integer number of samples.
    """
    if not all(np.isfinite(x) and x > 0 for x in (duration, bandwidth, sample_rate)):
        raise ValueError("duration, bandwidth and sample_rate must be positive and finite")
    if bandwidth >= sample_rate:
        raise ValueError("sample_rate must exceed the baseband bandwidth")
    count = duration * sample_rate
    n = int(round(count))
    if n < 2 or not np.isclose(count, n, rtol=0, atol=1e-6):
        raise ValueError("duration * sample_rate must be an integer >= 2")
    time = (np.arange(n) - (n - 1) / 2) / sample_rate
    return time, np.exp(1j * np.pi * bandwidth / duration * time**2)


def matched_filter(reference):
    """Return the conjugate time-reversed reference (no energy normalization)."""
    return np.conj(_vector(reference, "reference")[::-1])


def pulse_compress(received, reference):
    """Return full linear convolution with the reference's matched filter.

    Output index k corresponds to delay k - (len(reference) - 1) samples.
    An isolated unit-amplitude reference has peak equal to reference energy.
    """
    return fftconvolve(_vector(received, "received"), matched_filter(reference), mode="full")
