# SAR signal processing

A small Python project for complex baseband linear frequency modulated (LFM)
waveforms and SAR **range** pulse compression. Uses NumPy, SciPy, and Matplotlib.
This is a starting point for signal processing, rather than a full SAR image
formation pipeline.

## Install and run

Requires Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m sar.demo
python -m unittest discover -s tests -v
```

The demo saves `outputs/pulse_compression.png`, showing the chirp, a delayed
point-target echo, and the compressed response with its mainlobe highlighted.
Use `python -m sar.demo --show` for an interactive window, or `--output path.png`
to choose the destination. On a headless machine:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/sar-matplotlib python -m sar.demo
```

## Example

```python
import numpy as np
from sar import lfm_chirp, matched_filter, pulse_compress, sidelobe_metrics

fs = 80e6
seconds, chirp = lfm_chirp(duration=20e-6, bandwidth=10e6, sample_rate=fs)
received = np.pad(chirp, (320, 400))  # 4 µs delay, noiseless point target
kernel = matched_filter(chirp)
compressed = pulse_compress(received, chirp)
metrics = sidelobe_metrics(compressed)
delay_seconds = (metrics.peak_index - len(chirp) + 1) / fs
print(delay_seconds, metrics.pslr_db, metrics.islr_db)
```

## Conventions

- The unit-amplitude chirp is `exp(j*pi*(B/T)*t**2)` with a centered time axis.
  Bandwidth, duration, and sample rate must be positive; `fs > B` and `T*fs`
  must be an integer of at least two samples. Sampling omits the sweep endpoints.
- The matched filter is the conjugate time-reversed reference. Compression uses
  full linear FFT convolution, without circular wraparound or normalization.
  A unit-amplitude echo peaks at the reference energy (`sum(abs(reference)**2)`).
- Output index `k` represents lag `k - (len(reference) - 1)` samples. Divide by
  sample rate for delay, or multiply delay by propagation speed / 2 for range.
- **PSLR** = `10*log10(max sidelobe power / peak mainlobe power)`.
- **ISLR** = `10*log10(sum sidelobe power / sum mainlobe power)`.
  Both are normally negative; lower values indicate smaller sidelobes.
- Mainlobe bounds are inclusive, using the closest sampled local minimum on
  either side of the global peak by default. Pass `mainlobe_bounds=(lo, hi)`
  for explicit bounds, especially with noise, multiple targets, or windowing.
  Missing minima raise an error. Zero sidelobe power gives negative infinity.
  All samples outside the bounds contribute to the metrics, so observation
  length, sampling, and noise affect the result. Measure isolated target
  responses for meaningful waveform sidelobe characterization.

The default rectangular chirp has time-bandwidth product 200. Expect a peak
near 4 µs, PSLR near −13.3 dB, and ISLR near −10 dB; finite sampled chirps differ
from ideal sinc responses. No taper, noise, or propagation loss is applied.

## Layout

- `sar/processing.py`: chirp generation, matched filter, pulse compression
- `sar/metrics.py`: PSLR and ISLR with reported mainlobe bounds
- `sar/demo.py`: synthetic point-target experiment and plotting
- `tests/test_processing.py`: waveform, convolution, timing, and metric checks
