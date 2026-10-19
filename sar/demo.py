"""Run with python -m sar.demo --output outputs/pulse_compression.png."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from . import lfm_chirp, pulse_compress, sidelobe_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/pulse_compression.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    fs, bandwidth, duration = 80e6, 10e6, 20e-6
    t, chirp = lfm_chirp(duration, bandwidth, fs)
    delay = 320
    received = np.pad(chirp, (delay, 400))
    compressed = pulse_compress(received, chirp)
    metrics = sidelobe_metrics(compressed)
    lag = (np.arange(compressed.size) - chirp.size + 1) / fs
    amplitude_db = 20 * np.log10(np.maximum(np.abs(compressed) / np.abs(compressed).max(), 1e-6))
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), constrained_layout=True)
    axes[0].plot(t * 1e6, chirp.real, label="Real")
    axes[0].plot(t * 1e6, chirp.imag, alpha=0.6, label="Imaginary")
    axes[0].set(xlabel="Time (µs)", ylabel="Amplitude", title="Complex baseband LFM chirp")
    axes[0].legend()
    axes[1].plot(np.arange(received.size) / fs * 1e6, np.abs(received))
    axes[1].set(xlabel="Time (µs)", ylabel="Magnitude", title="Delayed point-target echo")
    axes[2].plot(lag * 1e6, amplitude_db)
    lo, hi = metrics.mainlobe_bounds
    axes[2].axvspan(lag[lo] * 1e6, lag[hi] * 1e6, alpha=0.2, color="orange", label="Mainlobe")
    axes[2].set(xlabel="Delay (µs)", ylabel="Normalized magnitude (dB)", ylim=(-65, 2),
                xlim=(delay / fs * 1e6 - 2, delay / fs * 1e6 + 2),
                title=f"Pulse compression: PSLR {metrics.pslr_db:.2f} dB; ISLR {metrics.islr_db:.2f} dB")
    axes[2].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"Peak delay: {lag[metrics.peak_index] * 1e6:.3f} µs")
    print(f"PSLR: {metrics.pslr_db:.2f} dB; ISLR: {metrics.islr_db:.2f} dB")
    print(f"Saved {args.output}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
