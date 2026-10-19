import unittest
import numpy as np
from numpy.testing import assert_allclose
from sar import lfm_chirp, matched_filter, pulse_compress, sidelobe_metrics


class ProcessingTests(unittest.TestCase):
    def test_chirp_phase_and_amplitude(self):
        t, x = lfm_chirp(20e-6, 10e6, 80e6)
        self.assertEqual(len(x), 1600)
        assert_allclose(np.abs(x), 1)
        frequency = np.diff(np.unwrap(np.angle(x))) * 80e6 / (2 * np.pi)
        assert_allclose(frequency, 10e6 / 20e-6 * (t[:-1] + t[1:]) / 2, atol=1e-5)

    def test_filter_convolution(self):
        reference = np.array([1 + 2j, 3 - 1j, -2j])
        received = np.array([2j, 1, 4 + 1j, 0])
        assert_allclose(matched_filter(reference), reference[::-1].conj())
        assert_allclose(pulse_compress(received, reference),
                        np.convolve(received, reference[::-1].conj()), atol=1e-12)

    def test_delayed_chirp_compression(self):
        _, x = lfm_chirp(20e-6, 10e6, 80e6)
        compressed = pulse_compress(np.pad(x, (320, 400)), x)
        metrics = sidelobe_metrics(compressed)
        self.assertEqual(metrics.peak_index, 320 + len(x) - 1)
        self.assertAlmostEqual(abs(compressed[metrics.peak_index]), len(x), places=8)
        self.assertTrue(-14 < metrics.pslr_db < -12)
        self.assertTrue(-11 < metrics.islr_db < -9)

    def test_metrics_known_power_ratios(self):
        result = sidelobe_metrics([1, 0, 2, 0, 1], (1, 3))
        self.assertAlmostEqual(result.pslr_db, 10 * np.log10(1 / 4))
        self.assertAlmostEqual(result.islr_db, 10 * np.log10(2 / 4))
        self.assertEqual(sidelobe_metrics([0, 2, 0], (1, 1)).pslr_db, -np.inf)

    def test_invalid_inputs(self):
        for args in [(0, 1, 2), (1, 2, 2), (1.1, 1, 3)]:
            with self.assertRaises(ValueError):
                lfm_chirp(*args)
        for response in ([0, 0, 0], [1, 2, 1]):
            with self.assertRaises(ValueError):
                sidelobe_metrics(response)
        with self.assertRaises(ValueError):
            pulse_compress([], [1])
        with self.assertRaises(ValueError):
            sidelobe_metrics([1, 2, 1], (0, 0))


if __name__ == '__main__':
    unittest.main()
