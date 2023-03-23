import unittest

import numpy as np

from lie_learn.spectral.SE2FFT import SE2_FFT, shift_fft, shift_ifft

class TestSE2FFT(unittest.TestCase):
    def test_SE2FFT(self):
        f = np.zeros((40, 40, 42))
        f[19:21, 10:30, :] = 1.

        F = SE2_FFT(spatial_grid_size=(40, 40, 42),
                    interpolation_method='Fourier',
                    spline_order=1,
                    oversampling_factor=5)

        f, f1c, f1p, f2, f2f, fh = F.analyze(f)

        fi, f1ci, f1pi, f2i, f2fi, fhi = F.synthesize(fh)

        self.assertTrue(np.sum(np.abs(fh-fhi)) < 1e-5)
        self.assertTrue(np.sum(np.abs(f2f-f2fi)) < 1e-5)
        self.assertTrue(np.sum(np.abs(f2-f2i)) < 1e-5)
        self.assertTrue(np.sum(np.abs(f1p-f1pi)) < 1e-5)
        self.assertTrue(np.sum(np.abs(f1c-f1ci)) < 1e-5)
        self.assertTrue(np.sum(np.abs(f-fi)) < 1e-5)

    def test_resample(self):
        f = np.zeros((40, 40, 42))
        f[19:21, 10:30, :] = 1.

        F = SE2_FFT(spatial_grid_size=(40, 40, 42),
                    interpolation_method='spline',
                    spline_order=1,
                    oversampling_factor=5)

        f1c = shift_fft(f)
        f1p = F.resample_c2p_3d(f1c)

        f1ci = F.resample_p2c_3d(f1p)
        fi = shift_ifft(f1ci)

        self.assertTrue(np.sum(np.abs(f1c-f1ci)) < 1e-5)
        self.assertTrue(np.sum(np.abs(f-fi)) < 1e-5)


if __name__ == '__main__':
    unittest.main()
