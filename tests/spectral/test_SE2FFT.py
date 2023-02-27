import numpy as np

from lie_learn.spectral.SE2FFT import SE2_FFT

def test_SE2FFT():
    f = np.zeros((40, 40, 42))
    f[19:21, 10:30, :] = 1.

    F = SE2_FFT(spatial_grid_size=(40, 40, 42),
                interpolation_method='spline',
                spline_order=1,
                oversampling_factor=5)

    f, f1c, f1p, f2, f2f, fh = F.analyze(f)

    fi, f1ci, f1pi, f2i, f2fi, fhi = F.synthesize(fh)

    assert np.isclose(np.sum(np.abs(fh-fhi)), 0.)
    assert np.isclose(np.sum(np.abs(f2f-f2fi)), 0.)
    assert np.isclose(np.sum(np.abs(f2-f2i)), 0.)
    assert np.isclose(np.sum(np.abs(f1p-f1pi)), 0.)
    assert np.isclose(np.sum(np.abs(f1c-f1ci)), 0.)
    assert np.isclose(np.sum(np.abs(f-fi)), 0.)

