from _rad_prof import rad_prof

import numpy as np

from nose.tools import eq_, ok_, set_trace


def test_rp():
    def inner(npts, nprofile, sigma, maxrad, useG):
        Y = np.linspace(0.0, 1.0, npts, endpoint=False)
        Y -= .5
        X = Y[:,np.newaxis]
        if useG:
            arr = np.exp(-(X**2 + Y**2) / (2. * sigma * sigma))
            rp_calc = np.exp(-X[:,0]**2 / (2. * sigma * sigma))
            center = np.where(arr == arr.max())
            center_x, center_y = float(center[0]) / npts, float(center[1]) / npts
        else:
            arr = np.ones((npts, npts))
            rp_calc = arr[:,0]
            center_x, center_y = 0.5, 0.5
        profile = np.empty(nprofile, dtype=np.double)
        rad_prof(arr=arr, scale=1.0,
                 center_x=center_x, center_y=center_y,
                 max_rad=maxrad, profile=profile)
        return arr, profile, rp_calc

    maxrad = 1.0
    nprofile = 100
    arr1, pro1, rp_calc1 = inner(100, nprofile, 0.05, maxrad, useG=True)
    arr2, pro2, rp_calc2 = inner(100, nprofile, 0.1, maxrad, useG=True)

    X1 = np.linspace(0, maxrad, nprofile)
    X_calc = np.linspace(0, maxrad, len(rp_calc1[50:]))

    # print arr1.sum(), pro1.sum()
    # print arr2.sum(), pro2.sum()

    # slopes = [(pro1[i] - pro1[0]) / (X1[i] - X1[0]) for i in range(1, len(pro1))]
    # print slopes

    import pylab as pl
    pl.ion()
    pl.plot(X1, pro1)
    pl.plot(X_calc, rp_calc1[50:])
    # from pprint import pprint
    # pprint(zip(X1, pro1))
    # print pro1[:50] / rp_calc1[50:]
    pl.figure()
    pl.plot(X1, pro2)
    pl.plot(X_calc, rp_calc2[50:])
    # print rp_calc2[49:] / pro2[:50]
    set_trace()


    ok_(np.allclose(np.sum(pro1), arr1.sum()))
    ok_(np.allclose(np.sum(pro2), arr2.sum()))
