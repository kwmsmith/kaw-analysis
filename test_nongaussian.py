import scipy
from scipy import stats
import numpy as np
import nongaussian as ng

from nose.tools import ok_, eq_, assert_almost_equal

np.random.seed(10)

def test_ng_zeros():
    all_zeros = np.zeros((100,100))
    nong_mask = ng.nongaussian_filter(all_zeros)
    
    eq_(nong_mask.size, all_zeros.size)
    ok_(not np.any(nong_mask))

def test_ng_dists():
    NN = 10**6
    for arr, superg, name in [
            (stats.logistic.rvs(size=NN), True, 'logistic'),
            (stats.uniform.rvs(size=NN), False, 'uniform'),
            (stats.norm.rvs(size=NN), False, 'norm'),
            (stats.bernoulli.rvs(0.9, size=NN), True, 'bernoulli')]:
        ng_dist.description = name
        yield ng_dist, arr, superg

def ng_dist(arr, supergaussian):
    NN = arr.size
    nong_mask = ng.nongaussian_filter(arr)

    lmean = arr.mean()
    lvar = arr.var()

    gauss_resid = arr[~nong_mask] - lmean
    nongauss_resid = arr[nong_mask] - lmean

    gauss_kurt = np.sum(gauss_resid**4) / (lvar**2 * NN)
    nongauss_kurt = np.sum(nongauss_resid**4) / (lvar**2 * NN)

    true_kurt = stats.kurtosis(arr)

    assert_almost_equal(gauss_kurt + nongauss_kurt - 3.0, true_kurt, places=6)

    if supergaussian:
        assert_almost_equal(gauss_kurt - 3.0, 0.0, places=2)
        assert_almost_equal(nongauss_kurt, true_kurt, places=2)
    else:
        assert_almost_equal(gauss_kurt - 3.0, true_kurt, places=2)
        assert_almost_equal(nongauss_kurt, 0.0, places=2)
