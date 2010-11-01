import numpy as np

import spectrum

from nose.tools import eq_, ok_, set_trace

def test_feature():
    X,Y = np.ogrid[-1:1:100j, -1:1:100j]
    F = X * Y * np.exp(-10*(X**2 + Y**2))
    Fxy = spectrum.hessian(F)
    Fxy_exact = np.exp(-10*(X**2 + Y**2)) * (-20*X**2 - 20*Y**2 + 400*X**2*Y**2 + 1.0)
    scale = np.max(Fxy_exact) / np.max(Fxy)
    Fxy = Fxy * scale
    err = np.sum(np.abs(Fxy - Fxy_exact))

def gen_data(N, a=10):
    X,Y = np.ogrid[-1:1:complex(0,N), -1:1:complex(0,N)]
    F = X * Y * np.exp(-a*(X**2 + Y**2))
    return F
