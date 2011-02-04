from . import spectrum as spect
import numpy as np

from nose.tools import ok_, eq_, set_trace

def test_get_spectrum_zeros():
    carr = np.zeros((10,5),dtype='F')
    x,sp = spect.get_spectrum(carr, 5)
    import pylab as pl
    pl.ion()
    pl.plot(x, sp)
    set_trace()

def test_get_spectrum_ones():
    carr = np.zeros((200,100),dtype='F')+1.0j
    npts = 101
    x,sp = spect.get_spectrum(carr, npts)

    approx = (np.pi*x)/npts*x[-1] + sp[0]

    assert np.allclose(sum(sp), sum(abs(carr.flatten())))
    import pylab as pl
    pl.ion()
    pl.plot(x, sp)
    set_trace()
