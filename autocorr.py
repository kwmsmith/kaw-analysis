import numpy as np
import scipy as sp

from numpy.fft import rfft2, irfft2

def correlation(f, g):
    f_fft = rfft2(f)
    g_fft = rfft2(g)
    g_conj = np.conj(g_fft)
    prod = f_fft * g_conj
    return np.real(irfft2(prod))

def autocorr(f):
    return correlation(f,f)
