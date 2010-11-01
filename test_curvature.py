import numpy as np
import curvature
import test_spectrum

from nose.tools import ok_, eq_, set_trace

def test_gauscurv_zero():
    zz = np.zeros((100,100), dtype='f')
    gc = curvature.gaus_curv(zz)
    ok_(np.allclose(zz, gc))

def test_meancurv():
    def meancurv_gen(N,m,n):
        rarr = test_spectrum.sin_cos_arr(N,m,n)
        mc = curvature.mean_curv(rarr)
        X,Y = np.ogrid[0:N, 0:N]
        X -= N/2; Y -= N/2
        Fx  =  (1.0*m)    * np.cos(2*m*np.pi*X/N)
        Fy  =  (1.0*n)    * np.sin(2*n*np.pi*Y/N)
        Fxx = -(1.0*m)**2 * np.sin(2*m*np.pi*X/N)
        Fyy =  (1.0*n)**2 * np.cos(2*n*np.pi*Y/N)

        denom = np.power(Fx**2 + Fy**2 + 1, 1.5)
        calc_mc = 0.5 * ((1+Fx**2)*Fyy + (1+Fy**2)*Fxx) / denom

        ok_(np.allclose(calc_mc, mc))



    for N in (128,):
        for m in xrange(5):
            for n in xrange(5):
                yield (meancurv_gen, N, m, n)

def test_gauscurv():

    def gauscurv_gen(N,m,n):
        rarr = test_spectrum.sin_cos_arr(N,m,n)
        gc = curvature.gaus_curv(rarr)
        X,Y = np.ogrid[0:N, 0:N]
        X -= N/2; Y -= N/2
        Fxx = -(1.0*m)**2 * np.sin(2*m*np.pi*X/N)
        Fyy =  (1.0*n)**2 * np.cos(2*n*np.pi*Y/N)
        Fx  =  (1.0*m)    * np.cos(2*m*np.pi*X/N)
        Fy  =  (1.0*n)    * np.sin(2*n*np.pi*Y/N)

        denom = (Fx**2 + Fy**2 + 1)**2
        calc_gc = Fxx*Fyy / denom

        ok_(np.allclose(calc_gc, gc))

    for N in (512,):
        for m in xrange(5):
            for n in xrange(5):
                yield (gauscurv_gen, N, m, n)
