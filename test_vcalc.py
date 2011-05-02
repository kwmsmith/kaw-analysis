import numpy as np
import vcalc

from nose.tools import ok_, eq_

def sin_arr(N,m):
    return _trig_arr(N,m,np.sin)

def cos_arr(N,m):
    return _trig_arr(N,m,np.cos)

def _trig_arr(N,m, func):
    X,Y = np.mgrid[0:N, 0:N]
    X -= N/2.0; Y -= N/2.0
    return func(2*m*np.pi*X/N)

def sin_cos_arr(N,m,n):
    X,Y = np.ogrid[0:N, 0:N]
    X -= N/2; Y -= N/2
    return np.sin(2*m*np.pi*X/N) - np.cos(2*n*np.pi*Y/N)

def sin_cos_prod(N,m,n):
    X,Y = np.ogrid[0:N, 0:N]
    X -= N/2; Y -= N/2
    return np.sin(2*m*np.pi*X/N) * np.cos(2*n*np.pi*Y/N)

def test_mult_by_k1():
    carr = np.zeros((32,16),dtype='F')+1.0+1.0j
    ck0 = vcalc.mult_by_k(carr, 0)
    assert np.allclose(carr, ck0)
    ck1 = vcalc.mult_by_k(carr, 1)
    kx0_line = (np.zeros((16,),dtype='F')+1.0+1.0j) * np.arange(16)
    assert np.allclose(kx0_line, ck1[0])

def test_mult_by_k2():
    carr = np.zeros((32,16),dtype='F')+1.0+1.0j
    ck2 = vcalc.mult_by_k(carr, 2)
    kx_nxby2_line = (np.zeros((16,),dtype='F')+1.0+1.0j) * (np.arange(16)**2 + 16**2)
    assert np.allclose(kx_nxby2_line, ck2[16])


def test_cderivative():
    def cderiv_gen(N,m,n):
        rarr = sin_cos_arr(N,m,n)
        rarr_xx = vcalc.cderivative(rarr, 'X_DIR', order=2)
        rarr_yy = vcalc.cderivative(rarr, 'Y_DIR', order=2)
        rarr_laplacian = vcalc.laplacian(rarr)
        ok_(np.allclose(rarr_laplacian, rarr_xx + rarr_yy))

    for N in (128, 32):
        for m in xrange(N/8):
            for n in xrange(N/8):
                yield (cderiv_gen, N,m,n)

def test_cderiv2():
    def sin_gen(N, m):
        rarr = sin_arr(N,m)
        rarr_x = vcalc.cderivative(rarr, 'X_DIR', order=1) * rarr.size
        rarr_y = vcalc.cderivative(rarr, 'Y_DIR', order=1) * rarr.size
        ok_(np.allclose(rarr_y, np.zeros(rarr_y.shape)))
        ok_(np.allclose(rarr_x, m * cos_arr(N, m)))

    for N in (128, 32):
        for m in xrange(N/8):
            yield (sin_gen, N,m)

    def cos_gen(N, m):
        rarr = np.transpose(cos_arr(N,m))
        rarr_x = vcalc.cderivative(rarr, 'X_DIR', order=1) * rarr.size
        rarr_y = vcalc.cderivative(rarr, 'Y_DIR', order=1) * rarr.size
        ok_(np.allclose(rarr_x, np.zeros(rarr_x.shape)))
        ok_(np.allclose(rarr_y, np.transpose(-m * sin_arr(N, m))))

    for N in (128, 32):
        for m in xrange(N/8):
            yield (cos_gen, N,m)

def test_laplacian():

    def laplacian_gen(N,m,n):
        rarr = sin_cos_arr(N,m,n)
        rarr_laplacian = vcalc.laplacian(rarr)
        X,Y = np.ogrid[0:N, 0:N]
        X -= N/2; Y -= N/2
        calc_laplacian = (- (1.0*m/N)**2 * np.sin(2*m*np.pi*X/N) +
                (1.0*n/N)**2 * np.cos(2*n*np.pi*Y/N))
        ok_(np.allclose(rarr_laplacian, calc_laplacian))

    for N in (128, 32):
        for m in xrange(N/8):
            for n in xrange(N/8):
                yield (laplacian_gen, N,m,n)


def test_dealias():
    nx, ny = 32, 16
    carr = np.zeros((nx,ny),dtype='F')+1.0+1.0j
    carr_check = carr.copy()
    vcalc.dealias(carr)
    np.fft.fftshift(carr_check, axes=[0])
    rad = np.floor(1./3. * nx)
    for i in xrange(nx):
        for j in xrange(ny):
            ii = i - nx/2
            dist = np.sqrt(ii**2 + j**2)
            if dist > rad:
                carr_check[ii,j] = 0.0+0.0j

    np.fft.ifftshift(carr_check, axes=[0])

    assert np.allclose(carr_check, carr)
