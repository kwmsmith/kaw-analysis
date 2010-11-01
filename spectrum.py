import numpy as np
from fileIO import load_rect_array
from scipy.fftpack import diff

def _is_cplx(arr):
    return arr.dtype.char in ('F', 'D', 'G')

def fft_dec(f):
    def _inner(arr, *args, **kwargs):
        if not _is_cplx(arr):
            carr = np.fft.rfft2(arr)
        else:
            carr = arr
        ret = f(carr, *args, **kwargs)
        if _is_cplx(arr) and _is_cplx(ret):
            return ret
        elif not _is_cplx(arr) and not _is_cplx(ret):
            return ret
        elif not _is_cplx(arr) and _is_cplx(ret):
            return irfft2(ret) / (ret.shape[0] * (ret.shape[1]-1)*2)
        else:
            return rfft2(ret)
    return _inner


def dealias(carr, dls_fac=1./3.):
    nx,ny = carr.shape
    kxidx, kyidx = get_k_idxs(*carr.shape)
    kdist = np.sqrt(kxidx**2 + kyidx**2)
    mask = kdist > np.floor(dls_fac * nx)
    carr[mask] = 0.0+0.0j

def get_k_idxs(nx,ny):
    kyidx = np.arange(0., ny)
    kxidx = np.arange(0.,nx)-nx/2
    kxidx[0] = -kxidx[0]
    kxidx = np.fft.ifftshift(kxidx)
    kxidx = kxidx[:,np.newaxis]
    return kxidx, kyidx

@fft_dec
def mult_by_k(carr, k_power):
    kxidx, kyidx = get_k_idxs(*carr.shape)
    kdist = np.sqrt(kxidx**2 + kyidx**2)
    return carr * kdist**k_power

@fft_dec
def laplacian(carr):
    return -1.0 * mult_by_k(carr, 2)

@fft_dec
def cderivative(carr, direction, order=1):
    nx, ny = carr.shape
    kxidx, kyidx = get_k_idxs(nx,ny)
    if direction == 'Y_DIR':
        karr = kyidx
    elif direction == 'X_DIR':
        karr = kxidx

    karr = (1.0j * karr)**order

    return karr * carr

def partial(rarr, direction, order=1, period=None):
    if direction == 'Y_DIR':
        rarr = rarr.T.copy()
    ret = [diff(row, order=order, period=period) for row in rarr]
    ret = np.vstack(ret)
    if direction == 'Y_DIR':
        ret = ret.T.copy()
    return ret

def hessian(f):
    fx = partial(f, 'X_DIR')
    fyx = partial(fx, 'Y_DIR')
    fxx = partial(f, 'X_DIR', 2)
    fyy = partial(f, 'Y_DIR', 2)
    # fx = cderivative(f, 'X_DIR').copy()
    # fyx = cderivative(fx, 'Y_DIR').copy()
    # fxx = cderivative(f, 'X_DIR', 2).copy()
    # fyy = cderivative(f, 'Y_DIR', 2).copy()
    return (fxx * fyy - fyx**2)

def irfft2(carr):
    nx, ny = carr.shape
    zero_col = np.zeros((nx,1),dtype=carr.dtype)
    if ny*2 == nx:
        temp_carr = np.hstack((carr, zero_col))
    elif (ny-1)*2 == nx:
        temp_carr = carr
    else:
        assert False
    return np.fft.irfft2(temp_carr)
    

def get_spectrum(carr, npoints):
    tmp_carr = np.fft.ifftshift(carr, axes=[0])
    abs_carr = np.abs(tmp_carr).flatten()
    nx, ny = tmp_carr.shape
    kmax = int(np.ceil(np.sqrt(nx**2 + ny**2)))
    xcenter = nx/2
    x_idx, y_idx = np.ogrid[0:nx,0:ny]
    x_idx -= xcenter

    dk = float(kmax) / (npoints-1)
    dist = np.sqrt(x_idx**2 + y_idx**2).reshape(-1) /dk

    floor_arr = np.array(np.floor(dist), dtype='i')
    ceil_arr = np.array(np.ceil(floor_arr + .6), dtype='i')

    floor_arr[floor_arr >= npoints] =  npoints-1
    ceil_arr[ceil_arr >= npoints] =  npoints-1

    x_interp_points = np.linspace(0.,kmax,npoints)

    spec = np.zeros((npoints,))

    left_side = (ceil_arr - dist) * abs_carr
    right_side = (dist - floor_arr) * abs_carr

    for i in xrange(nx*ny):
        fl_idx = floor_arr[i]
        ceil_idx = ceil_arr[i]
        val = abs_carr
        spec[fl_idx] += left_side[i]
        spec[ceil_idx] += right_side[i]

    return x_interp_points, spec

def plot_eng_spectra(cpsi, cvor, cden, npoints):
    import pylab as pl
    pl.ion()
    Ebk = mult_by_k(cpsi * np.conj(cpsi), 2)
    Evk = mult_by_k(cvor * np.conj(cvor), -2)
    Enk = cden * np.conj(cden)

    x,Eb_spect = get_spectrum(Ebk, npoints)
    x,Ev_spect = get_spectrum(Evk, npoints)
    x,En_spect = get_spectrum(Enk, npoints)

    pl.figure()
    pl.loglog(x, Eb_spect, 'ro-', label='magnetic')
    pl.loglog(x, Ev_spect, 'g^-', label='kinetic')
    pl.loglog(x, En_spect, 'b<-', label='internal')
    pl.legend()

def load_carrs(index, ndigits=7, basenames=('cpsi', 'cvor', 'cden'), with_records=False):
    fnames = ['%s_%07d' % (bn, index) for bn in basenames]
    return [load_rect_array(fname, with_records=with_records) for fname in fnames]
