"""
Vector Calculus tools for periodic boundary conditions.

X,Y derivatives, laplacian, etc...
"""

import numpy as np

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
    """
    the X direction is symmetric, the Y direction is asymmetric.
    """
    assert nx > ny
    kyidx = np.arange(0., ny)
    kxidx = np.arange(0.,nx)-nx/2
    kxidx[0] = -kxidx[0]
    kxidx = np.fft.ifftshift(kxidx)
    kxidx = kxidx[:,np.newaxis]
    return kxidx, kyidx

# @fft_dec
def mult_by_k(carr, k_power):
    """
    the X direction is symmetric, the Y direction is asymmetric.
    """
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
