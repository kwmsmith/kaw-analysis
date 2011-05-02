#!/usr/bin/env python

import numpy as np
# from fileIO import load_rect_array

import os

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

def get_dist_0(N):
    """
    returns the wraparound distance from the 0th element.

    >>> get_dist_0(4) == [0, 1, 2, 1]
    >>> get_dist_0(8) == [0, 1, 2, 3, 4, 3, 2, 1]
    >>> get_dist_0(7) == [0, 1, 2, 3, 3, 2, 1]
    """
    if not N % 2:
        return range(N/2+1) + range(N/2-1,0,-1)
    else:
        Nm1 = N - 1
        return range(Nm1/2+1) + range(Nm1/2, 0, -1)

def get_spectrum(arr, npoints):
    return integrate_theta(arr, npoints)

def correct_radius(x, spec, npoints):
    rad_correction = (2*np.pi*x)/npoints * x[-1] + spec[0]
    return x, spec/rad_correction

def integrate_theta(arr, npoints):
    nx, ny = arr.shape
    if np.iscomplexobj(arr):
        assert nx != ny
        arr = np.abs(arr)
    fl_arr = arr.flatten()
    if ny == nx/2 + 1:
        # only works with standard complex array arrangements.
        x_dist = get_dist_0(nx)
        y_dist = range(ny)
    elif nx == ny:
        # treat both dimensions the same
        x_dist = get_dist_0(nx)
        y_dist = get_dist_0(ny)
    else:
        raise ValueError("unsupported dimensions in input array, given %r" % (arr.shape,))

    x_dist, y_dist = np.array(x_dist), np.array(y_dist)

    distmax = int(np.ceil(np.sqrt(max(x_dist)**2 + max(y_dist)**2)))
    delta = float(distmax) / (npoints-1)
    x_dist = x_dist[:, np.newaxis]
    dist = np.sqrt(x_dist**2 + y_dist**2).reshape(-1) / delta

    floor_arr = np.array(np.floor(dist), dtype='i')
    ceil_arr =  np.array(np.ceil(floor_arr + .6), dtype='i')

    floor_arr[floor_arr >= npoints] = npoints - 1
    ceil_arr[ceil_arr >= npoints]   = npoints - 1

    x_interp_points = np.linspace(0., distmax, npoints)

    spec = np.zeros((npoints,))

    left_side = (ceil_arr - dist) * fl_arr
    right_side = (dist - floor_arr) * fl_arr

    for i in xrange(nx*ny):
        fl_idx = floor_arr[i]
        cl_idx = ceil_arr[i]
        spec[fl_idx] += left_side[i]
        spec[cl_idx] += right_side[i]

    return x_interp_points, spec

def _get_spectrum(carr, npoints):
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
        spec[fl_idx] += left_side[i]
        spec[ceil_idx] += right_side[i]

    return x_interp_points, spec

def spectra_plot(cpsi, cvor, cden, npoints, name, title):
    import pylab as pl
    Ebk = mult_by_k(cpsi * np.conj(cpsi), 2)
    Evk = mult_by_k(cvor * np.conj(cvor), -2)
    Enk = cden * np.conj(cden)


    x, Bspec = get_spectrum(Ebk, npoints)
    x, Vspec = get_spectrum(Evk, npoints)
    x, Nspec = get_spectrum(Enk, npoints)

    pl.figure()
    pl.loglog(x, Bspec, 'ro-', label='magnetic')
    pl.loglog(x, Vspec, 'g^-', label='kinetic')
    pl.loglog(x, Nspec, 'b<-', label='internal')
    pl.xlabel('wavenumber (norm. units)')
    pl.ylabel('Spect. amp. (norm. units)')
    pl.title(title)
    pl.legend()
    for ext in ('.eps', '.png'):
        pl.savefig(name + ext)

def spectra_h5(h5name, directory):
    import tables
    from visualization import h5_gen
    dta = tables.openFile(h5name, mode='r')

    def itr(gen):
        for x in gen:
            yield x.data()

    cdens = h5_gen(dta, 'cden')
    cpsis = h5_gen(dta, 'cpsi')
    cvors = h5_gen(dta, 'cvor')

    npoints = 256

    from itertools import izip

    try:
        os.makedirs(directory)
    except OSError:
        pass

    for (cpsi, cvor, cden) in izip(cpsis, cvors, cdens):
        arrs = [cpsi.read(), cvor.read(), cden.read()]
        for arr in arrs:
            arr.dtype = np.complex64
        arrs = [np.transpose(arr) for arr in arrs]
        nx = arrs[0].shape[0]
        czeros = np.zeros((nx,1), dtype=np.complex64)
        arrs = [np.hstack([arr, czeros]) for arr in arrs]
        name = os.path.join(directory, cpsi.name)
        assert cpsi.name == cvor.name == cden.name
        spectra_plot(*arrs, npoints=npoints, name=name, title=name)

    dta.close()

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

if __name__ == '__main__':
    import sys
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="fname",
                        help="hdf5 data file")
    opts, args = parser.parse_args()
    if not opts.fname:
        parser.print_help()
        sys.exit(1)

    spectra_h5(opts.fname, 'spectra')
