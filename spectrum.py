#!/usr/bin/env python

import numpy as np
import vcalc
# from fileIO import load_rect_array

import os

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
