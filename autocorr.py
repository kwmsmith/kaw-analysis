#!/usr/bin/env python

import os

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

def savefig_ac(f, title, fname):
    import pylab as pl
    fig = pl.figure()
    ac = autocorr(f)
    # normalize by 0,0 component
    ac /= ac[0,0]
    pl.imshow(ac)
    pl.title(title)
    pl.colorbar()
    for ext in ('.eps', '.png'):
        pl.savefig(fname + ext)
    del fig
    pl.close('all')

def autocorr_h5(h5name, directory, fields):
    # TODO: merge this with spectrum.spectra_h5
    import tables
    from visualization import h5_gen
    from itertools import izip

    dta = tables.openFile(h5name, mode='r')

    try:
        os.makedirs(directory)
    except OSError:
        pass

    for field in fields:
        for h5arr in h5_gen(dta, field):
            arr = h5arr.read()
            title = "%s%s" % (field, h5arr.name)
            name = os.path.join(directory, title)
            savefig_ac(arr, title, fname=name)
            del arr

    dta.close()


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

    autocorr_h5(opts.fname, 'autocorr', args)
