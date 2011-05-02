#!/usr/bin/env python

import os

import numpy as np
import scipy as sp

from numpy.fft import rfft2, irfft2

from spectrum import integrate_theta, correct_radius

def correlation(f, g):
    f_fft = rfft2(f)
    if f is g:
        g_fft = f_fft
    else:
        g_fft = rfft2(g)
    g_conj = np.conj(g_fft)
    prod = f_fft * g_conj
    return np.real(irfft2(prod))

def autocorr(f):
    return correlation(f,f)

def get_ac_radial(f):
    ac = autocorr(f)
    # normalize by 0,0 component
    ac /= ac[0,0]
    npts = f.shape[0]/2
    x, ac_radial = integrate_theta(ac, npts)
    x, ac_radial = correct_radius(x, ac_radial, npts)
    return x, ac_radial

def plot_multi(h5arrs, fields, title, fname):
    import pylab as pl
    fig = pl.figure()

    styles = ('go-', 'b>-', 'k^-', 'wD-')

    for idx, h5arr in enumerate(h5arrs):
        style = styles[idx%len(styles)]
        arr = h5arr.read()
        x, acrad = get_ac_radial(arr)
        pl.plot(x, acrad, style, label=fields[idx])

    pl.legend()
    pl.title(title)
    pl.xlabel('distance (norm. units)')
    pl.ylabel('field amplitude (norm. units)')
    for ext in ('.eps', '.png'):
        pl.savefig(fname + ext)
    del fig
    pl.close('all')

def savefig_ac(f, title, fname):
    import pylab as pl
    fig = pl.figure()
    x, ac_radial = get_ac_radial(f)
    pl.plot(x, ac_radial)
    pl.title(title)
    pl.xlabel('distance (norm. units)')
    pl.ylabel('field amplitude (norm. units)')
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

    field_gens = [h5_gen(dta, field) for field in fields]

    for h5arrs in izip(*field_gens):
        name = h5arrs[0].name
        title = '%s autocorrelation' % name
        fname = os.path.join(directory, name)
        plot_multi(h5arrs, fields, title, fname)

    # for field in fields:
        # for h5arr in h5_gen(dta, field):
            # arr = h5arr.read()
            # arr -= arr.mean()
            # name = "%s%s" % (field, h5arr.name)
            # title = "%s autocorrelation" % (name,)
            # name = os.path.join(directory, name)
            # savefig_ac(arr, title, fname=name)
            # del arr

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
