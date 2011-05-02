#!/usr/bin/env python

from concurrent import futures
import numpy as np
import matplotlib
matplotlib.use("Agg")
import pylab as pl
from itertools import izip
# import fileIO as io
import sys
import os

IMAGES_DIR = "images"

def imsave(fname, arr, **kwargs):
    figsize=(np.array(arr.shape)/100.0)[::-1]
    fig = pl.figure(figsize=figsize)
    pl.axes([0,0,1,1])
    pl.axis('off')
    fig.set_size_inches(figsize)
    pl.imshow(arr, origin='lower',cmap=pl.cm.hot, **kwargs)
    pl.savefig(fname, facecolor='white',edgecolor='white',dpi=80)
    pl.close(fig)
    del fig

def minmax(arrs):
    arr0 = arrs[0]
    gmn, gmx = np.min(arr0), np.max(arr0)
    for arr in arrs:
        mn, mx = np.min(arr), np.max(arr)
        if mn < gmn: gmn = mn
        if mx > gmx: gmx = mx
    return (gmn, gmx)

def make_images(basename, arr_dict, rescale=True, img_kind=['pdf']):
    # to let pylab release memory...
    pl.close('all')
    # will raise exception if <basename>_images already exists
    dirname = os.path.join(IMAGES_DIR, basename+'_images')
    try:
        os.makedirs(dirname)
    except OSError, (errno, msg):
        if errno != 17: # 17 is the 'file exists' error, we eat this
            raise
    arrs = arr_dict.values()
    if rescale:
        mn, mx = minmax(arrs)
        kwargs = dict(vmin=mn, vmax=mx)
    else:
        kwargs = {}
    for arr_name in arr_dict:
        arr = arr_dict[arr_name]
        for knd in img_kind:
            save_name = os.path.join(dirname,arr_name+"."+knd)
            imsave(save_name, arr, **kwargs)

def h5_gen(h5file, gpname):
    import tables
    if isinstance(h5file, tables.file.File):
        dta = h5file
    elif isinstance(h5file, basestring):
        dta = tables.openFile(h5file, mode='r')
    gp = dta.getNode('/%s' % gpname)
    for arr in gp:
        yield arr

def run_once(fname, field_name):
    import sys
    import tables
    dta = tables.openFile(fname, mode='r')
    try:
        if field_name.endswith('mag'):
            field_x = field_name[:-len('mag')] + 'x'
            field_y = field_name[:-len('mag')] + 'y'
            arrs_x = dta.walkNodes('/%s' % field_x, classname='Array')
            arrs_y = dta.walkNodes('/%s' % field_y, classname='Array')
            arr_dict = {}
            for arr_x, arr_y in izip(arrs_x, arrs_y):
                assert arr_x.name == arr_y.name
                nm = arr_x.name
                arr_x = arr_x.read()
                arr_y = arr_y.read()
                arr_mag = np.sqrt(arr_x**2 + arr_y**2)
                arr_dict[nm] = arr_mag
            make_images(field_name, arr_dict, rescale=options.rescale)
        else:
            arrs = dta.walkNodes('/%s' % field_name, classname='Array')
            arr_dict = {}
            for arr in arrs:
                try:
                    arr_dta = arr.read()
                except tables.exceptions.HDF5ExtError:
                    sys.stderr.write("error in reading array %s, turning off fletcher32" % arr.name)
                    arr.filters.fletcher32 = False
                    arr_dta = arr.read()
                arr_dict[arr.name] = arr_dta
            make_images(field_name, arr_dict, rescale=options.rescale)
    finally:
        dta.close()

def main_h5(options, args):
    field_names = args
    fnames = [options.fname] * len(field_names)
    with futures.ProcessPoolExecutor() as executor:
        for _ in executor.map(run_once, fnames, field_names):
            pass
    # import sys
    # import tables
    # dta = tables.openFile(options.fname, mode='r')
    # try:
        # for field_name in args:
            # gp = dta.getNode('/%s' % field_name)
            # arr_dict = {}
            # for arr in gp:
                # try:
                    # arr_dta = arr.read()
                # except tables.exceptions.HDF5ExtError:
                    # sys.stderr.write("error in reading array %s in %s, turning off fletcher32" % (arr.name, gp))
                    # arr.filters.fletcher32 = False
                    # arr_dta = arr.read()
                # arr_dict[arr.name] = arr_dta
            # make_images(field_name, arr_dict, rescale=options.rescale)
    # finally:
        # dta.close()

def make_quiver_overlay(background, arr_x, arr_y, title='', skip=8):
    nx, ny = arr_x.shape
    if nx != ny:
        raise ValueError("nx != ny")
    slicer = slice(0, nx, skip)
    X,Y = np.ogrid[slicer, slicer]
    # pl.imshow(background, origin='lower', cmap=pl.cm.hot)
    pl.imshow(background)
    pl.colorbar()
    pl.quiver(X, Y, arr_x[slicer, slicer], arr_y[slicer, slicer])
    pl.title(title)

if __name__ == '__main__':
    from optparse import OptionParser
    usage = "visualization.py [options] field_name1 [field_name2 ...]"
    parser = OptionParser(usage=usage)
    parser.add_option("-f", "--file", dest="fname",
                        help="hdf5 data file")
    parser.add_option('-r', '--norescale', dest='rescale',
                        action='store_false', default=True,
                        help='turn off rescaling')
    options, args = parser.parse_args()
    if not options.fname:
        parser.print_help()
        sys.exit(1)

    main_h5(options, args)
