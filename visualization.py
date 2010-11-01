#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use("Agg")
import pylab as pl
import fileIO as io
import sys
from itertools import izip
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

def make_images(basename, arr_dict, rescale=True, img_kind=["png", "eps"]):
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

def main_fs():
    from glob import glob
    from sys import argv
    import os
    for basename in argv[1:]:
        all_fnames = os.listdir('.')
        fnames = io.fnamefilter(basename, all_fnames)
        make_images(basename, fnames)

def main_h5(options, args):
    import sys
    import tables
    dta = tables.openFile(options.fname, mode='r')
    try:
        for field_name in args:
            gp = dta.getNode('/%s' % field_name)
            arr_dict = {}
            for arr in gp:
                try:
                    arr_dta = arr.read()
                except tables.exceptions.HDF5ExtError:
                    sys.stderr.write("error in reading array %s in %s, turning off fletcher32" % (arr.name, gp))
                    arr.filters.fletcher32 = False
                    arr_dta = arr.read()
                arr_dict[arr.name] = arr_dta
            make_images(field_name, arr_dict)
    finally:
        dta.close()

if __name__ == '__main__':
    from optparse import OptionParser
    usage = "visualization.py [options] field_name1 [field_name2 ...]"
    parser = OptionParser(usage=usage)
    parser.add_option("-f", "--file", dest="fname",
                        help="hdf5 data file")
    options, args = parser.parse_args()
    if not options.fname:
        parser.print_help()
        sys.exit(1)

    main_h5(options, args)
