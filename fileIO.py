from __future__ import with_statement
import re
import struct
from math import sqrt, ceil
import numpy as np

LITTLE_END = '<'
BIG_END = '>'

def get_endianness():
    from sys import byteorder
    if byteorder == 'little':
        return LITTLE_END
    elif byteorder == 'big':
        return BIG_END

NATIVE_ENDIANNESS = get_endianness()

def load_rect_array(fname, dtype=np.dtype('F'), endianness=NATIVE_ENDIANNESS,
        with_records=False):
    flat_arr = load_fort_bin_array(fname, dtype, endianness, with_records)
    xsize = int(ceil(sqrt(flat_arr.size/2)))
    assert 2 * xsize * xsize == flat_arr.size
    cmplx_arr = np.asfortranarray(flat_arr.reshape(xsize, 2*xsize).T)
    return cmplx_arr

def load_square_array(fname, dtype=np.dtype('f'), endianness=NATIVE_ENDIANNESS,
        with_records=False):
    flat_arr = load_fort_bin_array(fname, dtype, endianness, with_records)
    xsize = int(ceil(sqrt(flat_arr.size)))
    if xsize * xsize != flat_arr.size:
        import pdb; pdb.set_trace()
    farray = np.asfortranarray(flat_arr.reshape(xsize, xsize))
    return farray

def records(fh, dtype, endianness=NATIVE_ENDIANNESS):
    fmt = endianness+'I'
    while True:
        nbytes = fh.read(4)
        if not nbytes: return
        nbytes = struct.unpack(fmt,nbytes)[0]
        section = np.fromfile(fh, count = int(float(nbytes)/dtype.itemsize),
                dtype=dtype)
        after_bytes = struct.unpack(fmt,fh.read(4))[0]
        assert nbytes == after_bytes
        yield section

def load_fort_bin_array(fname, dtype=np.dtype('f'),
        endianness=NATIVE_ENDIANNESS, with_records=False):
    sections = []
    with file(fname) as fh:
        if with_records:
            sections = list(records(fh, dtype, endianness))
            arr = np.concatenate(sections)
        else:
            arr = np.fromfile(fh, count=-1, dtype=dtype)
    if NATIVE_ENDIANNESS != endianness: arr.byteswap(True)
    return arr

def write_fort_bin_array(fname, arr, dtype=np.dtype('f'),
        endianness=NATIVE_ENDIANNESS, with_records=False):
    fmt = endianness+'I'
    with file(fname, 'wb') as fh:
        if with_records:
            nbytes = arr.nbytes
            fh.write(struct.pack(fmt,nbytes))

        arr = arr if NATIVE_ENDIANNESS == endianness else arr.byteswap()
        if arr.dtype != dtype:
            arr = arr.astype(dtype)
        arr.tofile(fh)

        if with_records:
            fh.write(struct.pack(fmt,nbytes))

def fnamematch(base, numdigits=7):
    restr = r'^%s_\d{%d}$' % (base, numdigits)
    return re.compile(restr).match

def fnamefilter(base, fnames, numdigits=7):
    matcher = fnamematch(base, numdigits)
    return [fname for fname in fnames if matcher(fname)]

def load_all(base, fnames):
    filt_names = fnamefilter(base, fnames)
    arrs = np.vstack(load_fort_bin_array(name) for name in filt_names)
    return arrs

def file_iter(fnames, loader=load_square_array, kwargs={}):
    for fname in fnames:
        yield loader(fname, **kwargs)
