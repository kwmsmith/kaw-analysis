import numpy as np
from math import floor, ceil

def _nongaussian_filter(arr):
    N = arr.size
    arr_resid = arr.flatten() - arr.mean()
    arr_4th = arr_resid**4 / (arr.var()**2 * N)
    sorted_arrargs = np.argsort(arr_4th)
    cumsum, gidx = 0.0, 0
    for idx, arridx in enumerate(sorted_arrargs):
        cumsum += arr_4th[arridx]
        gidx = idx
        if cumsum > 3.0:
            break
    mask_arr = np.ones((N,), dtype='bool')
    mask_arr[sorted_arrargs[:gidx+1]] = False
    return mask_arr

def nongaussian_filter(arr):
    N = arr.size
    if not arr.var():
        return np.zeros((N,), dtype='bool').reshape(arr.shape)

    arr_resid = arr.flatten() - arr.mean()
    arr_4th = arr_resid**4 / (arr.var()**2 * N)
    sorted_arrargs = np.argsort(arr_4th)
    cumsum, gidx = 0.0, 0

    lowerb, upperb = 0, N

    while True:
        assert upperb >= lowerb
        if upperb - lowerb < 2 or cumsum == 3.0:
            gidx = lowerb
            break
        half = int(ceil(0.5 * (lowerb + upperb)))
        intersum = np.sum(arr_4th[sorted_arrargs[lowerb:half]])
        if intersum + cumsum <= 3.0:
            cumsum += intersum
            lowerb = half
        elif intersum + cumsum > 3.0:
            upperb = half

    mask_arr = np.ones((N,), dtype='bool')
    mask_arr[sorted_arrargs[:gidx+1]] = False
    return mask_arr.reshape(arr.shape)

def nongaussian_map(arr):
    """
    nongaussian_map(arr) -> arr

    returns new array with arr's gaussian component zeroed.
    """
    ng_mask = nongaussian_filter(arr)
    arr_cpy = arr.copy()
    arr_cpy[~ng_mask] = 0.0
    return arr_cpy
