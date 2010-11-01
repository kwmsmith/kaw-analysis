import numpy as np

from spectrum import partial

def qnd_opt(f, grad_thresh, abs_thresh):
    fx = partial(f, 'X_DIR')
    fy = partial(f, 'Y_DIR')
    grad_mag = fx**2 + fy**2
    return np.where((grad_mag <= grad_thresh) & (np.abs(f) >= abs_thresh))

def kern_opt(f, kern_size=3):
    assert kern_size in (3, 5, 7)
    offset = (kern_size-1) / 2
    nx, ny = f.shape
    Xs = []; Ys = []
    for i in range(offset, nx-offset):
        for j in range(offset, ny-offset):
            subarr = f[i-offset:i+offset+1, j-offset:j+offset+1]
            if f[i,j] <= np.min(subarr):
                Xs.append(i); Ys.append(j)
    return (np.array(Xs), np.array(Ys))

from scipy import signal

def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size) + y**2/float(sizey)))
    return g / g.sum()

def blur_image(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = signal.convolve(im, g, mode='valid')
    return(improc)

