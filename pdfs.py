import numpy as np
import pylab as pl

def fit_gaussian(arr):
    center = arr.mean()
    std_dev = arr.std()
    return center, std_dev

def plot_pdf_with_fit(arr, label=None):
    import pdb; pdb.set_trace()
    center, std_dev = fit_gaussian(arr)
    arr_z = (arr - center) / std_dev
    nbins = int(np.floor(np.sqrt(arr.size)))
    hist, bin_edges = np.histogram(arr_z, bins=nbins, normed=True)
    bin_edges = bin_edges[:-1]
    gauss_amp = 1./np.sqrt(2*np.pi)
    pl.semilogy(bin_edges, hist, 'ro-', label=label)
    pl.semilogy(bin_edges, gauss_amp * np.exp(-bin_edges**2), 'g-')
