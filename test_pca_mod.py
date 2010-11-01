from pca_mod import pca
import numpy as np

from nose.tools import set_trace

def _test_svd():
    M = np.array([1,0,0,0,2,
                  0,0,3,0,0,
                  0,0,0,0,0,
                  0,4,0,0,0], dtype='f').reshape(4,5)
    M = M.T
    
    ss_full = np.linalg.svd(M, full_matrices=True)
    ss_nfull = np.linalg.svd(M, full_matrices=False)
                  

def test_pca():
    nx = 50
    data = np.zeros((nx,nx),dtype='f')
    Nslices = nx/10

    data_stack = []
    for _ in xrange(Nslices):
        dc = data.copy()
        dc[_,_] = 1.0
        data_stack.append(dc.flatten())

    data_stack = np.vstack(data_stack).transpose()
    res = pca(data_stack)
