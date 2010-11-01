from matplotlib import mlab
import numpy as np

def _prepca(P, frac=0):
    """
    Compute the principal components of *P*.  *P* is a (*numVars*,
    *numObs*) array.  *frac* is the minimum fraction of variance that a
    component must contain to be included.

    Return value is a tuple of the form (*Pcomponents*, *Trans*,
    *fracVar*) where:

      - *Pcomponents* : a (numVars, numObs) array

      - *Trans* : the weights matrix, ie, *Pcomponents* = *Trans* *
         *P*

      - *fracVar* : the fraction of the variance accounted for by each
         component returned

    A similar function of the same name was in the Matlab (TM)
    R13 Neural Network Toolbox but is not found in later versions;
    its successor seems to be called "processpcs".
    """
    # U_nf,s_nf,v_nf = np.linalg.svd(P, full_matrices=False)
    U,s,v = np.linalg.svd(P, full_matrices=True)
    varEach = s**2/P.shape[1]
    totVar = varEach.sum()
    fracVar = varEach/totVar
    ind = slice((fracVar>=frac).sum())
    # select the components that are greater
    Trans = U[:,ind].transpose()
    # The transformed data
    Pcomponents = np.dot(Trans,P)
    return Pcomponents, Trans, fracVar[ind]

def pca(arr_stack):
    # return _prepca(arr_stack)
    # return mlab.prepca(arr_stack)
    return shlens_pca(arr_stack)

def shlens_pca(P):
    """
    Compute the principal components of *P*.  *P* is a (*numVars*,
    *numObs*) array.

    Returns (*signals*, *Pcomponents*, *V*)
    *signals* -- (*numVars*, *numObs*) array of projected data
    *Pcomponents* -- each column is a principal component
    *V* -- *numVars* length vector of variances
    """

    m,n = P.shape
    P_mean = np.mean(P, axis=1)
    P_centered = P - P_mean[:,np.newaxis]

    Y = P_centered.transpose() / np.sqrt(n-1)

    u,s,PC = np.linalg.svd(Y, full_matrices=True)

    S = np.diag(s)
    V = S**2

    signals = np.dot(PC.T, P_centered)

    return signals, PC, V
