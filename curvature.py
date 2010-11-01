import numpy as np
from spectrum import cderivative, fft_dec, irfft2

@fft_dec
def gaus_curv(arr):

    arr_x = cderivative(arr, 'X_DIR', 1)
    arr_y = cderivative(arr, 'Y_DIR', 1)

    rarr_x = irfft2(arr_x)
    rarr_y = irfft2(arr_y)
    denom = (rarr_x**2 + rarr_y**2 + 1)**2

    rarr_xx = irfft2(cderivative(arr_x, 'X_DIR', 1))
    rarr_yy = irfft2(cderivative(arr_y, 'Y_DIR', 1))
    rarr_xy = irfft2(cderivative(arr_x, 'Y_DIR', 1))

    return (rarr_xx * rarr_yy - rarr_xy**2) / denom

@fft_dec
def mean_curv(arr):

    arr_x = cderivative(arr, 'X_DIR', 1)
    arr_y = cderivative(arr, 'Y_DIR', 1)

    rarr_x = irfft2(arr_x)
    rarr_y = irfft2(arr_y)
    denom = np.power(rarr_x**2 + rarr_y**2 + 1, 1.5)

    rarr_xx = irfft2(cderivative(arr_x, 'X_DIR', 1))
    rarr_yy = irfft2(cderivative(arr_y, 'Y_DIR', 1))
    rarr_xy = irfft2(cderivative(arr_x, 'Y_DIR', 1))

    term1 = (1 + rarr_x**2) * rarr_yy
    term2 = (1 + rarr_y**2) * rarr_xx
    term3 = -2.0 * rarr_x * rarr_y * rarr_xy

    return 0.5 * (term1 + term2 + term3) / denom
