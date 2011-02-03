
def get_spectrum(carr, npoints):
    tmp_carr = np.fft.ifftshift(carr, axes=[0])
    abs_carr = np.abs(tmp_carr).flatten()
    nx, ny = tmp_carr.shape
    kmax = int(np.ceil(np.sqrt(nx**2 + ny**2)))
    xcenter = nx/2
    x_idx, y_idx = np.ogrid[0:nx,0:ny]
    x_idx -= xcenter

    dk = float(kmax) / (npoints-1)
    dist = np.sqrt(x_idx**2 + y_idx**2).reshape(-1) /dk

    floor_arr = np.array(np.floor(dist), dtype='i')
    ceil_arr = np.array(np.ceil(floor_arr + .6), dtype='i')

    floor_arr[floor_arr >= npoints] =  npoints-1
    ceil_arr[ceil_arr >= npoints] =  npoints-1

    x_interp_points = np.linspace(0.,kmax,npoints)

    spec = np.zeros((npoints,))

    left_side = (ceil_arr - dist) * abs_carr
    right_side = (dist - floor_arr) * abs_carr

    for i in xrange(nx*ny):
        fl_idx = floor_arr[i]
        ceil_idx = ceil_arr[i]
        spec[fl_idx] += left_side[i]
        spec[ceil_idx] += right_side[i]

    return x_interp_points, spec
