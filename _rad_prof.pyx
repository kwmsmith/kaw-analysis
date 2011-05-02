cdef extern from "rad_prof.h":
    int rad_prof__(double *arr,
                   size_t nx,
                   size_t ny,
                   double scale,
                   double center_x,
                   double center_y,
                   double max_rad,
                   double *profile,
                   size_t nprofile)

import numpy as np
cimport numpy as np

def rad_prof(np.ndarray[double, ndim=2] arr,
              double scale,
              double center_x,
              double center_y,
              double max_rad,
              np.ndarray[double, ndim=1] profile):

    cdef int rval

    profile.fill(0.0)

    rval = rad_prof__(<double*>arr.data,
                         <size_t>arr.shape[0],
                         <size_t>arr.shape[1],
                         scale,
                         center_x,
                         center_y,
                         max_rad,
                         <double*>profile.data,
                         profile.shape[0])
    if rval:
        raise RuntimeError("_rad_prof returned error code %d" % rval)
