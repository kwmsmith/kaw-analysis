#ifndef __RAD_PROF_H__
#define __RAD_PROF_H__

#include <stdlib.h>

int 
rad_prof__(const double *arr,
         const size_t nx,
         const size_t ny,
         const double scale,
         const double center_x,
         const double center_y,
         const double max_rad,
         double *profile,
         const size_t nprofile
         );

#endif
