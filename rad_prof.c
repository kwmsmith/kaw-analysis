#include "rad_prof.h"
#include <math.h>
#include <assert.h>

#define BOUNDS_CHECK

#ifdef BOUNDS_CHECK
#include <stdio.h>
#endif

#define IDX(i,j,ny) ((i)*(ny)+(j))

int rad_prof__(const double *arr, /* in */
              const size_t nx, /* in */
              const size_t ny, /* in */
              const double scale, /* in */
              const double center_x, /* in */
              const double center_y, /* in */
              const double max_rad, /* in */
              double *profile, /* out */
              const size_t nprofile /* in */
              )
{
    size_t i, j;
    double delta_x, delta_y, i_scl, j_scl;
    double *dist_arr = NULL;

    /* make distance array */
    dist_arr = (double *)malloc(nx*ny*sizeof(double));
    if(NULL == dist_arr) 
        return 1;

    /* initialize distance array */
    for(i=0; i<nx; i++) {
        for(j=0; j<ny; j++) {
            i_scl = (i * scale) / nx;
            j_scl = (j * scale) / ny;
            delta_x = i_scl - center_x;
            delta_y = j_scl - center_y;
            if(abs(delta_x) >= scale/2.0) {
                delta_x -= scale;
            }
            if(abs(delta_y) >= scale/2.0) {
                delta_y -= scale;
            }
            assert(delta_x <= scale/2.0);
            assert(delta_y <= scale/2.0);
            /* rescale the distance so that the distance scale tells us
             * which index in the profile array this point corresponds to.
             * so 0 -> 0 and
             * max_rad -> nprofile - 1.
             */
            dist_arr[IDX(i,j,ny)] = 
                sqrt(delta_x * delta_x + delta_y * delta_y) /
                max_rad * (nprofile-1);

        }
    }

    /* Iterate through distance array and linearly interpolate the arr's value
     * between the floor and ceil of distance arr, and accumulate these values
     * in the profile array.
     */
    {
        unsigned long *normsum = NULL;
        double dist, val, lower_idx, upper_idx, dx;
        double norm_max_dist = nprofile - 1;

        normsum = (unsigned long *)malloc(nprofile * sizeof(unsigned long));
        if(NULL == normsum) {
            printf("oh crap!\n");
            return 2;
        } else {
            for(i=0; i<nprofile; i++) {
                normsum[i] = 0L;
            }
        }

        for(i=0; i<nx; i++) {
            for(j=0; j<ny; j++) {
                dist = dist_arr[IDX(i,j,ny)];
                if(dist > norm_max_dist)
                    continue;
                lower_idx = floor(dist);
                upper_idx = ceil(dist);
                dx = dist - lower_idx;
                val = arr[IDX(i,j,ny)];
#ifdef BOUNDS_CHECK
                if((int)lower_idx >= nprofile || (int)upper_idx >= nprofile) {
                    printf("boundscheck error! (%zd,%zd) >= %zd\n",
                            (size_t)lower_idx, (size_t)upper_idx, nprofile);
                }
#endif
                profile[(int)lower_idx] += (1.-dx) * val;
                profile[(int)upper_idx] += dx * val;
                normsum[(int)lower_idx]++;
                normsum[(int)upper_idx]++;
            }
        }
        for(i=0; i<nprofile; i++) {
            if(normsum[i])
                profile[i] /= normsum[i];
        }
        if(normsum)
            free(normsum);
    }

    if(dist_arr)
        free(dist_arr);
    return 0;
}
