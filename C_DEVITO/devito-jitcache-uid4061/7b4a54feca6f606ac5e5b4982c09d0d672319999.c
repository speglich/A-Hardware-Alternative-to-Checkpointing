#define _POSIX_C_SOURCE 200809L
#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#include "omp.h"

struct dataobj
{
  void *restrict data;
  int * size;
  int * npsize;
  int * dsize;
  int * hsize;
  int * hofs;
  int * oofs;
} ;

struct profiler
{
  double section0;
} ;


int pad_vp(struct dataobj *restrict vp_vec, const int x_M, const int y_M, const int z_M, const int abc_x_l_ltkn, const int abc_x_r_rtkn, const int abc_y_l_ltkn, const int abc_y_r_rtkn, const int abc_z_l_ltkn, const int abc_z_r_rtkn, const int x_m, const int y_m, const int z_m, const int nthreads, struct profiler * timers)
{
  float (*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]][vp_vec->size[2]]) vp_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  /* Begin section0 */
  START_TIMER(section0)
  for (int abc_x_l = x_m; abc_x_l <= abc_x_l_ltkn + x_m - 1; abc_x_l += 1)
  {
    #pragma omp parallel num_threads(nthreads)
    {
      #pragma omp for collapse(1) schedule(static,1)
      for (int y = y_m; y <= y_M; y += 1)
      {
        #pragma omp simd aligned(vp:64)
        for (int z = z_m; z <= z_M; z += 1)
        {
          vp[abc_x_l + 6][y + 6][z + 6] = vp[46][y + 6][z + 6];
        }
      }
    }
  }
  for (int abc_x_r = -abc_x_r_rtkn + x_M + 1; abc_x_r <= x_M; abc_x_r += 1)
  {
    #pragma omp parallel num_threads(nthreads)
    {
      #pragma omp for collapse(1) schedule(static,1)
      for (int y = y_m; y <= y_M; y += 1)
      {
        #pragma omp simd aligned(vp:64)
        for (int z = z_m; z <= z_M; z += 1)
        {
          vp[abc_x_r + 6][y + 6][z + 6] = vp[x_M - 34][y + 6][z + 6];
        }
      }
    }
  }
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(1) schedule(static,1)
    for (int x = x_m; x <= x_M; x += 1)
    {
      for (int abc_y_l = y_m; abc_y_l <= abc_y_l_ltkn + y_m - 1; abc_y_l += 1)
      {
        #pragma omp simd aligned(vp:64)
        for (int z = z_m; z <= z_M; z += 1)
        {
          vp[x + 6][abc_y_l + 6][z + 6] = vp[x + 6][46][z + 6];
        }
      }
      for (int abc_y_r = -abc_y_r_rtkn + y_M + 1; abc_y_r <= y_M; abc_y_r += 1)
      {
        #pragma omp simd aligned(vp:64)
        for (int z = z_m; z <= z_M; z += 1)
        {
          vp[x + 6][abc_y_r + 6][z + 6] = vp[x + 6][y_M - 34][z + 6];
        }
      }
      for (int y = y_m; y <= y_M; y += 1)
      {
        for (int abc_z_l = z_m; abc_z_l <= abc_z_l_ltkn + z_m - 1; abc_z_l += 1)
        {
          vp[x + 6][y + 6][abc_z_l + 6] = vp[x + 6][y + 6][46];
        }
        for (int abc_z_r = -abc_z_r_rtkn + z_M + 1; abc_z_r <= z_M; abc_z_r += 1)
        {
          vp[x + 6][y + 6][abc_z_r + 6] = vp[x + 6][y + 6][z_M - 34];
        }
      }
    }
  }
  STOP_TIMER(section0,timers)
  /* End section0 */

  return 0;
}
