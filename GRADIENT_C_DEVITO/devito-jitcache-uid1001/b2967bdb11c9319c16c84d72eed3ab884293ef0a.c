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


int smoother(struct dataobj *restrict vel0_vec, struct dataobj *restrict vp_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int nthreads, struct profiler * timers)
{
  float (*restrict vel0)[vel0_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[vel0_vec->size[1]]) vel0_vec->data;
  float (*restrict vp)[vp_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]]) vp_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  /* Begin section0 */
  START_TIMER(section0)
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(1) schedule(static,1)
    for (int x = x_m; x <= x_M; x += 1)
    {
      #pragma omp simd aligned(vel0,vp:64)
      for (int y = y_m; y <= y_M; y += 1)
      {
        vel0[x + 4][y + 4] = (1.0F/8.0F)*(vp[x + 4][y] + vp[x + 4][y + 1] + vp[x + 4][y + 2] + vp[x + 4][y + 3] + vp[x + 4][y + 4] + vp[x + 4][y + 5] + vp[x + 4][y + 6] + vp[x + 4][y + 7]);
      }
    }
  }
  STOP_TIMER(section0,timers)
  /* End section0 */

  return 0;
}
