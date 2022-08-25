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
  double section1;
  double section2;
} ;


int initdamp(struct dataobj *restrict damp_vec, const float h_x, const float h_y, const int x_M, const int x_m, const int x_size, const int y_M, const int y_m, const int y_size, const int abc_x_l_ltkn, const int abc_x_r_rtkn, const int abc_y_l_ltkn, const int abc_y_r_rtkn, const int nthreads, struct profiler * timers)
{
  float *r0_vec;
  posix_memalign((void**)(&r0_vec),64,x_size*sizeof(float));
  float *r1_vec;
  posix_memalign((void**)(&r1_vec),64,x_size*sizeof(float));
  float *r2_vec;
  posix_memalign((void**)(&r2_vec),64,x_size*sizeof(float));
  float *r3_vec;
  posix_memalign((void**)(&r3_vec),64,x_size*sizeof(float));
  float *r4_vec;
  posix_memalign((void**)(&r4_vec),64,y_size*sizeof(float));
  float *r5_vec;
  posix_memalign((void**)(&r5_vec),64,y_size*sizeof(float));
  float *r6_vec;
  posix_memalign((void**)(&r6_vec),64,y_size*sizeof(float));
  float *r7_vec;
  posix_memalign((void**)(&r7_vec),64,y_size*sizeof(float));

  float (*restrict damp)[damp_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]]) damp_vec->data;
  float (*restrict r0) __attribute__ ((aligned (64))) = (float (*)) r0_vec;
  float (*restrict r1) __attribute__ ((aligned (64))) = (float (*)) r1_vec;
  float (*restrict r2) __attribute__ ((aligned (64))) = (float (*)) r2_vec;
  float (*restrict r3) __attribute__ ((aligned (64))) = (float (*)) r3_vec;
  float (*restrict r4) __attribute__ ((aligned (64))) = (float (*)) r4_vec;
  float (*restrict r5) __attribute__ ((aligned (64))) = (float (*)) r5_vec;
  float (*restrict r6) __attribute__ ((aligned (64))) = (float (*)) r6_vec;
  float (*restrict r7) __attribute__ ((aligned (64))) = (float (*)) r7_vec;

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
      #pragma omp simd aligned(damp:64)
      for (int y = y_m; y <= y_M; y += 1)
      {
        damp[x + 1][y + 1] = 0.0F;
      }
    }
  }
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(1) schedule(dynamic,1)
    for (int abc_x_l = x_m; abc_x_l <= abc_x_l_ltkn + x_m - 1; abc_x_l += 1)
    {
      r0[abc_x_l] = fabs(-2.5e-2F*abc_x_l + 2.5e-2F*x_m + 1.025F);
      r1[abc_x_l] = sin(6.28318530717959F*fabs(-2.5e-2F*abc_x_l + 2.5e-2F*x_m + 1.025F));

      #pragma omp simd aligned(damp:64)
      for (int y = y_m; y <= y_M; y += 1)
      {
        damp[abc_x_l + 1][y + 1] += (2.5904082296183e-1F*r0[abc_x_l] - 4.12276274369678e-2F*r1[abc_x_l])/h_x;
      }
    }
  }
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(1) schedule(dynamic,1)
    for (int abc_x_r = -abc_x_r_rtkn + x_M + 1; abc_x_r <= x_M; abc_x_r += 1)
    {
      r2[abc_x_r] = fabs(2.5e-2F*abc_x_r - 2.5e-2F*x_M + 1.025F);
      r3[abc_x_r] = sin(6.28318530717959F*fabs(2.5e-2F*abc_x_r - 2.5e-2F*x_M + 1.025F));

      #pragma omp simd aligned(damp:64)
      for (int y = y_m; y <= y_M; y += 1)
      {
        damp[abc_x_r + 1][y + 1] += (2.5904082296183e-1F*r2[abc_x_r] - 4.12276274369678e-2F*r3[abc_x_r])/h_x;
      }
    }
  }
  STOP_TIMER(section0,timers)
  /* End section0 */

  /* Begin section1 */
  START_TIMER(section1)
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(1) schedule(dynamic,1)
    for (int abc_y_r = -abc_y_r_rtkn + y_M + 1; abc_y_r <= y_M; abc_y_r += 1)
    {
      r6[abc_y_r] = fabs(2.5e-2F*abc_y_r - 2.5e-2F*y_M + 1.025F);
      r7[abc_y_r] = sin(6.28318530717959F*fabs(2.5e-2F*abc_y_r - 2.5e-2F*y_M + 1.025F));
    }
  }
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(1) schedule(dynamic,1)
    for (int abc_y_l = y_m; abc_y_l <= abc_y_l_ltkn + y_m - 1; abc_y_l += 1)
    {
      r4[abc_y_l] = fabs(-2.5e-2F*abc_y_l + 2.5e-2F*y_m + 1.025F);
      r5[abc_y_l] = sin(6.28318530717959F*fabs(-2.5e-2F*abc_y_l + 2.5e-2F*y_m + 1.025F));
    }
  }
  STOP_TIMER(section1,timers)
  /* End section1 */

  /* Begin section2 */
  START_TIMER(section2)
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(1) schedule(dynamic,1)
    for (int x = x_m; x <= x_M; x += 1)
    {
      #pragma omp simd aligned(damp:64)
      for (int abc_y_l = y_m; abc_y_l <= abc_y_l_ltkn + y_m - 1; abc_y_l += 1)
      {
        damp[x + 1][abc_y_l + 1] += (2.5904082296183e-1F*r4[abc_y_l] - 4.12276274369678e-2F*r5[abc_y_l])/h_y;
      }
      #pragma omp simd aligned(damp:64)
      for (int abc_y_r = -abc_y_r_rtkn + y_M + 1; abc_y_r <= y_M; abc_y_r += 1)
      {
        damp[x + 1][abc_y_r + 1] += (2.5904082296183e-1F*r6[abc_y_r] - 4.12276274369678e-2F*r7[abc_y_r])/h_y;
      }
    }
  }
  STOP_TIMER(section2,timers)
  /* End section2 */

  free(r0_vec);
  free(r1_vec);
  free(r2_vec);
  free(r3_vec);
  free(r4_vec);
  free(r5_vec);
  free(r6_vec);
  free(r7_vec);

  return 0;
}
