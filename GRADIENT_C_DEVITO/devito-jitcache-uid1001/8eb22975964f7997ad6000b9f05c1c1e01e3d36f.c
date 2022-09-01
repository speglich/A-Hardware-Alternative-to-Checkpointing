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
/* Backdoor edit at Tue Aug 30 14:28:53 2022*/ 
/* Backdoor edit at Tue Aug 30 14:29:59 2022*/ 
/* Backdoor edit at Tue Aug 30 14:33:43 2022*/ 
/* Backdoor edit at Tue Aug 30 14:36:02 2022*/ 
/* Backdoor edit at Tue Aug 30 14:51:27 2022*/ 
/* Backdoor edit at Tue Aug 30 15:13:53 2022*/ 
/* Backdoor edit at Tue Aug 30 15:15:48 2022*/ 
/* Backdoor edit at Tue Aug 30 16:22:42 2022*/ 
/* Backdoor edit at Tue Aug 30 19:29:16 2022*/ 
/* Backdoor edit at Tue Aug 30 19:32:19 2022*/ 
/* Backdoor edit at Tue Aug 30 23:12:49 2022*/ 
/* Backdoor edit at Tue Aug 30 23:13:16 2022*/ 
/* Backdoor edit at Tue Aug 30 23:13:31 2022*/ 
/* Backdoor edit at Tue Aug 30 23:13:53 2022*/ 
/* Backdoor edit at Tue Aug 30 23:14:25 2022*/ 
/* Backdoor edit at Tue Aug 30 23:14:46 2022*/ 
/* Backdoor edit at Tue Aug 30 23:15:29 2022*/ 
/* Backdoor edit at Tue Aug 30 23:16:13 2022*/ 
/* Backdoor edit at Tue Aug 30 23:16:41 2022*/ 
/* Backdoor edit at Tue Aug 30 23:17:03 2022*/ 
/* Backdoor edit at Tue Aug 30 23:17:15 2022*/ 
/* Backdoor edit at Tue Aug 30 23:25:33 2022*/ 
/* Backdoor edit at Tue Aug 30 23:26:02 2022*/ 
/* Backdoor edit at Tue Aug 30 23:33:19 2022*/ 
/* Backdoor edit at Tue Aug 30 23:35:30 2022*/ 
/* Backdoor edit at Tue Aug 30 23:49:44 2022*/ 
/* Backdoor edit at Tue Aug 30 23:53:00 2022*/ 
/* Backdoor edit at Tue Aug 30 23:58:50 2022*/ 
/* Backdoor edit at Wed Aug 31 00:01:06 2022*/ 
/* Backdoor edit at Wed Aug 31 00:01:32 2022*/ 
/* Backdoor edit at Wed Aug 31 00:02:15 2022*/ 
/* Backdoor edit at Wed Aug 31 00:04:55 2022*/ 
/* Backdoor edit at Wed Aug 31 00:05:19 2022*/ 
/* Backdoor edit at Wed Aug 31 00:09:32 2022*/ 
/* Backdoor edit at Wed Aug 31 00:14:17 2022*/ 
/* Backdoor edit at Wed Aug 31 00:15:07 2022*/ 
/* Backdoor edit at Wed Aug 31 00:16:16 2022*/ 
/* Backdoor edit at Wed Aug 31 00:49:01 2022*/ 
/* Backdoor edit at Wed Aug 31 00:49:49 2022*/ 
/* Backdoor edit at Wed Aug 31 00:54:52 2022*/ 
/* Backdoor edit at Wed Aug 31 01:09:49 2022*/ 
/* Backdoor edit at Wed Aug 31 01:15:03 2022*/ 
/* Backdoor edit at Wed Aug 31 01:16:10 2022*/ 
/* Backdoor edit at Wed Aug 31 01:16:54 2022*/ 
/* Backdoor edit at Wed Aug 31 01:19:57 2022*/ 
/* Backdoor edit at Wed Aug 31 01:58:52 2022*/ 
/* Backdoor edit at Wed Aug 31 02:04:35 2022*/ 
/* Backdoor edit at Wed Aug 31 02:15:54 2022*/ 
/* Backdoor edit at Wed Aug 31 02:20:45 2022*/ 
/* Backdoor edit at Wed Aug 31 02:21:08 2022*/ 
/* Backdoor edit at Wed Aug 31 13:25:55 2022*/ 
/* Backdoor edit at Wed Aug 31 15:03:00 2022*/ 
/* Backdoor edit at Wed Aug 31 16:27:23 2022*/ 
/* Backdoor edit at Wed Aug 31 16:28:04 2022*/ 
/* Backdoor edit at Wed Aug 31 16:28:34 2022*/ 
/* Backdoor edit at Wed Aug 31 16:35:36 2022*/ 
/* Backdoor edit at Wed Aug 31 16:42:39 2022*/ 
/* Backdoor edit at Wed Aug 31 16:42:59 2022*/ 
/* Backdoor edit at Wed Aug 31 16:51:59 2022*/ 
/* Backdoor edit at Wed Aug 31 16:58:09 2022*/ 
/* Backdoor edit at Wed Aug 31 17:13:42 2022*/ 
/* Backdoor edit at Wed Aug 31 17:14:23 2022*/ 
/* Backdoor edit at Wed Aug 31 17:18:16 2022*/ 
/* Backdoor edit at Wed Aug 31 17:18:40 2022*/ 
/* Backdoor edit at Wed Aug 31 18:17:27 2022*/ 
/* Backdoor edit at Wed Aug 31 18:42:47 2022*/ 
/* Backdoor edit at Wed Aug 31 18:43:27 2022*/ 
/* Backdoor edit at Wed Aug 31 18:56:35 2022*/ 
/* Backdoor edit at Wed Aug 31 19:20:41 2022*/ 
/* Backdoor edit at Wed Aug 31 19:37:48 2022*/ 
/* Backdoor edit at Wed Aug 31 19:38:14 2022*/ 
/* Backdoor edit at Wed Aug 31 19:54:47 2022*/ 
/* Backdoor edit at Wed Aug 31 20:32:42 2022*/ 
/* Backdoor edit at Wed Aug 31 20:33:02 2022*/ 
/* Backdoor edit at Wed Aug 31 20:37:11 2022*/ 
/* Backdoor edit at Wed Aug 31 21:13:00 2022*/ 
/* Backdoor edit at Wed Aug 31 21:27:45 2022*/ 
/* Backdoor edit at Wed Aug 31 21:29:16 2022*/ 
/* Backdoor edit at Wed Aug 31 21:40:20 2022*/ 
/* Backdoor edit at Wed Aug 31 21:41:41 2022*/ 
/* Backdoor edit at Wed Aug 31 21:41:58 2022*/ 
/* Backdoor edit at Wed Aug 31 21:43:09 2022*/ 
/* Backdoor edit at Wed Aug 31 21:47:58 2022*/ 
/* Backdoor edit at Wed Aug 31 21:49:01 2022*/ 
/* Backdoor edit at Thu Sep  1 11:34:26 2022*/ 
/* Backdoor edit at Thu Sep  1 11:35:20 2022*/ 
/* Backdoor edit at Thu Sep  1 11:44:10 2022*/ 
/* Backdoor edit at Thu Sep  1 11:45:07 2022*/ 
/* Backdoor edit at Thu Sep  1 11:46:08 2022*/ 
/* Backdoor edit at Thu Sep  1 11:48:01 2022*/ 
/* Backdoor edit at Thu Sep  1 12:38:01 2022*/ 
/* Backdoor edit at Thu Sep  1 12:54:56 2022*/ 
/* Backdoor edit at Thu Sep  1 12:55:11 2022*/ 
/* Backdoor edit at Thu Sep  1 13:15:30 2022*/ 
/* Backdoor edit at Thu Sep  1 13:15:52 2022*/ 
/* Backdoor edit at Thu Sep  1 13:17:17 2022*/ 
/* Backdoor edit at Thu Sep  1 13:49:41 2022*/ 
/* Backdoor edit at Thu Sep  1 13:50:38 2022*/ 
/* Backdoor edit at Thu Sep  1 13:51:18 2022*/ 
/* Backdoor edit at Thu Sep  1 13:51:57 2022*/ 
/* Backdoor edit at Thu Sep  1 13:54:50 2022*/ 
/* Backdoor edit at Thu Sep  1 13:55:48 2022*/ 
/* Backdoor edit at Thu Sep  1 13:56:50 2022*/ 
/* Backdoor edit at Thu Sep  1 13:57:28 2022*/ 
/* Backdoor edit at Thu Sep  1 13:57:56 2022*/ 
/* Backdoor edit at Thu Sep  1 13:58:24 2022*/ 
/* Backdoor edit at Thu Sep  1 13:59:19 2022*/ 
/* Backdoor edit at Thu Sep  1 13:59:48 2022*/ 
/* Backdoor edit at Thu Sep  1 14:00:11 2022*/ 
/* Backdoor edit at Thu Sep  1 14:02:13 2022*/ 
/* Backdoor edit at Thu Sep  1 14:02:30 2022*/ 
/* Backdoor edit at Thu Sep  1 14:13:18 2022*/ 
/* Backdoor edit at Thu Sep  1 14:14:52 2022*/ 
/* Backdoor edit at Thu Sep  1 14:17:08 2022*/ 
