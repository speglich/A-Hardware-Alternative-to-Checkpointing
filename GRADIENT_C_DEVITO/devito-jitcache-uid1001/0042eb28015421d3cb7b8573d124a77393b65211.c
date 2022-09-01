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


int pad_vp(struct dataobj *restrict vp_vec, const int x_M, const int y_M, const int abc_x_l_ltkn, const int abc_x_r_rtkn, const int abc_y_l_ltkn, const int abc_y_r_rtkn, const int x_m, const int y_m, const int nthreads, struct profiler * timers)
{
  float (*restrict vp)[vp_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]]) vp_vec->data;

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
        vp[abc_x_l + 4][y + 4] = vp[44][y + 4];
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
        vp[abc_x_r + 4][y + 4] = vp[x_M - 36][y + 4];
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
        vp[x + 4][abc_y_l + 4] = vp[x + 4][44];
      }
      for (int abc_y_r = -abc_y_r_rtkn + y_M + 1; abc_y_r <= y_M; abc_y_r += 1)
      {
        vp[x + 4][abc_y_r + 4] = vp[x + 4][y_M - 36];
      }
    }
  }
  STOP_TIMER(section0,timers)
  /* End section0 */

  return 0;
}
/* Backdoor edit at Tue Aug 30 14:28:54 2022*/ 
/* Backdoor edit at Tue Aug 30 14:29:59 2022*/ 
/* Backdoor edit at Tue Aug 30 14:33:44 2022*/ 
/* Backdoor edit at Tue Aug 30 14:36:03 2022*/ 
/* Backdoor edit at Tue Aug 30 14:51:28 2022*/ 
/* Backdoor edit at Tue Aug 30 15:13:54 2022*/ 
/* Backdoor edit at Tue Aug 30 15:15:48 2022*/ 
/* Backdoor edit at Tue Aug 30 16:22:42 2022*/ 
/* Backdoor edit at Tue Aug 30 19:29:16 2022*/ 
/* Backdoor edit at Tue Aug 30 19:32:19 2022*/ 
/* Backdoor edit at Tue Aug 30 23:13:32 2022*/ 
/* Backdoor edit at Tue Aug 30 23:13:54 2022*/ 
/* Backdoor edit at Tue Aug 30 23:15:30 2022*/ 
/* Backdoor edit at Tue Aug 30 23:16:14 2022*/ 
/* Backdoor edit at Tue Aug 30 23:16:42 2022*/ 
/* Backdoor edit at Tue Aug 30 23:17:16 2022*/ 
/* Backdoor edit at Wed Aug 31 00:02:16 2022*/ 
/* Backdoor edit at Wed Aug 31 00:04:55 2022*/ 
/* Backdoor edit at Wed Aug 31 00:05:19 2022*/ 
/* Backdoor edit at Wed Aug 31 00:14:18 2022*/ 
/* Backdoor edit at Wed Aug 31 00:15:08 2022*/ 
/* Backdoor edit at Wed Aug 31 00:16:17 2022*/ 
/* Backdoor edit at Wed Aug 31 00:49:02 2022*/ 
/* Backdoor edit at Wed Aug 31 00:49:50 2022*/ 
/* Backdoor edit at Wed Aug 31 00:54:52 2022*/ 
/* Backdoor edit at Wed Aug 31 01:09:49 2022*/ 
/* Backdoor edit at Wed Aug 31 01:15:04 2022*/ 
/* Backdoor edit at Wed Aug 31 01:16:11 2022*/ 
/* Backdoor edit at Wed Aug 31 01:16:55 2022*/ 
/* Backdoor edit at Wed Aug 31 01:19:58 2022*/ 
/* Backdoor edit at Wed Aug 31 01:58:53 2022*/ 
/* Backdoor edit at Wed Aug 31 02:04:35 2022*/ 
/* Backdoor edit at Wed Aug 31 02:15:55 2022*/ 
/* Backdoor edit at Wed Aug 31 02:20:46 2022*/ 
/* Backdoor edit at Wed Aug 31 02:21:09 2022*/ 
/* Backdoor edit at Wed Aug 31 13:25:56 2022*/ 
/* Backdoor edit at Wed Aug 31 15:03:01 2022*/ 
/* Backdoor edit at Wed Aug 31 16:27:24 2022*/ 
/* Backdoor edit at Wed Aug 31 16:28:05 2022*/ 
/* Backdoor edit at Wed Aug 31 16:28:34 2022*/ 
/* Backdoor edit at Wed Aug 31 16:35:37 2022*/ 
/* Backdoor edit at Wed Aug 31 16:42:40 2022*/ 
/* Backdoor edit at Wed Aug 31 16:42:59 2022*/ 
/* Backdoor edit at Wed Aug 31 16:52:00 2022*/ 
/* Backdoor edit at Wed Aug 31 16:58:09 2022*/ 
/* Backdoor edit at Wed Aug 31 17:13:42 2022*/ 
/* Backdoor edit at Wed Aug 31 17:14:24 2022*/ 
/* Backdoor edit at Wed Aug 31 17:18:17 2022*/ 
/* Backdoor edit at Wed Aug 31 17:18:41 2022*/ 
/* Backdoor edit at Wed Aug 31 18:17:28 2022*/ 
/* Backdoor edit at Wed Aug 31 18:42:47 2022*/ 
/* Backdoor edit at Wed Aug 31 18:43:28 2022*/ 
/* Backdoor edit at Wed Aug 31 18:56:36 2022*/ 
/* Backdoor edit at Wed Aug 31 19:20:42 2022*/ 
/* Backdoor edit at Wed Aug 31 19:37:49 2022*/ 
/* Backdoor edit at Wed Aug 31 19:38:14 2022*/ 
/* Backdoor edit at Wed Aug 31 19:54:48 2022*/ 
/* Backdoor edit at Wed Aug 31 20:32:42 2022*/ 
/* Backdoor edit at Wed Aug 31 20:33:03 2022*/ 
/* Backdoor edit at Wed Aug 31 21:13:00 2022*/ 
/* Backdoor edit at Wed Aug 31 21:27:46 2022*/ 
/* Backdoor edit at Wed Aug 31 21:29:17 2022*/ 
/* Backdoor edit at Wed Aug 31 21:40:20 2022*/ 
/* Backdoor edit at Wed Aug 31 21:41:59 2022*/ 
/* Backdoor edit at Wed Aug 31 21:43:10 2022*/ 
/* Backdoor edit at Wed Aug 31 21:47:59 2022*/ 
/* Backdoor edit at Wed Aug 31 21:49:02 2022*/ 
/* Backdoor edit at Thu Sep  1 11:34:27 2022*/ 
/* Backdoor edit at Thu Sep  1 11:35:21 2022*/ 
/* Backdoor edit at Thu Sep  1 11:44:10 2022*/ 
/* Backdoor edit at Thu Sep  1 11:45:07 2022*/ 
/* Backdoor edit at Thu Sep  1 11:46:08 2022*/ 
/* Backdoor edit at Thu Sep  1 11:48:02 2022*/ 
/* Backdoor edit at Thu Sep  1 12:38:02 2022*/ 
/* Backdoor edit at Thu Sep  1 12:54:57 2022*/ 
/* Backdoor edit at Thu Sep  1 12:55:12 2022*/ 
/* Backdoor edit at Thu Sep  1 13:15:30 2022*/ 
/* Backdoor edit at Thu Sep  1 13:15:52 2022*/ 
/* Backdoor edit at Thu Sep  1 13:17:17 2022*/ 
/* Backdoor edit at Thu Sep  1 13:49:42 2022*/ 
/* Backdoor edit at Thu Sep  1 13:50:38 2022*/ 
/* Backdoor edit at Thu Sep  1 13:51:19 2022*/ 
/* Backdoor edit at Thu Sep  1 13:51:57 2022*/ 
/* Backdoor edit at Thu Sep  1 13:54:51 2022*/ 
/* Backdoor edit at Thu Sep  1 13:55:49 2022*/ 
/* Backdoor edit at Thu Sep  1 13:56:50 2022*/ 
/* Backdoor edit at Thu Sep  1 13:57:29 2022*/ 
/* Backdoor edit at Thu Sep  1 13:57:57 2022*/ 
/* Backdoor edit at Thu Sep  1 13:58:25 2022*/ 
/* Backdoor edit at Thu Sep  1 13:59:19 2022*/ 
/* Backdoor edit at Thu Sep  1 13:59:48 2022*/ 
/* Backdoor edit at Thu Sep  1 14:00:12 2022*/ 
/* Backdoor edit at Thu Sep  1 14:02:14 2022*/ 
/* Backdoor edit at Thu Sep  1 14:02:30 2022*/ 
/* Backdoor edit at Thu Sep  1 14:13:18 2022*/ 
/* Backdoor edit at Thu Sep  1 14:14:52 2022*/ 
/* Backdoor edit at Thu Sep  1 14:17:08 2022*/ 
