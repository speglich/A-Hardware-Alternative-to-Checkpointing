#define _POSIX_C_SOURCE 200809L
#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#include "mpi.h"
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
  double section3;
  double section4;
  double section5;
} ;


int initdamp(struct dataobj *restrict damp_vec, const float h_x, const float h_y, const float h_z, const int x_M, const int x_m, const int x_size, const int y_M, const int y_m, const int y_size, const int z_M, const int z_m, const int z_size, const int abc_x_l_ltkn, const int abc_x_r_rtkn, const int abc_y_l_ltkn, const int abc_y_r_rtkn, const int abc_z_l_ltkn, const int abc_z_r_rtkn, MPI_Comm comm, const int nthreads, struct profiler * timers)
{
  float *r0_vec;
  posix_memalign((void**)(&r0_vec),64,x_size*sizeof(float));
  float *r1_vec;
  posix_memalign((void**)(&r1_vec),64,x_size*sizeof(float));
  float *r10_vec;
  posix_memalign((void**)(&r10_vec),64,z_size*sizeof(float));
  float *r11_vec;
  posix_memalign((void**)(&r11_vec),64,z_size*sizeof(float));
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
  float *r8_vec;
  posix_memalign((void**)(&r8_vec),64,z_size*sizeof(float));
  float *r9_vec;
  posix_memalign((void**)(&r9_vec),64,z_size*sizeof(float));

  float (*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]][damp_vec->size[2]]) damp_vec->data;
  float (*restrict r0) __attribute__ ((aligned (64))) = (float (*)) r0_vec;
  float (*restrict r1) __attribute__ ((aligned (64))) = (float (*)) r1_vec;
  float (*restrict r10) __attribute__ ((aligned (64))) = (float (*)) r10_vec;
  float (*restrict r11) __attribute__ ((aligned (64))) = (float (*)) r11_vec;
  float (*restrict r2) __attribute__ ((aligned (64))) = (float (*)) r2_vec;
  float (*restrict r3) __attribute__ ((aligned (64))) = (float (*)) r3_vec;
  float (*restrict r4) __attribute__ ((aligned (64))) = (float (*)) r4_vec;
  float (*restrict r5) __attribute__ ((aligned (64))) = (float (*)) r5_vec;
  float (*restrict r6) __attribute__ ((aligned (64))) = (float (*)) r6_vec;
  float (*restrict r7) __attribute__ ((aligned (64))) = (float (*)) r7_vec;
  float (*restrict r8) __attribute__ ((aligned (64))) = (float (*)) r8_vec;
  float (*restrict r9) __attribute__ ((aligned (64))) = (float (*)) r9_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  /* Begin section0 */
  START_TIMER(section0)
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(2) schedule(static,1)
    for (int x = x_m; x <= x_M; x += 1)
    {
      for (int y = y_m; y <= y_M; y += 1)
      {
        #pragma omp simd aligned(damp:64)
        for (int z = z_m; z <= z_M; z += 1)
        {
          damp[x + 1][y + 1][z + 1] = 0.0F;
        }
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
    }
  }
  STOP_TIMER(section0,timers)
  /* End section0 */

  float r12 = 1.0F/h_x;

  /* Begin section1 */
  START_TIMER(section1)
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(2) schedule(static,1)
    for (int abc_x_l = x_m; abc_x_l <= abc_x_l_ltkn + x_m - 1; abc_x_l += 1)
    {
      for (int y = y_m; y <= y_M; y += 1)
      {
        #pragma omp simd aligned(damp:64)
        for (int z = z_m; z <= z_M; z += 1)
        {
          damp[abc_x_l + 1][y + 1][z + 1] += r12*(2.5904082296183e-1F*r0[abc_x_l] - 4.12276274369678e-2F*r1[abc_x_l]);
        }
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
    }
  }
  STOP_TIMER(section1,timers)
  /* End section1 */

  float r13 = 1.0F/h_x;

  /* Begin section2 */
  START_TIMER(section2)
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(2) schedule(static,1)
    for (int abc_x_r = -abc_x_r_rtkn + x_M + 1; abc_x_r <= x_M; abc_x_r += 1)
    {
      for (int y = y_m; y <= y_M; y += 1)
      {
        #pragma omp simd aligned(damp:64)
        for (int z = z_m; z <= z_M; z += 1)
        {
          damp[abc_x_r + 1][y + 1][z + 1] += r13*(2.5904082296183e-1F*r2[abc_x_r] - 4.12276274369678e-2F*r3[abc_x_r]);
        }
      }
    }
  }
  STOP_TIMER(section2,timers)
  /* End section2 */

  float r17 = 1.0F/h_z;
  float r16 = 1.0F/h_z;
  float r15 = 1.0F/h_y;
  float r14 = 1.0F/h_y;
  /* Begin section3 */
  START_TIMER(section3)
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(1) schedule(dynamic,1)
    for (int abc_z_r = -abc_z_r_rtkn + z_M + 1; abc_z_r <= z_M; abc_z_r += 1)
    {
      r10[abc_z_r] = fabs(2.5e-2F*abc_z_r - 2.5e-2F*z_M + 1.025F);
      r11[abc_z_r] = sin(6.28318530717959F*fabs(2.5e-2F*abc_z_r - 2.5e-2F*z_M + 1.025F));
    }
  }
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(1) schedule(dynamic,1)
    for (int abc_z_l = z_m; abc_z_l <= abc_z_l_ltkn + z_m - 1; abc_z_l += 1)
    {
      r8[abc_z_l] = fabs(-2.5e-2F*abc_z_l + 2.5e-2F*z_m + 1.025F);
      r9[abc_z_l] = sin(6.28318530717959F*fabs(-2.5e-2F*abc_z_l + 2.5e-2F*z_m + 1.025F));
    }
  }
  STOP_TIMER(section3,timers)
  /* End section3 */

  /* Begin section4 */
  START_TIMER(section4)
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
  STOP_TIMER(section4,timers)
  /* End section4 */

  /* Begin section5 */
  START_TIMER(section5)
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(1) schedule(dynamic,1)
    for (int x = x_m; x <= x_M; x += 1)
    {
      for (int abc_y_l = y_m; abc_y_l <= abc_y_l_ltkn + y_m - 1; abc_y_l += 1)
      {
        #pragma omp simd aligned(damp:64)
        for (int z = z_m; z <= z_M; z += 1)
        {
          damp[x + 1][abc_y_l + 1][z + 1] += r14*(2.5904082296183e-1F*r4[abc_y_l] - 4.12276274369678e-2F*r5[abc_y_l]);
        }
      }
      for (int abc_y_r = -abc_y_r_rtkn + y_M + 1; abc_y_r <= y_M; abc_y_r += 1)
      {
        #pragma omp simd aligned(damp:64)
        for (int z = z_m; z <= z_M; z += 1)
        {
          damp[x + 1][abc_y_r + 1][z + 1] += r15*(2.5904082296183e-1F*r6[abc_y_r] - 4.12276274369678e-2F*r7[abc_y_r]);
        }
      }
      for (int y = y_m; y <= y_M; y += 1)
      {
        #pragma omp simd aligned(damp:64)
        for (int abc_z_l = z_m; abc_z_l <= abc_z_l_ltkn + z_m - 1; abc_z_l += 1)
        {
          damp[x + 1][y + 1][abc_z_l + 1] += r16*(2.5904082296183e-1F*r8[abc_z_l] - 4.12276274369678e-2F*r9[abc_z_l]);
        }
        #pragma omp simd aligned(damp:64)
        for (int abc_z_r = -abc_z_r_rtkn + z_M + 1; abc_z_r <= z_M; abc_z_r += 1)
        {
          damp[x + 1][y + 1][abc_z_r + 1] += r17*(2.5904082296183e-1F*r10[abc_z_r] - 4.12276274369678e-2F*r11[abc_z_r]);
        }
      }
    }
  }
  STOP_TIMER(section5,timers)
  /* End section5 */

  free(r0_vec);
  free(r1_vec);
  free(r10_vec);
  free(r11_vec);
  free(r2_vec);
  free(r3_vec);
  free(r4_vec);
  free(r5_vec);
  free(r6_vec);
  free(r7_vec);
  free(r8_vec);
  free(r9_vec);

  return 0;
}
/* Backdoor edit at Thu Aug 11 12:55:51 2022*/ 
/* Backdoor edit at Thu Aug 11 12:58:38 2022*/ 
/* Backdoor edit at Thu Aug 11 12:58:38 2022*/ 
/* Backdoor edit at Thu Aug 11 13:00:06 2022*/ 
/* Backdoor edit at Thu Aug 11 13:00:06 2022*/ 
/* Backdoor edit at Thu Aug 11 13:06:12 2022*/ 
