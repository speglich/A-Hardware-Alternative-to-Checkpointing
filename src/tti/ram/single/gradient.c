#define _POSIX_C_SOURCE 200809L
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
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
  double section3;
} ;


int GradientTTI(struct dataobj *restrict damp_vec, struct dataobj *restrict delta_vec, struct dataobj *restrict dm_vec, const float dt, struct dataobj *restrict du_vec, struct dataobj *restrict dv_vec, struct dataobj *restrict epsilon_vec, const float o_x, const float o_y, const float o_z, struct dataobj *restrict phi_vec, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, struct dataobj *restrict theta_vec, struct dataobj *restrict u0_vec, struct dataobj *restrict v0_vec, struct dataobj *restrict vp_vec, const int x0_blk0_size, const int x_M, const int x_m, const int x_size, const int y0_blk0_size, const int y_M, const int y_m, const int y_size, const int z_M, const int z_m, const int z_size, const int p_rec_M, const int p_rec_m, const int time_M, const int time_m, const int x1_blk0_size, const int y1_blk0_size, const int nthreads, const int nthreads_nonaffine, struct profiler * timers)
{
  float **pr22_vec;
  posix_memalign((void**)(&pr22_vec),64,nthreads*sizeof(float*));
  float **pr23_vec;
  posix_memalign((void**)(&pr23_vec),64,nthreads*sizeof(float*));
  float **pr24_vec;
  posix_memalign((void**)(&pr24_vec),64,nthreads*sizeof(float*));
  float *r0_vec;
  posix_memalign((void**)(&r0_vec),64,(x_size + 8)*(y_size + 8)*(z_size + 8)*sizeof(float));
  float *r1_vec;
  posix_memalign((void**)(&r1_vec),64,(x_size + 8)*(y_size + 8)*(z_size + 8)*sizeof(float));
  float *r2_vec;
  posix_memalign((void**)(&r2_vec),64,(x_size + 8)*(y_size + 8)*(z_size + 8)*sizeof(float));
  float *r3_vec;
  posix_memalign((void**)(&r3_vec),64,(x_size + 8)*(y_size + 8)*(z_size + 8)*sizeof(float));
  #pragma omp parallel num_threads(nthreads)
  {
    const int tid = omp_get_thread_num();
    posix_memalign((void**)(&(pr22_vec[tid])),64,(x0_blk0_size + 8)*(y0_blk0_size + 8)*(z_size + 8)*sizeof(float));
    posix_memalign((void**)(&(pr23_vec[tid])),64,(x0_blk0_size + 3)*(y0_blk0_size + 3)*(z_size + 3)*sizeof(float));
    posix_memalign((void**)(&(pr24_vec[tid])),64,(x0_blk0_size + 3)*(y0_blk0_size + 3)*(z_size + 3)*sizeof(float));
  }

  float (*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]][damp_vec->size[2]]) damp_vec->data;
  float (*restrict delta)[delta_vec->size[1]][delta_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[delta_vec->size[1]][delta_vec->size[2]]) delta_vec->data;
  float (*restrict dm)[dm_vec->size[1]][dm_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[dm_vec->size[1]][dm_vec->size[2]]) dm_vec->data;
  float (*restrict du)[du_vec->size[1]][du_vec->size[2]][du_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[du_vec->size[1]][du_vec->size[2]][du_vec->size[3]]) du_vec->data;
  float (*restrict dv)[dv_vec->size[1]][dv_vec->size[2]][dv_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[dv_vec->size[1]][dv_vec->size[2]][dv_vec->size[3]]) dv_vec->data;
  float (*restrict epsilon)[epsilon_vec->size[1]][epsilon_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[epsilon_vec->size[1]][epsilon_vec->size[2]]) epsilon_vec->data;
  float (*restrict phi)[phi_vec->size[1]][phi_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[phi_vec->size[1]][phi_vec->size[2]]) phi_vec->data;
  float **pr22 = (float**) pr22_vec;
  float **pr23 = (float**) pr23_vec;
  float **pr24 = (float**) pr24_vec;
  float (*restrict r0)[y_size + 8][z_size + 8] __attribute__ ((aligned (64))) = (float (*)[y_size + 8][z_size + 8]) r0_vec;
  float (*restrict r1)[y_size + 8][z_size + 8] __attribute__ ((aligned (64))) = (float (*)[y_size + 8][z_size + 8]) r1_vec;
  float (*restrict r2)[y_size + 8][z_size + 8] __attribute__ ((aligned (64))) = (float (*)[y_size + 8][z_size + 8]) r2_vec;
  float (*restrict r3)[y_size + 8][z_size + 8] __attribute__ ((aligned (64))) = (float (*)[y_size + 8][z_size + 8]) r3_vec;
  float (*restrict rec)[rec_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_vec->size[1]]) rec_vec->data;
  float (*restrict rec_coords)[rec_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_coords_vec->size[1]]) rec_coords_vec->data;
  float (*restrict theta)[theta_vec->size[1]][theta_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[theta_vec->size[1]][theta_vec->size[2]]) theta_vec->data;
  float (*restrict u0)[u0_vec->size[1]][u0_vec->size[2]][u0_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u0_vec->size[1]][u0_vec->size[2]][u0_vec->size[3]]) u0_vec->data;
  float (*restrict v0)[v0_vec->size[1]][v0_vec->size[2]][v0_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[v0_vec->size[1]][v0_vec->size[2]][v0_vec->size[3]]) v0_vec->data;
  float (*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]][vp_vec->size[2]]) vp_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  float r4 = 1.0F/(dt*dt);
  float r5 = 1.0F/dt;

  /* Begin section0 */
  START_TIMER(section0)
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(2) schedule(dynamic,1)
    for (int x = x_m - 4; x <= x_M + 4; x += 1)
    {
      for (int y = y_m - 4; y <= y_M + 4; y += 1)
      {
        #pragma omp simd aligned(delta,phi,theta:32)
        for (int z = z_m - 4; z <= z_M + 4; z += 1)
        {
          r0[x + 4][y + 4][z + 4] = cos(theta[x + 8][y + 8][z + 8]);
          r1[x + 4][y + 4][z + 4] = sqrt(2*delta[x + 8][y + 8][z + 8] + 1);
          r2[x + 4][y + 4][z + 4] = sin(theta[x + 8][y + 8][z + 8])*cos(phi[x + 8][y + 8][z + 8]);
          r3[x + 4][y + 4][z + 4] = sin(phi[x + 8][y + 8][z + 8])*sin(theta[x + 8][y + 8][z + 8]);
        }
      }
    }
  }
  STOP_TIMER(section0,timers)
  /* End section0 */

  for (int time = time_M, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3); time >= time_m; time -= 1, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3))
  {
    /* Begin section1 */
    START_TIMER(section1)
    #pragma omp parallel num_threads(nthreads)
    {
      const int tid = omp_get_thread_num();
      float (*restrict r22)[y0_blk0_size + 8][z_size + 8] __attribute__ ((aligned (64))) = (float (*)[y0_blk0_size + 8][z_size + 8]) pr22[tid];
      float (*restrict r23)[y0_blk0_size + 3][z_size + 3] __attribute__ ((aligned (64))) = (float (*)[y0_blk0_size + 3][z_size + 3]) pr23[tid];
      float (*restrict r24)[y0_blk0_size + 3][z_size + 3] __attribute__ ((aligned (64))) = (float (*)[y0_blk0_size + 3][z_size + 3]) pr24[tid];

      #pragma omp for collapse(2) schedule(dynamic,1)
      for (int x0_blk0 = x_m; x0_blk0 <= x_M; x0_blk0 += x0_blk0_size)
      {
        for (int y0_blk0 = y_m; y0_blk0 <= y_M; y0_blk0 += y0_blk0_size)
        {
          for (int x = x0_blk0 - 4, xs = 0; x <= MIN(x0_blk0 + x0_blk0_size + 3, x_M + 4); x += 1, xs += 1)
          {
            for (int y = y0_blk0 - 4, ys = 0; y <= MIN(y0_blk0 + y0_blk0_size + 3, y_M + 4); y += 1, ys += 1)
            {
              #pragma omp simd aligned(du,dv,epsilon:32)
              for (int z = z_m - 4; z <= z_M + 4; z += 1)
              {
                r22[xs][ys][z + 4] = (2*epsilon[x + 8][y + 8][z + 8] + 1)*du[t0][x + 6][y + 6][z + 6] + r1[x + 4][y + 4][z + 4]*dv[t0][x + 6][y + 6][z + 6];
              }
            }
          }
          for (int x = x0_blk0 - 2, xs = 0; x <= MIN(x0_blk0 + x0_blk0_size, x_M + 1); x += 1, xs += 1)
          {
            for (int y = y0_blk0 - 2, ys = 0; y <= MIN(y0_blk0 + y0_blk0_size, y_M + 1); y += 1, ys += 1)
            {
              #pragma omp simd aligned(du,dv,epsilon:32)
              for (int z = z_m - 2; z <= z_M + 1; z += 1)
              {
                float r27 = 5.00000007e-2F*(r1[x + 5][y + 5][z + 5]*du[t0][x + 7][y + 7][z + 7] + dv[t0][x + 7][y + 7][z + 7]);
                float r26 = 5.00000007e-2F*((2*epsilon[x + 9][y + 9][z + 9] + 1)*du[t0][x + 7][y + 7][z + 7] + r1[x + 5][y + 5][z + 5]*dv[t0][x + 7][y + 7][z + 7]);
                r23[xs][ys][z + 2] = -(r26 + 1.66666669e-2F*((2*epsilon[x + 7][y + 9][z + 9] + 1)*du[t0][x + 5][y + 7][z + 7] + r1[x + 3][y + 5][z + 5]*dv[t0][x + 5][y + 7][z + 7]) + 1.0e-1F*(-(2*epsilon[x + 8][y + 9][z + 9] + 1)*du[t0][x + 6][y + 7][z + 7] - r1[x + 4][y + 5][z + 5]*dv[t0][x + 6][y + 7][z + 7]) + 3.33333338e-2F*((2*epsilon[x + 10][y + 9][z + 9] + 1)*du[t0][x + 8][y + 7][z + 7] + r1[x + 6][y + 5][z + 5]*dv[t0][x + 8][y + 7][z + 7]))*r2[x + 5][y + 5][z + 5] - (r26 + 1.66666669e-2F*((2*epsilon[x + 9][y + 7][z + 9] + 1)*du[t0][x + 7][y + 5][z + 7] + r1[x + 5][y + 3][z + 5]*dv[t0][x + 7][y + 5][z + 7]) + 1.0e-1F*(-(2*epsilon[x + 9][y + 8][z + 9] + 1)*du[t0][x + 7][y + 6][z + 7] - r1[x + 5][y + 4][z + 5]*dv[t0][x + 7][y + 6][z + 7]) + 3.33333338e-2F*((2*epsilon[x + 9][y + 10][z + 9] + 1)*du[t0][x + 7][y + 8][z + 7] + r1[x + 5][y + 6][z + 5]*dv[t0][x + 7][y + 8][z + 7]))*r3[x + 5][y + 5][z + 5] - (r26 + 1.66666669e-2F*((2*epsilon[x + 9][y + 9][z + 7] + 1)*du[t0][x + 7][y + 7][z + 5] + r1[x + 5][y + 5][z + 3]*dv[t0][x + 7][y + 7][z + 5]) + 1.0e-1F*(-(2*epsilon[x + 9][y + 9][z + 8] + 1)*du[t0][x + 7][y + 7][z + 6] - r1[x + 5][y + 5][z + 4]*dv[t0][x + 7][y + 7][z + 6]) + 3.33333338e-2F*((2*epsilon[x + 9][y + 9][z + 10] + 1)*du[t0][x + 7][y + 7][z + 8] + r1[x + 5][y + 5][z + 6]*dv[t0][x + 7][y + 7][z + 8]))*r0[x + 5][y + 5][z + 5];
                r24[xs][ys][z + 2] = -(r27 + 1.66666669e-2F*(r1[x + 3][y + 5][z + 5]*du[t0][x + 5][y + 7][z + 7] + dv[t0][x + 5][y + 7][z + 7]) + 1.0e-1F*(-r1[x + 4][y + 5][z + 5]*du[t0][x + 6][y + 7][z + 7] - dv[t0][x + 6][y + 7][z + 7]) + 3.33333338e-2F*(r1[x + 6][y + 5][z + 5]*du[t0][x + 8][y + 7][z + 7] + dv[t0][x + 8][y + 7][z + 7]))*r2[x + 5][y + 5][z + 5] - (r27 + 1.66666669e-2F*(r1[x + 5][y + 3][z + 5]*du[t0][x + 7][y + 5][z + 7] + dv[t0][x + 7][y + 5][z + 7]) + 1.0e-1F*(-r1[x + 5][y + 4][z + 5]*du[t0][x + 7][y + 6][z + 7] - dv[t0][x + 7][y + 6][z + 7]) + 3.33333338e-2F*(r1[x + 5][y + 6][z + 5]*du[t0][x + 7][y + 8][z + 7] + dv[t0][x + 7][y + 8][z + 7]))*r3[x + 5][y + 5][z + 5] - (r27 + 1.66666669e-2F*(r1[x + 5][y + 5][z + 3]*du[t0][x + 7][y + 7][z + 5] + dv[t0][x + 7][y + 7][z + 5]) + 1.0e-1F*(-r1[x + 5][y + 5][z + 4]*du[t0][x + 7][y + 7][z + 6] - dv[t0][x + 7][y + 7][z + 6]) + 3.33333338e-2F*(r1[x + 5][y + 5][z + 6]*du[t0][x + 7][y + 7][z + 8] + dv[t0][x + 7][y + 7][z + 8]))*r0[x + 5][y + 5][z + 5];
              }
            }
          }
          for (int x = x0_blk0, xs = 0; x <= MIN(x0_blk0 + x0_blk0_size - 1, x_M); x += 1, xs += 1)
          {
            for (int y = y0_blk0, ys = 0; y <= MIN(y0_blk0 + y0_blk0_size - 1, y_M); y += 1, ys += 1)
            {
              #pragma omp simd aligned(damp,du,dv,vp:32)
              for (int z = z_m; z <= z_M; z += 1)
              {
                float r29 = 1.0F/(vp[x + 8][y + 8][z + 8]*vp[x + 8][y + 8][z + 8]);
                float r28 = 1.0F/(r4*r29 + r5*damp[x + 1][y + 1][z + 1]);
                du[t1][x + 6][y + 6][z + 6] = r28*(r5*damp[x + 1][y + 1][z + 1]*du[t0][x + 6][y + 6][z + 6] + r29*(-r4*(-2.0F*du[t0][x + 6][y + 6][z + 6]) - r4*du[t2][x + 6][y + 6][z + 6]) + 3.33333338e-2F*(-r0[x + 4][y + 4][z + 3]*r23[xs + 1][ys + 1][z] - r2[x + 3][y + 4][z + 4]*r23[xs][ys + 1][z + 1] - r23[xs + 1][ys][z + 1]*r3[x + 4][y + 3][z + 4]) + 5.00000007e-2F*(-r0[x + 4][y + 4][z + 4]*r23[xs + 1][ys + 1][z + 1] - r2[x + 4][y + 4][z + 4]*r23[xs + 1][ys + 1][z + 1] - r23[xs + 1][ys + 1][z + 1]*r3[x + 4][y + 4][z + 4]) + 1.0e-1F*(r0[x + 4][y + 4][z + 5]*r23[xs + 1][ys + 1][z + 2] + r2[x + 5][y + 4][z + 4]*r23[xs + 2][ys + 1][z + 1] + r23[xs + 1][ys + 2][z + 1]*r3[x + 4][y + 5][z + 4]) + 1.66666669e-2F*(-r0[x + 4][y + 4][z + 6]*r23[xs + 1][ys + 1][z + 3] - r2[x + 6][y + 4][z + 4]*r23[xs + 3][ys + 1][z + 1] - r23[xs + 1][ys + 3][z + 1]*r3[x + 4][y + 6][z + 4]) + 1.78571425e-5F*(-r22[xs][ys + 4][z + 4] - r22[xs + 4][ys][z + 4] - r22[xs + 4][ys + 4][z] - r22[xs + 4][ys + 4][z + 8] - r22[xs + 4][ys + 8][z + 4] - r22[xs + 8][ys + 4][z + 4]) + 2.53968248e-4F*(r22[xs + 1][ys + 4][z + 4] + r22[xs + 4][ys + 1][z + 4] + r22[xs + 4][ys + 4][z + 1] + r22[xs + 4][ys + 4][z + 7] + r22[xs + 4][ys + 7][z + 4] + r22[xs + 7][ys + 4][z + 4]) + 1.99999996e-3F*(-r22[xs + 2][ys + 4][z + 4] - r22[xs + 4][ys + 2][z + 4] - r22[xs + 4][ys + 4][z + 2] - r22[xs + 4][ys + 4][z + 6] - r22[xs + 4][ys + 6][z + 4] - r22[xs + 6][ys + 4][z + 4]) + 1.59999996e-2F*(r22[xs + 3][ys + 4][z + 4] + r22[xs + 4][ys + 3][z + 4] + r22[xs + 4][ys + 4][z + 3] + r22[xs + 4][ys + 4][z + 5] + r22[xs + 4][ys + 5][z + 4] + r22[xs + 5][ys + 4][z + 4]) - 8.54166647e-2F*r22[xs + 4][ys + 4][z + 4]);
                dv[t1][x + 6][y + 6][z + 6] = r28*(r5*damp[x + 1][y + 1][z + 1]*dv[t0][x + 6][y + 6][z + 6] + r29*(-r4*(-2.0F*dv[t0][x + 6][y + 6][z + 6]) - r4*dv[t2][x + 6][y + 6][z + 6]) + 3.33333338e-2F*(r0[x + 4][y + 4][z + 3]*r24[xs + 1][ys + 1][z] + r2[x + 3][y + 4][z + 4]*r24[xs][ys + 1][z + 1] + r24[xs + 1][ys][z + 1]*r3[x + 4][y + 3][z + 4]) + 5.00000007e-2F*(r0[x + 4][y + 4][z + 4]*r24[xs + 1][ys + 1][z + 1] + r2[x + 4][y + 4][z + 4]*r24[xs + 1][ys + 1][z + 1] + r24[xs + 1][ys + 1][z + 1]*r3[x + 4][y + 4][z + 4]) + 1.0e-1F*(-r0[x + 4][y + 4][z + 5]*r24[xs + 1][ys + 1][z + 2] - r2[x + 5][y + 4][z + 4]*r24[xs + 2][ys + 1][z + 1] - r24[xs + 1][ys + 2][z + 1]*r3[x + 4][y + 5][z + 4]) + 1.66666669e-2F*(r0[x + 4][y + 4][z + 6]*r24[xs + 1][ys + 1][z + 3] + r2[x + 6][y + 4][z + 4]*r24[xs + 3][ys + 1][z + 1] + r24[xs + 1][ys + 3][z + 1]*r3[x + 4][y + 6][z + 4]));
              }
            }
          }
        }
      }
    }
    STOP_TIMER(section1,timers)
    /* End section1 */

    /* Begin section2 */
    START_TIMER(section2)
    #pragma omp parallel num_threads(nthreads_nonaffine)
    {
      int chunk_size = (int)(fmax(1, (1.0F/3.0F)*(p_rec_M - p_rec_m + 1)/nthreads_nonaffine));
      #pragma omp for collapse(1) schedule(dynamic,chunk_size)
      for (int p_rec = p_rec_m; p_rec <= p_rec_M; p_rec += 1)
      {
        float posx = -o_x + rec_coords[p_rec][0];
        float posy = -o_y + rec_coords[p_rec][1];
        float posz = -o_z + rec_coords[p_rec][2];
        int ii_rec_0 = (int)(floor(1.0e-1F*posx));
        int ii_rec_1 = (int)(floor(1.0e-1F*posy));
        int ii_rec_2 = (int)(floor(1.0e-1F*posz));
        int ii_rec_3 = 1 + (int)(floor(1.0e-1F*posz));
        int ii_rec_4 = 1 + (int)(floor(1.0e-1F*posy));
        int ii_rec_5 = 1 + (int)(floor(1.0e-1F*posx));
        float px = (float)(posx - 1.0e+1F*(int)(floor(1.0e-1F*posx)));
        float py = (float)(posy - 1.0e+1F*(int)(floor(1.0e-1F*posy)));
        float pz = (float)(posz - 1.0e+1F*(int)(floor(1.0e-1F*posz)));
        if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1 && ii_rec_2 <= z_M + 1)
        {
          float r6 = (dt*dt)*(vp[ii_rec_0 + 8][ii_rec_1 + 8][ii_rec_2 + 8]*vp[ii_rec_0 + 8][ii_rec_1 + 8][ii_rec_2 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py + 1.0e-2F*px*pz - 1.0e-1F*px + 1.0e-2F*py*pz - 1.0e-1F*py - 1.0e-1F*pz + 1)*rec[time][p_rec];
          #pragma omp atomic update
          du[t1][ii_rec_0 + 6][ii_rec_1 + 6][ii_rec_2 + 6] += r6;
        }
        if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= z_M + 1)
        {
          float r7 = (dt*dt)*(vp[ii_rec_0 + 8][ii_rec_1 + 8][ii_rec_3 + 8]*vp[ii_rec_0 + 8][ii_rec_1 + 8][ii_rec_3 + 8])*(1.0e-3F*px*py*pz - 1.0e-2F*px*pz - 1.0e-2F*py*pz + 1.0e-1F*pz)*rec[time][p_rec];
          #pragma omp atomic update
          du[t1][ii_rec_0 + 6][ii_rec_1 + 6][ii_rec_3 + 6] += r7;
        }
        if (ii_rec_0 >= x_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_2 <= z_M + 1 && ii_rec_4 <= y_M + 1)
        {
          float r8 = (dt*dt)*(vp[ii_rec_0 + 8][ii_rec_4 + 8][ii_rec_2 + 8]*vp[ii_rec_0 + 8][ii_rec_4 + 8][ii_rec_2 + 8])*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*py*pz + 1.0e-1F*py)*rec[time][p_rec];
          #pragma omp atomic update
          du[t1][ii_rec_0 + 6][ii_rec_4 + 6][ii_rec_2 + 6] += r8;
        }
        if (ii_rec_0 >= x_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_3 <= z_M + 1 && ii_rec_4 <= y_M + 1)
        {
          float r9 = (dt*dt)*(vp[ii_rec_0 + 8][ii_rec_4 + 8][ii_rec_3 + 8]*vp[ii_rec_0 + 8][ii_rec_4 + 8][ii_rec_3 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*py*pz)*rec[time][p_rec];
          #pragma omp atomic update
          du[t1][ii_rec_0 + 6][ii_rec_4 + 6][ii_rec_3 + 6] += r9;
        }
        if (ii_rec_1 >= y_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_2 <= z_M + 1 && ii_rec_5 <= x_M + 1)
        {
          float r10 = (dt*dt)*(vp[ii_rec_5 + 8][ii_rec_1 + 8][ii_rec_2 + 8]*vp[ii_rec_5 + 8][ii_rec_1 + 8][ii_rec_2 + 8])*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*px*pz + 1.0e-1F*px)*rec[time][p_rec];
          #pragma omp atomic update
          du[t1][ii_rec_5 + 6][ii_rec_1 + 6][ii_rec_2 + 6] += r10;
        }
        if (ii_rec_1 >= y_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= z_M + 1 && ii_rec_5 <= x_M + 1)
        {
          float r11 = (dt*dt)*(vp[ii_rec_5 + 8][ii_rec_1 + 8][ii_rec_3 + 8]*vp[ii_rec_5 + 8][ii_rec_1 + 8][ii_rec_3 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*pz)*rec[time][p_rec];
          #pragma omp atomic update
          du[t1][ii_rec_5 + 6][ii_rec_1 + 6][ii_rec_3 + 6] += r11;
        }
        if (ii_rec_2 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_2 <= z_M + 1 && ii_rec_4 <= y_M + 1 && ii_rec_5 <= x_M + 1)
        {
          float r12 = (dt*dt)*(vp[ii_rec_5 + 8][ii_rec_4 + 8][ii_rec_2 + 8]*vp[ii_rec_5 + 8][ii_rec_4 + 8][ii_rec_2 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py)*rec[time][p_rec];
          #pragma omp atomic update
          du[t1][ii_rec_5 + 6][ii_rec_4 + 6][ii_rec_2 + 6] += r12;
        }
        if (ii_rec_3 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_3 <= z_M + 1 && ii_rec_4 <= y_M + 1 && ii_rec_5 <= x_M + 1)
        {
          float r13 = 1.0e-3F*px*py*pz*(dt*dt)*(vp[ii_rec_5 + 8][ii_rec_4 + 8][ii_rec_3 + 8]*vp[ii_rec_5 + 8][ii_rec_4 + 8][ii_rec_3 + 8])*rec[time][p_rec];
          #pragma omp atomic update
          du[t1][ii_rec_5 + 6][ii_rec_4 + 6][ii_rec_3 + 6] += r13;
        }
        posx = -o_x + rec_coords[p_rec][0];
        posy = -o_y + rec_coords[p_rec][1];
        posz = -o_z + rec_coords[p_rec][2];
        ii_rec_0 = (int)(floor(1.0e-1F*posx));
        ii_rec_1 = (int)(floor(1.0e-1F*posy));
        ii_rec_2 = (int)(floor(1.0e-1F*posz));
        ii_rec_3 = 1 + (int)(floor(1.0e-1F*posz));
        ii_rec_4 = 1 + (int)(floor(1.0e-1F*posy));
        ii_rec_5 = 1 + (int)(floor(1.0e-1F*posx));
        px = (float)(posx - 1.0e+1F*(int)(floor(1.0e-1F*posx)));
        py = (float)(posy - 1.0e+1F*(int)(floor(1.0e-1F*posy)));
        pz = (float)(posz - 1.0e+1F*(int)(floor(1.0e-1F*posz)));
        if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1 && ii_rec_2 <= z_M + 1)
        {
          float r14 = (dt*dt)*(vp[ii_rec_0 + 8][ii_rec_1 + 8][ii_rec_2 + 8]*vp[ii_rec_0 + 8][ii_rec_1 + 8][ii_rec_2 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py + 1.0e-2F*px*pz - 1.0e-1F*px + 1.0e-2F*py*pz - 1.0e-1F*py - 1.0e-1F*pz + 1)*rec[time][p_rec];
          #pragma omp atomic update
          dv[t1][ii_rec_0 + 6][ii_rec_1 + 6][ii_rec_2 + 6] += r14;
        }
        if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= z_M + 1)
        {
          float r15 = (dt*dt)*(vp[ii_rec_0 + 8][ii_rec_1 + 8][ii_rec_3 + 8]*vp[ii_rec_0 + 8][ii_rec_1 + 8][ii_rec_3 + 8])*(1.0e-3F*px*py*pz - 1.0e-2F*px*pz - 1.0e-2F*py*pz + 1.0e-1F*pz)*rec[time][p_rec];
          #pragma omp atomic update
          dv[t1][ii_rec_0 + 6][ii_rec_1 + 6][ii_rec_3 + 6] += r15;
        }
        if (ii_rec_0 >= x_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_2 <= z_M + 1 && ii_rec_4 <= y_M + 1)
        {
          float r16 = (dt*dt)*(vp[ii_rec_0 + 8][ii_rec_4 + 8][ii_rec_2 + 8]*vp[ii_rec_0 + 8][ii_rec_4 + 8][ii_rec_2 + 8])*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*py*pz + 1.0e-1F*py)*rec[time][p_rec];
          #pragma omp atomic update
          dv[t1][ii_rec_0 + 6][ii_rec_4 + 6][ii_rec_2 + 6] += r16;
        }
        if (ii_rec_0 >= x_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_3 <= z_M + 1 && ii_rec_4 <= y_M + 1)
        {
          float r17 = (dt*dt)*(vp[ii_rec_0 + 8][ii_rec_4 + 8][ii_rec_3 + 8]*vp[ii_rec_0 + 8][ii_rec_4 + 8][ii_rec_3 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*py*pz)*rec[time][p_rec];
          #pragma omp atomic update
          dv[t1][ii_rec_0 + 6][ii_rec_4 + 6][ii_rec_3 + 6] += r17;
        }
        if (ii_rec_1 >= y_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_2 <= z_M + 1 && ii_rec_5 <= x_M + 1)
        {
          float r18 = (dt*dt)*(vp[ii_rec_5 + 8][ii_rec_1 + 8][ii_rec_2 + 8]*vp[ii_rec_5 + 8][ii_rec_1 + 8][ii_rec_2 + 8])*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*px*pz + 1.0e-1F*px)*rec[time][p_rec];
          #pragma omp atomic update
          dv[t1][ii_rec_5 + 6][ii_rec_1 + 6][ii_rec_2 + 6] += r18;
        }
        if (ii_rec_1 >= y_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= z_M + 1 && ii_rec_5 <= x_M + 1)
        {
          float r19 = (dt*dt)*(vp[ii_rec_5 + 8][ii_rec_1 + 8][ii_rec_3 + 8]*vp[ii_rec_5 + 8][ii_rec_1 + 8][ii_rec_3 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*pz)*rec[time][p_rec];
          #pragma omp atomic update
          dv[t1][ii_rec_5 + 6][ii_rec_1 + 6][ii_rec_3 + 6] += r19;
        }
        if (ii_rec_2 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_2 <= z_M + 1 && ii_rec_4 <= y_M + 1 && ii_rec_5 <= x_M + 1)
        {
          float r20 = (dt*dt)*(vp[ii_rec_5 + 8][ii_rec_4 + 8][ii_rec_2 + 8]*vp[ii_rec_5 + 8][ii_rec_4 + 8][ii_rec_2 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py)*rec[time][p_rec];
          #pragma omp atomic update
          dv[t1][ii_rec_5 + 6][ii_rec_4 + 6][ii_rec_2 + 6] += r20;
        }
        if (ii_rec_3 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_3 <= z_M + 1 && ii_rec_4 <= y_M + 1 && ii_rec_5 <= x_M + 1)
        {
          float r21 = 1.0e-3F*px*py*pz*(dt*dt)*(vp[ii_rec_5 + 8][ii_rec_4 + 8][ii_rec_3 + 8]*vp[ii_rec_5 + 8][ii_rec_4 + 8][ii_rec_3 + 8])*rec[time][p_rec];
          #pragma omp atomic update
          dv[t1][ii_rec_5 + 6][ii_rec_4 + 6][ii_rec_3 + 6] += r21;
        }
      }
    }
    STOP_TIMER(section2,timers)
    /* End section2 */

    /* Begin section3 */
    START_TIMER(section3)
    #pragma omp parallel num_threads(nthreads)
    {
      #pragma omp for collapse(2) schedule(dynamic,1)
      for (int x1_blk0 = x_m; x1_blk0 <= x_M; x1_blk0 += x1_blk0_size)
      {
        for (int y1_blk0 = y_m; y1_blk0 <= y_M; y1_blk0 += y1_blk0_size)
        {
          for (int x = x1_blk0; x <= MIN(x1_blk0 + x1_blk0_size - 1, x_M); x += 1)
          {
            for (int y = y1_blk0; y <= MIN(y1_blk0 + y1_blk0_size - 1, y_M); y += 1)
            {
              #pragma omp simd aligned(dm,du,dv,u0,v0:32)
              for (int z = z_m; z <= z_M; z += 1)
              {
                dm[x + 1][y + 1][z + 1] += -(r4*(-2.0F*du[t0][x + 6][y + 6][z + 6]) + r4*du[t1][x + 6][y + 6][z + 6] + r4*du[t2][x + 6][y + 6][z + 6])*u0[time][x + 6][y + 6][z + 6] - (r4*(-2.0F*dv[t0][x + 6][y + 6][z + 6]) + r4*dv[t1][x + 6][y + 6][z + 6] + r4*dv[t2][x + 6][y + 6][z + 6])*v0[time][x + 6][y + 6][z + 6];
              }
            }
          }
        }
      }
    }
    STOP_TIMER(section3,timers)
    /* End section3 */
  }

  #pragma omp parallel num_threads(nthreads)
  {
    const int tid = omp_get_thread_num();
    free(pr22_vec[tid]);
    free(pr23_vec[tid]);
    free(pr24_vec[tid]);
  }
  free(pr22_vec);
  free(pr23_vec);
  free(pr24_vec);
  free(r0_vec);
  free(r1_vec);
  free(r2_vec);
  free(r3_vec);

  return 0;
}
