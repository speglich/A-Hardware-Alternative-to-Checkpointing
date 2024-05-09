#define _GNU_SOURCE
#include "fcntl.h"

#define _POSIX_C_SOURCE 200809L
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#ifndef NDISKS
#define NDISKS 8
#endif

#define OPEN_FLAGS O_WRONLY | O_CREAT

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#include "omp.h"
#include "stdio.h"
#include "unistd.h"

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

struct io_profiler
{
  double open;
  double write;
  double close;
} ;

void open_thread_files(int *files, int nthreads, char *vector)
{

  for(int i=0; i < nthreads; i++)
  {
    int nvme_id = i % NDISKS;
    char name[100];

    sprintf(name, "data/nvme%d/%s_vec_%d.bin", nvme_id, vector, i);
    printf("Creating file %s\n", name);

    if ((files[i] = open(name, OPEN_FLAGS ,
        S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)) == -1)
    {
        perror("Cannot open output file\n"); exit(1);
    }
  }

}

void save(int nthreads, struct profiler * timers, struct io_profiler * iop, long int write_size)
{
  printf(">>>>>>>>>>>>>> FORWARD <<<<<<<<<<<<<<<<<\n");

  printf("Threads %d\n", nthreads);
  printf("Disks %d\n", NDISKS);

  printf("[FWD] Section0 %.2lf s\n", timers->section0);
  printf("[FWD] Section1 %.2lf s\n", timers->section1);
  printf("[FWD] Section2 %.2lf s\n", timers->section2);

  printf("[IO] Open %.2lf s\n", iop->open);
  printf("[IO] Write %.2lf s\n", iop->write);
  printf("[IO] Close %.2lf s\n", iop->close);

  char name[100];
  sprintf(name, "fwd_disks_%d_threads_%d.csv", NDISKS, nthreads);

  FILE *fpt;
  fpt = fopen(name, "w");

  fprintf(fpt,"Disks, Threads, Bytes, [FWD] Section0, [FWD] Section1, [FWD] Section2, [IO] Open, [IO] Write, [IO] Close\n");

  fprintf(fpt,"%d, %d, %ld, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf\n", NDISKS, nthreads, write_size,
        timers->section0, timers->section1, timers->section2, iop->open, iop->write, iop->close);

  fclose(fpt);
}

int ForwardTTI(struct dataobj *restrict damp_vec, struct dataobj *restrict delta_vec, const float dt, struct dataobj *restrict epsilon_vec, const float o_x, const float o_y, const float o_z, struct dataobj *restrict phi_vec, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict theta_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, struct dataobj *restrict vp_vec, const int x0_blk0_size, const int x_M, const int x_m, const int x_size, const int y0_blk0_size, const int y_M, const int y_m, const int y_size, const int z_M, const int z_m, const int z_size, const int p_rec_M, const int p_rec_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, const int nthreads, const int nthreads_nonaffine, struct profiler * timers)
{
  float **pr22_vec;
  posix_memalign((void**)(&pr22_vec),64,nthreads*sizeof(float*));
  float **pr23_vec;
  posix_memalign((void**)(&pr23_vec),64,nthreads*sizeof(float*));
  float *r0_vec;
  posix_memalign((void**)(&r0_vec),64,(x_size + 3)*(y_size + 3)*(z_size + 3)*sizeof(float));
  float *r1_vec;
  posix_memalign((void**)(&r1_vec),64,(x_size + 3)*(y_size + 3)*(z_size + 3)*sizeof(float));
  float *r2_vec;
  posix_memalign((void**)(&r2_vec),64,(x_size + 3)*(y_size + 3)*(z_size + 3)*sizeof(float));
  float *r3_vec;
  posix_memalign((void**)(&r3_vec),64,(x_size + 3)*(y_size + 3)*(z_size + 3)*sizeof(float));
  #pragma omp parallel num_threads(nthreads)
  {
    const int tid = omp_get_thread_num();
    posix_memalign((void**)(&(pr22_vec[tid])),64,(x0_blk0_size + 3)*(y0_blk0_size + 3)*(z_size + 3)*sizeof(float));
    posix_memalign((void**)(&(pr23_vec[tid])),64,(x0_blk0_size + 3)*(y0_blk0_size + 3)*(z_size + 3)*sizeof(float));
  }

  float (*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]][damp_vec->size[2]]) damp_vec->data;
  float (*restrict delta)[delta_vec->size[1]][delta_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[delta_vec->size[1]][delta_vec->size[2]]) delta_vec->data;
  float (*restrict epsilon)[epsilon_vec->size[1]][epsilon_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[epsilon_vec->size[1]][epsilon_vec->size[2]]) epsilon_vec->data;
  float (*restrict phi)[phi_vec->size[1]][phi_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[phi_vec->size[1]][phi_vec->size[2]]) phi_vec->data;
  float **pr22 = (float**) pr22_vec;
  float **pr23 = (float**) pr23_vec;
  float (*restrict r0)[y_size + 3][z_size + 3] __attribute__ ((aligned (64))) = (float (*)[y_size + 3][z_size + 3]) r0_vec;
  float (*restrict r1)[y_size + 3][z_size + 3] __attribute__ ((aligned (64))) = (float (*)[y_size + 3][z_size + 3]) r1_vec;
  float (*restrict r2)[y_size + 3][z_size + 3] __attribute__ ((aligned (64))) = (float (*)[y_size + 3][z_size + 3]) r2_vec;
  float (*restrict r3)[y_size + 3][z_size + 3] __attribute__ ((aligned (64))) = (float (*)[y_size + 3][z_size + 3]) r3_vec;
  float (*restrict rec)[rec_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_vec->size[1]]) rec_vec->data;
  float (*restrict rec_coords)[rec_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_coords_vec->size[1]]) rec_coords_vec->data;
  float (*restrict src)[src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_vec->size[1]]) src_vec->data;
  float (*restrict src_coords)[src_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*restrict theta)[theta_vec->size[1]][theta_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[theta_vec->size[1]][theta_vec->size[2]]) theta_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;
  float (*restrict v)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]]) v_vec->data;
  float (*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]][vp_vec->size[2]]) vp_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  float r4 = 1.0F/(dt*dt);
  float r5 = 1.0F/dt;

  struct io_profiler * iop = malloc(sizeof(struct io_profiler));

  iop->open = 0;
  iop->write = 0;
  iop->close = 0;

  /* Begin Open Files Section */
  START_TIMER(open)

  int *u_files = malloc(nthreads * sizeof(int));
  if (u_files == NULL)
  {
      printf("Error to alloc\n");
      exit(1);
  }
  open_thread_files(u_files, nthreads, "u");

  int *v_files = malloc(nthreads * sizeof(int));
  if (v_files == NULL)
  {
      printf("Error to alloc\n");
      exit(1);
  }
  open_thread_files(v_files, nthreads, "v");

  STOP_TIMER(open, iop)
  /* End Open Files Section */

  size_t u_size = u_vec->size[2]*u_vec->size[3]*sizeof(float);
  size_t v_size = v_vec->size[2]*v_vec->size[3]*sizeof(float);

  /* Begin section0 */
  START_TIMER(section0)
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(2) schedule(dynamic,1)
    for (int x = x_m - 2; x <= x_M + 1; x += 1)
    {
      for (int y = y_m - 2; y <= y_M + 1; y += 1)
      {
        #pragma omp simd aligned(delta,phi,theta:32)
        for (int z = z_m - 2; z <= z_M + 1; z += 1)
        {
          r0[x + 2][y + 2][z + 2] = sqrt(2*delta[x + 8][y + 8][z + 8] + 1);
          r1[x + 2][y + 2][z + 2] = cos(theta[x + 9][y + 9][z + 9]);
          r2[x + 2][y + 2][z + 2] = sin(theta[x + 9][y + 9][z + 9])*cos(phi[x + 9][y + 9][z + 9]);
          r3[x + 2][y + 2][z + 2] = sin(phi[x + 9][y + 9][z + 9])*sin(theta[x + 9][y + 9][z + 9]);
        }
      }
    }
  }
  STOP_TIMER(section0,timers)
  /* End section0 */

  for (int time = time_m, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3); time <= time_M; time += 1, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3))
  {
    /* Begin section1 */
    START_TIMER(section1)
    #pragma omp parallel num_threads(nthreads)
    {
      const int tid = omp_get_thread_num();
      float (*restrict r22)[y0_blk0_size + 3][z_size + 3] __attribute__ ((aligned (64))) = (float (*)[y0_blk0_size + 3][z_size + 3]) pr22[tid];
      float (*restrict r23)[y0_blk0_size + 3][z_size + 3] __attribute__ ((aligned (64))) = (float (*)[y0_blk0_size + 3][z_size + 3]) pr23[tid];

      #pragma omp for collapse(2) schedule(dynamic,1)
      for (int x0_blk0 = x_m; x0_blk0 <= x_M; x0_blk0 += x0_blk0_size)
      {
        for (int y0_blk0 = y_m; y0_blk0 <= y_M; y0_blk0 += y0_blk0_size)
        {
          for (int x = x0_blk0 - 2, xs = 0; x <= MIN(x0_blk0 + x0_blk0_size, x_M + 1); x += 1, xs += 1)
          {
            for (int y = y0_blk0 - 2, ys = 0; y <= MIN(y0_blk0 + y0_blk0_size, y_M + 1); y += 1, ys += 1)
            {
              #pragma omp simd aligned(u,v:32)
              for (int z = z_m - 2; z <= z_M + 1; z += 1)
              {
                float r26 = 5.00000007e-2F*v[t0][x + 7][y + 7][z + 7];
                float r25 = 5.00000007e-2F*u[t0][x + 7][y + 7][z + 7];
                r22[xs][ys][z + 2] = -(r25 + 1.66666669e-2F*u[t0][x + 5][y + 7][z + 7] - 1.0e-1F*u[t0][x + 6][y + 7][z + 7] + 3.33333338e-2F*u[t0][x + 8][y + 7][z + 7])*r2[x + 2][y + 2][z + 2] - (r25 + 1.66666669e-2F*u[t0][x + 7][y + 5][z + 7] - 1.0e-1F*u[t0][x + 7][y + 6][z + 7] + 3.33333338e-2F*u[t0][x + 7][y + 8][z + 7])*r3[x + 2][y + 2][z + 2] - (r25 + 1.66666669e-2F*u[t0][x + 7][y + 7][z + 5] - 1.0e-1F*u[t0][x + 7][y + 7][z + 6] + 3.33333338e-2F*u[t0][x + 7][y + 7][z + 8])*r1[x + 2][y + 2][z + 2];
                r23[xs][ys][z + 2] = -(r26 + 1.66666669e-2F*v[t0][x + 5][y + 7][z + 7] - 1.0e-1F*v[t0][x + 6][y + 7][z + 7] + 3.33333338e-2F*v[t0][x + 8][y + 7][z + 7])*r2[x + 2][y + 2][z + 2] - (r26 + 1.66666669e-2F*v[t0][x + 7][y + 5][z + 7] - 1.0e-1F*v[t0][x + 7][y + 6][z + 7] + 3.33333338e-2F*v[t0][x + 7][y + 8][z + 7])*r3[x + 2][y + 2][z + 2] - (r26 + 1.66666669e-2F*v[t0][x + 7][y + 7][z + 5] - 1.0e-1F*v[t0][x + 7][y + 7][z + 6] + 3.33333338e-2F*v[t0][x + 7][y + 7][z + 8])*r1[x + 2][y + 2][z + 2];
              }
            }
          }
          for (int x = x0_blk0, xs = 0; x <= MIN(x0_blk0 + x0_blk0_size - 1, x_M); x += 1, xs += 1)
          {
            for (int y = y0_blk0, ys = 0; y <= MIN(y0_blk0 + y0_blk0_size - 1, y_M); y += 1, ys += 1)
            {
              #pragma omp simd aligned(damp,epsilon,u,v,vp:32)
              for (int z = z_m; z <= z_M; z += 1)
              {
                float r33 = 3.33333338e-2F*(r1[x + 1][y + 1][z]*r23[xs + 1][ys + 1][z] + r2[x][y + 1][z + 1]*r23[xs][ys + 1][z + 1] + r23[xs + 1][ys][z + 1]*r3[x + 1][y][z + 1]);
                float r32 = 1.66666669e-2F*(r1[x + 1][y + 1][z + 3]*r23[xs + 1][ys + 1][z + 3] + r2[x + 3][y + 1][z + 1]*r23[xs + 3][ys + 1][z + 1] + r23[xs + 1][ys + 3][z + 1]*r3[x + 1][y + 3][z + 1]);
                float r31 = 1.0e-1F*(-r1[x + 1][y + 1][z + 2]*r23[xs + 1][ys + 1][z + 2] - r2[x + 2][y + 1][z + 1]*r23[xs + 2][ys + 1][z + 1] - r23[xs + 1][ys + 2][z + 1]*r3[x + 1][y + 2][z + 1]);
                float r30 = 5.00000007e-2F*(r1[x + 1][y + 1][z + 1]*r23[xs + 1][ys + 1][z + 1] + r2[x + 1][y + 1][z + 1]*r23[xs + 1][ys + 1][z + 1] + r23[xs + 1][ys + 1][z + 1]*r3[x + 1][y + 1][z + 1]);
                float r29 = 1.0F/(vp[x + 8][y + 8][z + 8]*vp[x + 8][y + 8][z + 8]);
                float r28 = 1.0F/(r4*r29 + r5*damp[x + 1][y + 1][z + 1]);
                float r27 = 3.33333338e-2F*(-r1[x + 1][y + 1][z]*r22[xs + 1][ys + 1][z] - r2[x][y + 1][z + 1]*r22[xs][ys + 1][z + 1] - r22[xs + 1][ys][z + 1]*r3[x + 1][y][z + 1]) + 5.00000007e-2F*(-r1[x + 1][y + 1][z + 1]*r22[xs + 1][ys + 1][z + 1] - r2[x + 1][y + 1][z + 1]*r22[xs + 1][ys + 1][z + 1] - r22[xs + 1][ys + 1][z + 1]*r3[x + 1][y + 1][z + 1]) + 1.0e-1F*(r1[x + 1][y + 1][z + 2]*r22[xs + 1][ys + 1][z + 2] + r2[x + 2][y + 1][z + 1]*r22[xs + 2][ys + 1][z + 1] + r22[xs + 1][ys + 2][z + 1]*r3[x + 1][y + 2][z + 1]) + 1.66666669e-2F*(-r1[x + 1][y + 1][z + 3]*r22[xs + 1][ys + 1][z + 3] - r2[x + 3][y + 1][z + 1]*r22[xs + 3][ys + 1][z + 1] - r22[xs + 1][ys + 3][z + 1]*r3[x + 1][y + 3][z + 1]) + 1.11111109e-4F*(u[t0][x + 3][y + 6][z + 6] + u[t0][x + 6][y + 3][z + 6] + u[t0][x + 6][y + 6][z + 3] + u[t0][x + 6][y + 6][z + 9] + u[t0][x + 6][y + 9][z + 6] + u[t0][x + 9][y + 6][z + 6]) + 1.49999997e-3F*(-u[t0][x + 4][y + 6][z + 6] - u[t0][x + 6][y + 4][z + 6] - u[t0][x + 6][y + 6][z + 4] - u[t0][x + 6][y + 6][z + 8] - u[t0][x + 6][y + 8][z + 6] - u[t0][x + 8][y + 6][z + 6]) + 1.49999997e-2F*(u[t0][x + 5][y + 6][z + 6] + u[t0][x + 6][y + 5][z + 6] + u[t0][x + 6][y + 6][z + 5] + u[t0][x + 6][y + 6][z + 7] + u[t0][x + 6][y + 7][z + 6] + u[t0][x + 7][y + 6][z + 6]) - 8.16666648e-2F*u[t0][x + 6][y + 6][z + 6];
                u[t2][x + 6][y + 6][z + 6] = r28*(r5*damp[x + 1][y + 1][z + 1]*u[t0][x + 6][y + 6][z + 6] + r27*(2*epsilon[x + 8][y + 8][z + 8] + 1) + r29*(-r4*(-2.0F*u[t0][x + 6][y + 6][z + 6]) - r4*u[t1][x + 6][y + 6][z + 6]) + (r30 + r31 + r32 + r33)*r0[x + 2][y + 2][z + 2]);
                v[t2][x + 6][y + 6][z + 6] = r28*(r5*damp[x + 1][y + 1][z + 1]*v[t0][x + 6][y + 6][z + 6] + r27*r0[x + 2][y + 2][z + 2] + r29*(-r4*(-2.0F*v[t0][x + 6][y + 6][z + 6]) - r4*v[t1][x + 6][y + 6][z + 6]) + r30 + r31 + r32 + r33);
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
      int chunk_size = (int)(fmax(1, (1.0F/3.0F)*(p_src_M - p_src_m + 1)/nthreads_nonaffine));
      #pragma omp for collapse(1) schedule(dynamic,chunk_size)
      for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1)
      {
        float posx = -o_x + src_coords[p_src][0];
        float posy = -o_y + src_coords[p_src][1];
        float posz = -o_z + src_coords[p_src][2];
        int ii_src_0 = (int)(floor(1.0e-1F*posx));
        int ii_src_1 = (int)(floor(1.0e-1F*posy));
        int ii_src_2 = (int)(floor(1.0e-1F*posz));
        int ii_src_3 = 1 + (int)(floor(1.0e-1F*posz));
        int ii_src_4 = 1 + (int)(floor(1.0e-1F*posy));
        int ii_src_5 = 1 + (int)(floor(1.0e-1F*posx));
        float px = (float)(posx - 1.0e+1F*(int)(floor(1.0e-1F*posx)));
        float py = (float)(posy - 1.0e+1F*(int)(floor(1.0e-1F*posy)));
        float pz = (float)(posz - 1.0e+1F*(int)(floor(1.0e-1F*posz)));
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1)
        {
          float r6 = (dt*dt)*(vp[ii_src_0 + 8][ii_src_1 + 8][ii_src_2 + 8]*vp[ii_src_0 + 8][ii_src_1 + 8][ii_src_2 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py + 1.0e-2F*px*pz - 1.0e-1F*px + 1.0e-2F*py*pz - 1.0e-1F*py - 1.0e-1F*pz + 1)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_0 + 6][ii_src_1 + 6][ii_src_2 + 6] += r6;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
        {
          float r7 = (dt*dt)*(vp[ii_src_0 + 8][ii_src_1 + 8][ii_src_3 + 8]*vp[ii_src_0 + 8][ii_src_1 + 8][ii_src_3 + 8])*(1.0e-3F*px*py*pz - 1.0e-2F*px*pz - 1.0e-2F*py*pz + 1.0e-1F*pz)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_0 + 6][ii_src_1 + 6][ii_src_3 + 6] += r7;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r8 = (dt*dt)*(vp[ii_src_0 + 8][ii_src_4 + 8][ii_src_2 + 8]*vp[ii_src_0 + 8][ii_src_4 + 8][ii_src_2 + 8])*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*py*pz + 1.0e-1F*py)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_0 + 6][ii_src_4 + 6][ii_src_2 + 6] += r8;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r9 = (dt*dt)*(vp[ii_src_0 + 8][ii_src_4 + 8][ii_src_3 + 8]*vp[ii_src_0 + 8][ii_src_4 + 8][ii_src_3 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*py*pz)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_0 + 6][ii_src_4 + 6][ii_src_3 + 6] += r9;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r10 = (dt*dt)*(vp[ii_src_5 + 8][ii_src_1 + 8][ii_src_2 + 8]*vp[ii_src_5 + 8][ii_src_1 + 8][ii_src_2 + 8])*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*px*pz + 1.0e-1F*px)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_5 + 6][ii_src_1 + 6][ii_src_2 + 6] += r10;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r11 = (dt*dt)*(vp[ii_src_5 + 8][ii_src_1 + 8][ii_src_3 + 8]*vp[ii_src_5 + 8][ii_src_1 + 8][ii_src_3 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*pz)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_5 + 6][ii_src_1 + 6][ii_src_3 + 6] += r11;
        }
        if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r12 = (dt*dt)*(vp[ii_src_5 + 8][ii_src_4 + 8][ii_src_2 + 8]*vp[ii_src_5 + 8][ii_src_4 + 8][ii_src_2 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_5 + 6][ii_src_4 + 6][ii_src_2 + 6] += r12;
        }
        if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r13 = 1.0e-3F*px*py*pz*(dt*dt)*(vp[ii_src_5 + 8][ii_src_4 + 8][ii_src_3 + 8]*vp[ii_src_5 + 8][ii_src_4 + 8][ii_src_3 + 8])*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_5 + 6][ii_src_4 + 6][ii_src_3 + 6] += r13;
        }
        posx = -o_x + src_coords[p_src][0];
        posy = -o_y + src_coords[p_src][1];
        posz = -o_z + src_coords[p_src][2];
        ii_src_0 = (int)(floor(1.0e-1F*posx));
        ii_src_1 = (int)(floor(1.0e-1F*posy));
        ii_src_2 = (int)(floor(1.0e-1F*posz));
        ii_src_3 = 1 + (int)(floor(1.0e-1F*posz));
        ii_src_4 = 1 + (int)(floor(1.0e-1F*posy));
        ii_src_5 = 1 + (int)(floor(1.0e-1F*posx));
        px = (float)(posx - 1.0e+1F*(int)(floor(1.0e-1F*posx)));
        py = (float)(posy - 1.0e+1F*(int)(floor(1.0e-1F*posy)));
        pz = (float)(posz - 1.0e+1F*(int)(floor(1.0e-1F*posz)));
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1)
        {
          float r14 = (dt*dt)*(vp[ii_src_0 + 8][ii_src_1 + 8][ii_src_2 + 8]*vp[ii_src_0 + 8][ii_src_1 + 8][ii_src_2 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py + 1.0e-2F*px*pz - 1.0e-1F*px + 1.0e-2F*py*pz - 1.0e-1F*py - 1.0e-1F*pz + 1)*src[time][p_src];
          #pragma omp atomic update
          v[t2][ii_src_0 + 6][ii_src_1 + 6][ii_src_2 + 6] += r14;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
        {
          float r15 = (dt*dt)*(vp[ii_src_0 + 8][ii_src_1 + 8][ii_src_3 + 8]*vp[ii_src_0 + 8][ii_src_1 + 8][ii_src_3 + 8])*(1.0e-3F*px*py*pz - 1.0e-2F*px*pz - 1.0e-2F*py*pz + 1.0e-1F*pz)*src[time][p_src];
          #pragma omp atomic update
          v[t2][ii_src_0 + 6][ii_src_1 + 6][ii_src_3 + 6] += r15;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r16 = (dt*dt)*(vp[ii_src_0 + 8][ii_src_4 + 8][ii_src_2 + 8]*vp[ii_src_0 + 8][ii_src_4 + 8][ii_src_2 + 8])*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*py*pz + 1.0e-1F*py)*src[time][p_src];
          #pragma omp atomic update
          v[t2][ii_src_0 + 6][ii_src_4 + 6][ii_src_2 + 6] += r16;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r17 = (dt*dt)*(vp[ii_src_0 + 8][ii_src_4 + 8][ii_src_3 + 8]*vp[ii_src_0 + 8][ii_src_4 + 8][ii_src_3 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*py*pz)*src[time][p_src];
          #pragma omp atomic update
          v[t2][ii_src_0 + 6][ii_src_4 + 6][ii_src_3 + 6] += r17;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r18 = (dt*dt)*(vp[ii_src_5 + 8][ii_src_1 + 8][ii_src_2 + 8]*vp[ii_src_5 + 8][ii_src_1 + 8][ii_src_2 + 8])*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*px*pz + 1.0e-1F*px)*src[time][p_src];
          #pragma omp atomic update
          v[t2][ii_src_5 + 6][ii_src_1 + 6][ii_src_2 + 6] += r18;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r19 = (dt*dt)*(vp[ii_src_5 + 8][ii_src_1 + 8][ii_src_3 + 8]*vp[ii_src_5 + 8][ii_src_1 + 8][ii_src_3 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*pz)*src[time][p_src];
          #pragma omp atomic update
          v[t2][ii_src_5 + 6][ii_src_1 + 6][ii_src_3 + 6] += r19;
        }
        if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r20 = (dt*dt)*(vp[ii_src_5 + 8][ii_src_4 + 8][ii_src_2 + 8]*vp[ii_src_5 + 8][ii_src_4 + 8][ii_src_2 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py)*src[time][p_src];
          #pragma omp atomic update
          v[t2][ii_src_5 + 6][ii_src_4 + 6][ii_src_2 + 6] += r20;
        }
        if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r21 = 1.0e-3F*px*py*pz*(dt*dt)*(vp[ii_src_5 + 8][ii_src_4 + 8][ii_src_3 + 8]*vp[ii_src_5 + 8][ii_src_4 + 8][ii_src_3 + 8])*src[time][p_src];
          #pragma omp atomic update
          v[t2][ii_src_5 + 6][ii_src_4 + 6][ii_src_3 + 6] += r21;
        }
      }
    }
    STOP_TIMER(section2,timers)
    /* End section2 */

    /* Begin section3 */
    START_TIMER(section3)
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
        float sum = 0.0F;
        if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1 && ii_rec_2 <= z_M + 1)
        {
          sum += (u[t0][ii_rec_0 + 6][ii_rec_1 + 6][ii_rec_2 + 6] + v[t0][ii_rec_0 + 6][ii_rec_1 + 6][ii_rec_2 + 6])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py + 1.0e-2F*px*pz - 1.0e-1F*px + 1.0e-2F*py*pz - 1.0e-1F*py - 1.0e-1F*pz + 1);
        }
        if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= z_M + 1)
        {
          sum += (u[t0][ii_rec_0 + 6][ii_rec_1 + 6][ii_rec_3 + 6] + v[t0][ii_rec_0 + 6][ii_rec_1 + 6][ii_rec_3 + 6])*(1.0e-3F*px*py*pz - 1.0e-2F*px*pz - 1.0e-2F*py*pz + 1.0e-1F*pz);
        }
        if (ii_rec_0 >= x_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_2 <= z_M + 1 && ii_rec_4 <= y_M + 1)
        {
          sum += (u[t0][ii_rec_0 + 6][ii_rec_4 + 6][ii_rec_2 + 6] + v[t0][ii_rec_0 + 6][ii_rec_4 + 6][ii_rec_2 + 6])*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*py*pz + 1.0e-1F*py);
        }
        if (ii_rec_0 >= x_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_3 <= z_M + 1 && ii_rec_4 <= y_M + 1)
        {
          sum += (-1.0e-3F*px*py*pz + 1.0e-2F*py*pz)*(u[t0][ii_rec_0 + 6][ii_rec_4 + 6][ii_rec_3 + 6] + v[t0][ii_rec_0 + 6][ii_rec_4 + 6][ii_rec_3 + 6]);
        }
        if (ii_rec_1 >= y_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_2 <= z_M + 1 && ii_rec_5 <= x_M + 1)
        {
          sum += (u[t0][ii_rec_5 + 6][ii_rec_1 + 6][ii_rec_2 + 6] + v[t0][ii_rec_5 + 6][ii_rec_1 + 6][ii_rec_2 + 6])*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*px*pz + 1.0e-1F*px);
        }
        if (ii_rec_1 >= y_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= z_M + 1 && ii_rec_5 <= x_M + 1)
        {
          sum += (-1.0e-3F*px*py*pz + 1.0e-2F*px*pz)*(u[t0][ii_rec_5 + 6][ii_rec_1 + 6][ii_rec_3 + 6] + v[t0][ii_rec_5 + 6][ii_rec_1 + 6][ii_rec_3 + 6]);
        }
        if (ii_rec_2 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_2 <= z_M + 1 && ii_rec_4 <= y_M + 1 && ii_rec_5 <= x_M + 1)
        {
          sum += (-1.0e-3F*px*py*pz + 1.0e-2F*px*py)*(u[t0][ii_rec_5 + 6][ii_rec_4 + 6][ii_rec_2 + 6] + v[t0][ii_rec_5 + 6][ii_rec_4 + 6][ii_rec_2 + 6]);
        }
        if (ii_rec_3 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_3 <= z_M + 1 && ii_rec_4 <= y_M + 1 && ii_rec_5 <= x_M + 1)
        {
          sum += 1.0e-3F*px*py*pz*(u[t0][ii_rec_5 + 6][ii_rec_4 + 6][ii_rec_3 + 6] + v[t0][ii_rec_5 + 6][ii_rec_4 + 6][ii_rec_3 + 6]);
        }
        rec[time][p_rec] = sum;
      }
    }
    STOP_TIMER(section3,timers)
    /* End section3 */

    /* Begin write section */
    START_TIMER(write)

    int border_x = 10;
    int border_y = 10;
    int border_z = 10;

    // Write U and V to disk
    #pragma omp parallel for schedule(static,1) num_threads(nthreads)
    for(int i=border_x; i < u_vec->size[1]-border_x;i++)
    {
      int tid = i%nthreads;
      size_t uz_size = (u_vec->size[3]-2*border_z)*sizeof(float);
      for(int j=border_y; j < u_vec->size[2]-border_y;j++)
      {
        int ret = write(u_files[tid], &u[t0][i][j][border_z], uz_size);
        if (ret != uz_size) {
          perror("Error writing to file");
          exit(1);
        }
      }
    }

    #pragma omp parallel for schedule(static,1) num_threads(nthreads)
    for(int i=border_x; i < v_vec->size[1]-border_x;i++)
    {
      int tid = i%nthreads;
      size_t vz_size = (v_vec->size[3]-2*border_z)*sizeof(float);
      for(int j=border_y; j < v_vec->size[2]-border_y;j++)
      {
        int ret = write(v_files[tid], &v[t0][i][j][border_z], vz_size);
        if (ret != vz_size) {
          perror("Error writing to file");
          exit(1);
        }
      }
    }

    STOP_TIMER(write, iop);
    /* End write section */

  }

  /* Begin close section */
  START_TIMER(close)
  for(int i=0; i < nthreads; i++){
    close(u_files[i]);
  }

  for(int i=0; i < nthreads; i++){
    close(v_files[i]);
  }
  STOP_TIMER(close, iop)
  /* End close section */

  long int write_size = (time_M - time_m+1) * u_vec->size[1] * u_size;
  write_size += (time_M - time_m+1) * v_vec->size[1] * v_size;

  save(nthreads, timers, iop, write_size);

  #pragma omp parallel num_threads(nthreads)
  {
    const int tid = omp_get_thread_num();
    free(pr22_vec[tid]);
    free(pr23_vec[tid]);
  }
  free(pr22_vec);
  free(pr23_vec);
  free(r0_vec);
  free(r1_vec);
  free(r2_vec);
  free(r3_vec);

  free(iop);
  free(u_files);
  free(v_files);

  return 0;
}
