#define _GNU_SOURCE
#include "fcntl.h"

#define _POSIX_C_SOURCE 200809L
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#ifndef NDISKS
#define NDISKS 4
#endif

#ifndef DPS
#define DPS 4
#endif

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#include "mpi.h"
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

struct neighborhood
{
  int lll, llc, llr, lcl, lcc, lcr, lrl, lrc, lrr;
  int cll, clc, clr, ccl, ccc, ccr, crl, crc, crr;
  int rll, rlc, rlr, rcl, rcc, rcr, rrl, rrc, rrr;
} ;

struct profiler
{
  double section0;
  double section1;
  double section2;
} ;

struct io_profiler
{
  double open;
  double write;
  double close;
} ;

static void sendrecvtxyz(struct dataobj *restrict a0_vec, const int buf_x_size, const int buf_y_size, const int buf_z_size, int ogtime, int ogx, int ogy, int ogz, int ostime, int osx, int osy, int osz, int fromrank, int torank, MPI_Comm comm, const int nthreads);
static void gathertxyz(float *restrict buf0_vec, const int buf_x_size, const int buf_y_size, const int buf_z_size, struct dataobj *restrict a0_vec, int otime, int ox, int oy, int oz, const int nthreads);
static void scattertxyz(float *restrict buf1_vec, const int buf_x_size, const int buf_y_size, const int buf_z_size, struct dataobj *restrict a0_vec, int otime, int ox, int oy, int oz, const int nthreads);
static void haloupdate0(struct dataobj *restrict a0_vec, MPI_Comm comm, struct neighborhood * nb, int otime, const int nthreads);

void open_thread_files(int *files, int nthreads)
{

  for(int i=0; i < nthreads; i++)
  {

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int socket = (myrank % 2) * DPS; // Disks per Socket
    int nvme_id = socket + i % NDISKS;
    char name[100];

    sprintf(name, "data/nvme%d/socket_%d_thread_%d.data", nvme_id, myrank, i);
    if ((files[i] = open(name, O_DIRECT | O_WRONLY | O_CREAT ,
        S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)) == -1)
    {
      char error[140];
      sprintf(error, "Cannot open output %s!\n", name);
      perror(error); exit(1);
    } else {
      printf("Creating file %s\n", name);
    }
  }

}

void save(int nthreads, struct profiler * timers, struct io_profiler * iop, long int write_size)
{

  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  if (myrank != 0)
  {
    return;
  }

  printf(">>>>>>>>>>>>>> MPI FORWARD <<<<<<<<<<<<<<<<<\n");

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

int Forward(struct dataobj *restrict damp_vec, const float dt, const float o_x, const float o_y, const float o_z, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int p_rec_M, const int p_rec_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, const int x0_blk0_size, const int y0_blk0_size, MPI_Comm comm, struct neighborhood * nb, const int nthreads, const int nthreads_nonaffine, struct profiler * timers)
{
  float (*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]][damp_vec->size[2]]) damp_vec->data;
  float (*restrict rec)[rec_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_vec->size[1]]) rec_vec->data;
  float (*restrict rec_coords)[rec_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_coords_vec->size[1]]) rec_coords_vec->data;
  float (*restrict src)[src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_vec->size[1]]) src_vec->data;
  float (*restrict src_coords)[src_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;
  float (*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]][vp_vec->size[2]]) vp_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  float r0 = 1.0F/(dt*dt);
  float r1 = 1.0F/dt;

  struct io_profiler * iop = malloc(sizeof(struct io_profiler));

  iop->open = 0;
  iop->write = 0;
  iop->close = 0;

  /* Begin Open Files Section */
  START_TIMER(open)

  int *files = malloc(nthreads * sizeof(int));
  if (files == NULL)
  {
      printf("Error to alloc\n");
      exit(1);
  }
  open_thread_files(files, nthreads);

  STOP_TIMER(open, iop)
  /* End Open Files Section */

  size_t u_size = u_vec->size[2]*u_vec->size[3]*sizeof(float);

  for (int time = time_m, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3); time <= time_M; time += 1, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3))
  {
    /* Begin section0 */
    START_TIMER(section0)
    haloupdate0(u_vec,comm,nb,t0,nthreads);
    #pragma omp parallel num_threads(nthreads)
    {
      #pragma omp for collapse(2) schedule(dynamic,1)
      for (int x0_blk0 = x_m; x0_blk0 <= x_M; x0_blk0 += x0_blk0_size)
      {
        for (int y0_blk0 = y_m; y0_blk0 <= y_M; y0_blk0 += y0_blk0_size)
        {
          for (int x = x0_blk0; x <= MIN(x0_blk0 + x0_blk0_size - 1, x_M); x += 1)
          {
            for (int y = y0_blk0; y <= MIN(y0_blk0 + y0_blk0_size - 1, y_M); y += 1)
            {
              #pragma omp simd aligned(damp,u,vp:64)
              for (int z = z_m; z <= z_M; z += 1)
              {
                float r10 = 1.0F/(vp[x + 6][y + 6][z + 6]*vp[x + 6][y + 6][z + 6]);
                u[t2][x + 6][y + 6][z + 6] = (r1*damp[x + 1][y + 1][z + 1]*u[t0][x + 6][y + 6][z + 6] + r10*(-r0*(-2.0F*u[t0][x + 6][y + 6][z + 6]) - r0*u[t1][x + 6][y + 6][z + 6]) + 1.77777773e-5F*(u[t0][x + 3][y + 6][z + 6] + u[t0][x + 6][y + 3][z + 6] + u[t0][x + 6][y + 6][z + 3] + u[t0][x + 6][y + 6][z + 9] + u[t0][x + 6][y + 9][z + 6] + u[t0][x + 9][y + 6][z + 6]) + 2.39999994e-4F*(-u[t0][x + 4][y + 6][z + 6] - u[t0][x + 6][y + 4][z + 6] - u[t0][x + 6][y + 6][z + 4] - u[t0][x + 6][y + 6][z + 8] - u[t0][x + 6][y + 8][z + 6] - u[t0][x + 8][y + 6][z + 6]) + 2.39999994e-3F*(u[t0][x + 5][y + 6][z + 6] + u[t0][x + 6][y + 5][z + 6] + u[t0][x + 6][y + 6][z + 5] + u[t0][x + 6][y + 6][z + 7] + u[t0][x + 6][y + 7][z + 6] + u[t0][x + 7][y + 6][z + 6]) - 1.30666663e-2F*u[t0][x + 6][y + 6][z + 6])/(r0*r10 + r1*damp[x + 1][y + 1][z + 1]);
              }
            }
          }
        }
      }
    }
    STOP_TIMER(section0,timers)
    /* End section0 */

    /* Begin section1 */
    START_TIMER(section1)
    #pragma omp parallel num_threads(nthreads_nonaffine)
    {
      int chunk_size = (int)(fmax(1, (1.0F/3.0F)*(p_src_M - p_src_m + 1)/nthreads_nonaffine));
      #pragma omp for collapse(1) schedule(dynamic,chunk_size)
      for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1)
      {
        float posx = -o_x + src_coords[p_src][0];
        float posy = -o_y + src_coords[p_src][1];
        float posz = -o_z + src_coords[p_src][2];
        int ii_src_0 = (int)(floor(4.0e-2F*posx));
        int ii_src_1 = (int)(floor(4.0e-2F*posy));
        int ii_src_2 = (int)(floor(4.0e-2F*posz));
        int ii_src_3 = 1 + (int)(floor(4.0e-2F*posz));
        int ii_src_4 = 1 + (int)(floor(4.0e-2F*posy));
        int ii_src_5 = 1 + (int)(floor(4.0e-2F*posx));
        float px = (float)(posx - 2.5e+1F*(int)(floor(4.0e-2F*posx)));
        float py = (float)(posy - 2.5e+1F*(int)(floor(4.0e-2F*posy)));
        float pz = (float)(posz - 2.5e+1F*(int)(floor(4.0e-2F*posz)));
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1)
        {
          float r2 = (dt*dt)*(vp[ii_src_0 + 6][ii_src_1 + 6][ii_src_2 + 6]*vp[ii_src_0 + 6][ii_src_1 + 6][ii_src_2 + 6])*(-6.4e-5F*px*py*pz + 1.6e-3F*px*py + 1.6e-3F*px*pz - 4.0e-2F*px + 1.6e-3F*py*pz - 4.0e-2F*py - 4.0e-2F*pz + 1)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_0 + 6][ii_src_1 + 6][ii_src_2 + 6] += r2;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
        {
          float r3 = (dt*dt)*(vp[ii_src_0 + 6][ii_src_1 + 6][ii_src_3 + 6]*vp[ii_src_0 + 6][ii_src_1 + 6][ii_src_3 + 6])*(6.4e-5F*px*py*pz - 1.6e-3F*px*pz - 1.6e-3F*py*pz + 4.0e-2F*pz)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_0 + 6][ii_src_1 + 6][ii_src_3 + 6] += r3;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r4 = (dt*dt)*(vp[ii_src_0 + 6][ii_src_4 + 6][ii_src_2 + 6]*vp[ii_src_0 + 6][ii_src_4 + 6][ii_src_2 + 6])*(6.4e-5F*px*py*pz - 1.6e-3F*px*py - 1.6e-3F*py*pz + 4.0e-2F*py)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_0 + 6][ii_src_4 + 6][ii_src_2 + 6] += r4;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r5 = (dt*dt)*(vp[ii_src_0 + 6][ii_src_4 + 6][ii_src_3 + 6]*vp[ii_src_0 + 6][ii_src_4 + 6][ii_src_3 + 6])*(-6.4e-5F*px*py*pz + 1.6e-3F*py*pz)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_0 + 6][ii_src_4 + 6][ii_src_3 + 6] += r5;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r6 = (dt*dt)*(vp[ii_src_5 + 6][ii_src_1 + 6][ii_src_2 + 6]*vp[ii_src_5 + 6][ii_src_1 + 6][ii_src_2 + 6])*(6.4e-5F*px*py*pz - 1.6e-3F*px*py - 1.6e-3F*px*pz + 4.0e-2F*px)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_5 + 6][ii_src_1 + 6][ii_src_2 + 6] += r6;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r7 = (dt*dt)*(vp[ii_src_5 + 6][ii_src_1 + 6][ii_src_3 + 6]*vp[ii_src_5 + 6][ii_src_1 + 6][ii_src_3 + 6])*(-6.4e-5F*px*py*pz + 1.6e-3F*px*pz)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_5 + 6][ii_src_1 + 6][ii_src_3 + 6] += r7;
        }
        if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r8 = (dt*dt)*(vp[ii_src_5 + 6][ii_src_4 + 6][ii_src_2 + 6]*vp[ii_src_5 + 6][ii_src_4 + 6][ii_src_2 + 6])*(-6.4e-5F*px*py*pz + 1.6e-3F*px*py)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_5 + 6][ii_src_4 + 6][ii_src_2 + 6] += r8;
        }
        if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r9 = 6.4e-5F*px*py*pz*(dt*dt)*(vp[ii_src_5 + 6][ii_src_4 + 6][ii_src_3 + 6]*vp[ii_src_5 + 6][ii_src_4 + 6][ii_src_3 + 6])*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_5 + 6][ii_src_4 + 6][ii_src_3 + 6] += r9;
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
        int ii_rec_0 = (int)(floor(4.0e-2F*posx));
        int ii_rec_1 = (int)(floor(4.0e-2F*posy));
        int ii_rec_2 = (int)(floor(4.0e-2F*posz));
        int ii_rec_3 = 1 + (int)(floor(4.0e-2F*posz));
        int ii_rec_4 = 1 + (int)(floor(4.0e-2F*posy));
        int ii_rec_5 = 1 + (int)(floor(4.0e-2F*posx));
        float px = (float)(posx - 2.5e+1F*(int)(floor(4.0e-2F*posx)));
        float py = (float)(posy - 2.5e+1F*(int)(floor(4.0e-2F*posy)));
        float pz = (float)(posz - 2.5e+1F*(int)(floor(4.0e-2F*posz)));
        float sum = 0.0F;
        if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1 && ii_rec_2 <= z_M + 1)
        {
          sum += (-6.4e-5F*px*py*pz + 1.6e-3F*px*py + 1.6e-3F*px*pz - 4.0e-2F*px + 1.6e-3F*py*pz - 4.0e-2F*py - 4.0e-2F*pz + 1)*u[t0][ii_rec_0 + 6][ii_rec_1 + 6][ii_rec_2 + 6];
        }
        if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= z_M + 1)
        {
          sum += (6.4e-5F*px*py*pz - 1.6e-3F*px*pz - 1.6e-3F*py*pz + 4.0e-2F*pz)*u[t0][ii_rec_0 + 6][ii_rec_1 + 6][ii_rec_3 + 6];
        }
        if (ii_rec_0 >= x_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_2 <= z_M + 1 && ii_rec_4 <= y_M + 1)
        {
          sum += (6.4e-5F*px*py*pz - 1.6e-3F*px*py - 1.6e-3F*py*pz + 4.0e-2F*py)*u[t0][ii_rec_0 + 6][ii_rec_4 + 6][ii_rec_2 + 6];
        }
        if (ii_rec_0 >= x_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_3 <= z_M + 1 && ii_rec_4 <= y_M + 1)
        {
          sum += (-6.4e-5F*px*py*pz + 1.6e-3F*py*pz)*u[t0][ii_rec_0 + 6][ii_rec_4 + 6][ii_rec_3 + 6];
        }
        if (ii_rec_1 >= y_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_2 <= z_M + 1 && ii_rec_5 <= x_M + 1)
        {
          sum += (6.4e-5F*px*py*pz - 1.6e-3F*px*py - 1.6e-3F*px*pz + 4.0e-2F*px)*u[t0][ii_rec_5 + 6][ii_rec_1 + 6][ii_rec_2 + 6];
        }
        if (ii_rec_1 >= y_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= z_M + 1 && ii_rec_5 <= x_M + 1)
        {
          sum += (-6.4e-5F*px*py*pz + 1.6e-3F*px*pz)*u[t0][ii_rec_5 + 6][ii_rec_1 + 6][ii_rec_3 + 6];
        }
        if (ii_rec_2 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_2 <= z_M + 1 && ii_rec_4 <= y_M + 1 && ii_rec_5 <= x_M + 1)
        {
          sum += (-6.4e-5F*px*py*pz + 1.6e-3F*px*py)*u[t0][ii_rec_5 + 6][ii_rec_4 + 6][ii_rec_2 + 6];
        }
        if (ii_rec_3 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_3 <= z_M + 1 && ii_rec_4 <= y_M + 1 && ii_rec_5 <= x_M + 1)
        {
          sum += 6.4e-5F*px*py*pz*u[t0][ii_rec_5 + 6][ii_rec_4 + 6][ii_rec_3 + 6];
        }
        rec[time][p_rec] = sum;
      }
    }
    STOP_TIMER(section2,timers)
    /* End section2 */

    /* Begin write section */
    START_TIMER(write)
    #pragma omp parallel for schedule(static,1)
    for(int i=0; i < u_vec->size[1];i++)
    {
      int tid = i%nthreads;
      int ret = write(files[tid], u[t0][i], u_size);
      if (ret != u_size) {
          perror("Write size mismatch with u_size");
          exit(1);
      }
    }
    STOP_TIMER(write, iop);
    /* End write section */
  }

  /* Begin close section */
  START_TIMER(close)
  for(int i=0; i < nthreads; i++){
    close(files[i]);
  }
  STOP_TIMER(close, iop)
  /* End close section */

  long int write_size = (time_M - time_m+1) * u_vec->size[1] * u_size;
  save(nthreads, timers, iop, write_size);

  free(iop);
  free(files);

  return 0;
}

static void sendrecvtxyz(struct dataobj *restrict a0_vec, const int buf_x_size, const int buf_y_size, const int buf_z_size, int ogtime, int ogx, int ogy, int ogz, int ostime, int osx, int osy, int osz, int fromrank, int torank, MPI_Comm comm, const int nthreads)
{
  float *bufg0_vec;
  posix_memalign((void**)(&bufg0_vec),64,buf_x_size*buf_y_size*buf_z_size*sizeof(float));
  float *bufs0_vec;
  posix_memalign((void**)(&bufs0_vec),64,buf_x_size*buf_y_size*buf_z_size*sizeof(float));

  MPI_Request rrecv;
  MPI_Request rsend;

  MPI_Irecv(bufs0_vec,buf_x_size*buf_y_size*buf_z_size,MPI_FLOAT,fromrank,13,comm,&(rrecv));
  if (torank != MPI_PROC_NULL)
  {
    gathertxyz(bufg0_vec,buf_x_size,buf_y_size,buf_z_size,a0_vec,ogtime,ogx,ogy,ogz,nthreads);
  }
  MPI_Isend(bufg0_vec,buf_x_size*buf_y_size*buf_z_size,MPI_FLOAT,torank,13,comm,&(rsend));
  MPI_Wait(&(rsend),MPI_STATUS_IGNORE);
  MPI_Wait(&(rrecv),MPI_STATUS_IGNORE);
  if (fromrank != MPI_PROC_NULL)
  {
    scattertxyz(bufs0_vec,buf_x_size,buf_y_size,buf_z_size,a0_vec,ostime,osx,osy,osz,nthreads);
  }

  free(bufg0_vec);
  free(bufs0_vec);
}

static void gathertxyz(float *restrict buf0_vec, const int buf_x_size, const int buf_y_size, const int buf_z_size, struct dataobj *restrict a0_vec, int otime, int ox, int oy, int oz, const int nthreads)
{
  float (*restrict a0)[a0_vec->size[1]][a0_vec->size[2]][a0_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[a0_vec->size[1]][a0_vec->size[2]][a0_vec->size[3]]) a0_vec->data;
  float (*restrict buf0)[buf_y_size][buf_z_size] __attribute__ ((aligned (64))) = (float (*)[buf_y_size][buf_z_size]) buf0_vec;

  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(2) schedule(static,1)
    for (int x = 0; x <= buf_x_size - 1; x += 1)
    {
      for (int y = 0; y <= buf_y_size - 1; y += 1)
      {
        #pragma omp simd aligned(a0:64)
        for (int z = 0; z <= buf_z_size - 1; z += 1)
        {
          buf0[x][y][z] = a0[otime][x + ox][y + oy][z + oz];
        }
      }
    }
  }
}

static void scattertxyz(float *restrict buf1_vec, const int buf_x_size, const int buf_y_size, const int buf_z_size, struct dataobj *restrict a0_vec, int otime, int ox, int oy, int oz, const int nthreads)
{
  float (*restrict a0)[a0_vec->size[1]][a0_vec->size[2]][a0_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[a0_vec->size[1]][a0_vec->size[2]][a0_vec->size[3]]) a0_vec->data;
  float (*restrict buf1)[buf_y_size][buf_z_size] __attribute__ ((aligned (64))) = (float (*)[buf_y_size][buf_z_size]) buf1_vec;

  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(2) schedule(static,1)
    for (int x = 0; x <= buf_x_size - 1; x += 1)
    {
      for (int y = 0; y <= buf_y_size - 1; y += 1)
      {
        #pragma omp simd aligned(a0:64)
        for (int z = 0; z <= buf_z_size - 1; z += 1)
        {
          a0[otime][x + ox][y + oy][z + oz] = buf1[x][y][z];
        }
      }
    }
  }
}

static void haloupdate0(struct dataobj *restrict a0_vec, MPI_Comm comm, struct neighborhood * nb, int otime, const int nthreads)
{
  sendrecvtxyz(a0_vec,a0_vec->hsize[3],a0_vec->npsize[2],a0_vec->npsize[3],otime,a0_vec->oofs[2],a0_vec->hofs[4],a0_vec->hofs[6],otime,a0_vec->hofs[3],a0_vec->hofs[4],a0_vec->hofs[6],nb->rcc,nb->lcc,comm,nthreads);
  sendrecvtxyz(a0_vec,a0_vec->hsize[2],a0_vec->npsize[2],a0_vec->npsize[3],otime,a0_vec->oofs[3],a0_vec->hofs[4],a0_vec->hofs[6],otime,a0_vec->hofs[2],a0_vec->hofs[4],a0_vec->hofs[6],nb->lcc,nb->rcc,comm,nthreads);
  sendrecvtxyz(a0_vec,a0_vec->npsize[1],a0_vec->hsize[5],a0_vec->npsize[3],otime,a0_vec->hofs[2],a0_vec->oofs[4],a0_vec->hofs[6],otime,a0_vec->hofs[2],a0_vec->hofs[5],a0_vec->hofs[6],nb->crc,nb->clc,comm,nthreads);
  sendrecvtxyz(a0_vec,a0_vec->npsize[1],a0_vec->hsize[4],a0_vec->npsize[3],otime,a0_vec->hofs[2],a0_vec->oofs[5],a0_vec->hofs[6],otime,a0_vec->hofs[2],a0_vec->hofs[4],a0_vec->hofs[6],nb->clc,nb->crc,comm,nthreads);
  sendrecvtxyz(a0_vec,a0_vec->npsize[1],a0_vec->npsize[2],a0_vec->hsize[7],otime,a0_vec->hofs[2],a0_vec->hofs[4],a0_vec->oofs[6],otime,a0_vec->hofs[2],a0_vec->hofs[4],a0_vec->hofs[7],nb->ccr,nb->ccl,comm,nthreads);
  sendrecvtxyz(a0_vec,a0_vec->npsize[1],a0_vec->npsize[2],a0_vec->hsize[6],otime,a0_vec->hofs[2],a0_vec->hofs[4],a0_vec->oofs[7],otime,a0_vec->hofs[2],a0_vec->hofs[4],a0_vec->hofs[6],nb->ccl,nb->ccr,comm,nthreads);
}
