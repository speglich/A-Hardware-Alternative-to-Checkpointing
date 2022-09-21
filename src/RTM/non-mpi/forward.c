#define _POSIX_C_SOURCE 200809L
#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#ifndef NDISKS
#define NDISKS 8
#endif

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#include "omp.h"
#include "stdio.h"
#include "unistd.h"
#include "fcntl.h"

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

struct io_profiler
{
  double open;
  double write;
  double close;
} ;

void open_thread_files(int *files, int nthreads)
{

  for(int i=0; i < nthreads; i++)
  {
    int nvme_id = i % NDISKS;
    char name[100];

    sprintf(name, "data/nvme%d/thread_%d.data", nvme_id, i);
    printf("Creating file %s\n", name);

    if ((files[i] = open(name, O_WRONLY | O_CREAT | O_TRUNC,
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
  fpt = fopen(name, "a");

  fprintf(fpt,"Disks, Threads, Bytes, [FWD] Section0, [FWD] Section1, [FWD] Section2, [IO] Open, [IO] Write, [IO] Close\n");

  fprintf(fpt,"%d, %d, %ld, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf\n", NDISKS, nthreads, write_size,
        timers->section0, timers->section1, timers->section2, iop->open, iop->write, iop->close);

  fclose(fpt);
}

int Forward(struct dataobj *restrict damp_vec, const float dt, const float o_x, const float o_y, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int p_rec_M, const int p_rec_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, const int nthreads, const int nthreads_nonaffine, struct profiler * timers)
{
  float (*restrict damp)[damp_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]]) damp_vec->data;
  float (*restrict rec)[rec_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_vec->size[1]]) rec_vec->data;
  float (*restrict rec_coords)[rec_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_coords_vec->size[1]]) rec_coords_vec->data;
  float (*restrict src)[src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_vec->size[1]]) src_vec->data;
  float (*restrict src_coords)[src_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]]) u_vec->data;
  float (*restrict vp)[vp_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]]) vp_vec->data;

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

  size_t u_size = u_vec->size[2]*sizeof(float);

  for (int time = time_m; time <= time_M; time += 1)
  {
    /* Begin section0 */
    START_TIMER(section0)
    #pragma omp parallel num_threads(nthreads)
    {
      #pragma omp for collapse(1) schedule(dynamic,1)
      for (int x = x_m; x <= x_M; x += 1)
      {
        #pragma omp simd aligned(damp,u,vp:64)
        for (int y = y_m; y <= y_M; y += 1)
        {
          float r6 = 1.0F/(vp[x + 2][y + 2]*vp[x + 2][y + 2]);
          u[time + 1][x + 4][y + 4] = (r1*damp[x + 1][y + 1]*u[time][x + 4][y + 4] + r6*(-r0*(-2.0F*u[time][x + 4][y + 4]) - r0*u[time - 1][x + 4][y + 4]) + 1.48148152e-3F*(-u[time][x + 2][y + 4] - u[time][x + 4][y + 2] - u[time][x + 4][y + 6] - u[time][x + 6][y + 4]) + 2.37037043e-2F*(u[time][x + 3][y + 4] + u[time][x + 4][y + 3] + u[time][x + 4][y + 5] + u[time][x + 5][y + 4]) - 8.8888891e-2F*u[time][x + 4][y + 4])/(r0*r6 + r1*damp[x + 1][y + 1]);
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
        int ii_src_0 = (int)(floor(1.33333e-1F*posx));
        int ii_src_1 = (int)(floor(1.33333e-1F*posy));
        int ii_src_2 = 1 + (int)(floor(1.33333e-1F*posy));
        int ii_src_3 = 1 + (int)(floor(1.33333e-1F*posx));
        float px = (float)(posx - 7.5F*(int)(floor(1.33333e-1F*posx)));
        float py = (float)(posy - 7.5F*(int)(floor(1.33333e-1F*posy)));
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1)
        {
          float r2 = (dt*dt)*(vp[ii_src_0 + 2][ii_src_1 + 2]*vp[ii_src_0 + 2][ii_src_1 + 2])*(1.77778e-2F*px*py - 1.33333e-1F*px - 1.33333e-1F*py + 1)*src[time][p_src];
          #pragma omp atomic update
          u[time + 1][ii_src_0 + 4][ii_src_1 + 4] += r2;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_2 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= y_M + 1)
        {
          float r3 = (dt*dt)*(vp[ii_src_0 + 2][ii_src_2 + 2]*vp[ii_src_0 + 2][ii_src_2 + 2])*(-1.77778e-2F*px*py + 1.33333e-1F*py)*src[time][p_src];
          #pragma omp atomic update
          u[time + 1][ii_src_0 + 4][ii_src_2 + 4] += r3;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_3 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= x_M + 1)
        {
          float r4 = (dt*dt)*(vp[ii_src_3 + 2][ii_src_1 + 2]*vp[ii_src_3 + 2][ii_src_1 + 2])*(-1.77778e-2F*px*py + 1.33333e-1F*px)*src[time][p_src];
          #pragma omp atomic update
          u[time + 1][ii_src_3 + 4][ii_src_1 + 4] += r4;
        }
        if (ii_src_2 >= y_m - 1 && ii_src_3 >= x_m - 1 && ii_src_2 <= y_M + 1 && ii_src_3 <= x_M + 1)
        {
          float r5 = 1.77778e-2F*px*py*(dt*dt)*(vp[ii_src_3 + 2][ii_src_2 + 2]*vp[ii_src_3 + 2][ii_src_2 + 2])*src[time][p_src];
          #pragma omp atomic update
          u[time + 1][ii_src_3 + 4][ii_src_2 + 4] += r5;
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
        int ii_rec_0 = (int)(floor(1.33333e-1F*posx));
        int ii_rec_1 = (int)(floor(1.33333e-1F*posy));
        int ii_rec_2 = 1 + (int)(floor(1.33333e-1F*posy));
        int ii_rec_3 = 1 + (int)(floor(1.33333e-1F*posx));
        float px = (float)(posx - 7.5F*(int)(floor(1.33333e-1F*posx)));
        float py = (float)(posy - 7.5F*(int)(floor(1.33333e-1F*posy)));
        float sum = 0.0F;
        if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1)
        {
          sum += (1.77778e-2F*px*py - 1.33333e-1F*px - 1.33333e-1F*py + 1)*u[time][ii_rec_0 + 4][ii_rec_1 + 4];
        }
        if (ii_rec_0 >= x_m - 1 && ii_rec_2 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_2 <= y_M + 1)
        {
          sum += (-1.77778e-2F*px*py + 1.33333e-1F*py)*u[time][ii_rec_0 + 4][ii_rec_2 + 4];
        }
        if (ii_rec_1 >= y_m - 1 && ii_rec_3 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= x_M + 1)
        {
          sum += (-1.77778e-2F*px*py + 1.33333e-1F*px)*u[time][ii_rec_3 + 4][ii_rec_1 + 4];
        }
        if (ii_rec_2 >= y_m - 1 && ii_rec_3 >= x_m - 1 && ii_rec_2 <= y_M + 1 && ii_rec_3 <= x_M + 1)
        {
          sum += 1.77778e-2F*px*py*u[time][ii_rec_3 + 4][ii_rec_2 + 4];
        }
        rec[time][p_rec] = sum;
      }
    }
    STOP_TIMER(section2,timers)
    /* End section2 */

    /* Begin write section */
    START_TIMER(write)
    #pragma omp parallel for schedule(static,1) num_threads(nthreads)
    for(int i=0; i < u_vec->size[1];i++)
    {
      int tid = i%nthreads;
      int ret = write(files[tid], u[time][i], u_size);
      if (ret != u_size) {
          perror("Cannot open output file");
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
