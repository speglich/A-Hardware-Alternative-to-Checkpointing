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
} ;

struct io_profiler
{
  double open;
  double read;
  double close;
} ;

void open_thread_files(int *files, int nthreads)
{

  for(int i=0; i < nthreads; i++)
  {
    int nvme_id = i % NDISKS;
    char name[100];

    sprintf(name, "data/nvme%d/thread_%d.data", nvme_id, i);
    printf("Reading file %s\n", name);

    if ((files[i] = open(name, O_RDONLY,
        S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)) == -1)
    {
        perror("Cannot open output file\n"); exit(1);
    }
  }

  return;
}

void save(int nthreads, struct profiler * timers, struct io_profiler * iop, long int read_size)
{
  printf(">>>>>>>>>>>>>> REVERSE <<<<<<<<<<<<<<<<<\n");

  printf("Threads %d\n", nthreads);
  printf("Disks %d\n", NDISKS);

  printf("[REV] Section0 %.2lf s\n", timers->section0);
  printf("[REV] Section1 %.2lf s\n", timers->section1);

  printf("[IO] Open %.2lf s\n", iop->open);
  printf("[IO] Read %.2lf s\n", iop->read);
  printf("[IO] Close %.2lf s\n", iop->close);

  char name[100];
  sprintf(name, "rev_disks_%d_threads_%d.csv", NDISKS, nthreads);

  FILE *fpt;
  fpt = fopen(name, "a");

  fprintf(fpt,"Disks, Threads, Bytes, [REV] Section0, [REV] Section1, [IO] Open, [IO] Read, [IO] Close\n");

  fprintf(fpt,"%d, %d, %ld, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf\n", NDISKS, nthreads, read_size,
        timers->section0, timers->section1, iop->open, iop->read, iop->close);

  fclose(fpt);
}

int Kernel(struct dataobj *restrict damp_vec, const float dt, struct dataobj *restrict image_vec, const float o_x, const float o_y, struct dataobj *restrict residual_vec, struct dataobj *restrict residual_coords_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, struct dataobj *restrict vp_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int p_residual_M, const int p_residual_m, const int time_M, const int time_m, const int nthreads, const int nthreads_nonaffine, struct profiler * timers)
{
  float (*restrict damp)[damp_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]]) damp_vec->data;
  float (*restrict image)[image_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[image_vec->size[1]]) image_vec->data;
  float (*restrict residual)[residual_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[residual_vec->size[1]]) residual_vec->data;
  float (*restrict residual_coords)[residual_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[residual_coords_vec->size[1]]) residual_coords_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]]) u_vec->data;
  float (*restrict v)[v_vec->size[1]][v_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[v_vec->size[1]][v_vec->size[2]]) v_vec->data;
  float (*restrict vp)[vp_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]]) vp_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  float r0 = 1.0F/(dt*dt);
  float r1 = 1.0F/dt;

  struct io_profiler * iop = malloc(sizeof(struct io_profiler));

  iop->open = 0;
  iop->read = 0;
  iop->close = 0;

  /* Begin open files Section */
  START_TIMER(open)

  int *files = malloc(nthreads * sizeof(int));
  if (files == NULL)
  {
    printf("Error to alloc\n");
    exit(1);
  }

  open_thread_files(files, nthreads);

  int *counters = malloc(nthreads * sizeof(int));

  if (counters == NULL)
  {
    printf("Error to alloc\n");
    exit(1);
  }

  for(int i=0; i < nthreads; i++){
    counters[i] = 1;
  }

  STOP_TIMER(open, iop)
  /* End open files section */

  size_t u_size = u_vec->size[2]*sizeof(float);

  for (int time = time_M, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3); time >= time_m; time -= 1, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3))
  {

    /* Begin read section */
    START_TIMER(read)
    #pragma omp parallel for schedule(static,1) num_threads(nthreads)
    for(int i= u_vec->size[1]-1;i>=0;i--)
    {
      int tid = i%nthreads;

      off_t offset = counters[tid] * u_size;
      lseek(files[tid], -1 * offset, SEEK_END);

      int ret = read(files[tid], u[time][i], u_size);

      if (ret != u_size) {
          printf("%d", ret);
          perror("Cannot open output file");
          exit(1);
      }

      counters[tid]++;
    }
    STOP_TIMER(read, iop);
    /* End read section */

    /* Begin section0 */
    START_TIMER(section0)
    #pragma omp parallel num_threads(nthreads)
    {
      #pragma omp for collapse(1) schedule(dynamic,1)
      for (int x = x_m; x <= x_M; x += 1)
      {
        #pragma omp simd aligned(damp,image,u,v,vp:64)
        for (int y = y_m; y <= y_M; y += 1)
        {
          float r6 = 1.0F/(vp[x + 2][y + 2]*vp[x + 2][y + 2]);
          v[t1][x + 4][y + 4] = (r1*damp[x + 1][y + 1]*v[t0][x + 4][y + 4] + r6*(-r0*(-2.0F*v[t0][x + 4][y + 4]) - r0*v[t2][x + 4][y + 4]) + 1.48148152e-3F*(-v[t0][x + 2][y + 4] - v[t0][x + 4][y + 2] - v[t0][x + 4][y + 6] - v[t0][x + 6][y + 4]) + 2.37037043e-2F*(v[t0][x + 3][y + 4] + v[t0][x + 4][y + 3] + v[t0][x + 4][y + 5] + v[t0][x + 5][y + 4]) - 8.8888891e-2F*v[t0][x + 4][y + 4])/(r0*r6 + r1*damp[x + 1][y + 1]);
          image[x + 1][y + 1] = image[x + 1][y + 1] - u[time][x + 4][y + 4]*v[t0][x + 4][y + 4];
        }
      }
    }
    STOP_TIMER(section0,timers)
    /* End section0 */

    /* Begin section1 */
    START_TIMER(section1)
    #pragma omp parallel num_threads(nthreads_nonaffine)
    {
      int chunk_size = (int)(fmax(1, (1.0F/3.0F)*(p_residual_M - p_residual_m + 1)/nthreads_nonaffine));
      #pragma omp for collapse(1) schedule(dynamic,chunk_size)
      for (int p_residual = p_residual_m; p_residual <= p_residual_M; p_residual += 1)
      {
        float posx = -o_x + residual_coords[p_residual][0];
        float posy = -o_y + residual_coords[p_residual][1];
        int ii_residual_0 = (int)(floor(1.33333e-1F*posx));
        int ii_residual_1 = (int)(floor(1.33333e-1F*posy));
        int ii_residual_2 = 1 + (int)(floor(1.33333e-1F*posy));
        int ii_residual_3 = 1 + (int)(floor(1.33333e-1F*posx));
        float px = (float)(posx - 7.5F*(int)(floor(1.33333e-1F*posx)));
        float py = (float)(posy - 7.5F*(int)(floor(1.33333e-1F*posy)));
        if (ii_residual_0 >= x_m - 1 && ii_residual_1 >= y_m - 1 && ii_residual_0 <= x_M + 1 && ii_residual_1 <= y_M + 1)
        {
          float r2 = 9.54919784643327e-1F*(vp[ii_residual_0 + 2][ii_residual_1 + 2]*vp[ii_residual_0 + 2][ii_residual_1 + 2])*(1.77778e-2F*px*py - 1.33333e-1F*px - 1.33333e-1F*py + 1)*residual[time][p_residual];
          #pragma omp atomic update
          v[t1][ii_residual_0 + 4][ii_residual_1 + 4] += r2;
        }
        if (ii_residual_0 >= x_m - 1 && ii_residual_2 >= y_m - 1 && ii_residual_0 <= x_M + 1 && ii_residual_2 <= y_M + 1)
        {
          float r3 = 9.54919784643327e-1F*(vp[ii_residual_0 + 2][ii_residual_2 + 2]*vp[ii_residual_0 + 2][ii_residual_2 + 2])*(-1.77778e-2F*px*py + 1.33333e-1F*py)*residual[time][p_residual];
          #pragma omp atomic update
          v[t1][ii_residual_0 + 4][ii_residual_2 + 4] += r3;
        }
        if (ii_residual_1 >= y_m - 1 && ii_residual_3 >= x_m - 1 && ii_residual_1 <= y_M + 1 && ii_residual_3 <= x_M + 1)
        {
          float r4 = 9.54919784643327e-1F*(vp[ii_residual_3 + 2][ii_residual_1 + 2]*vp[ii_residual_3 + 2][ii_residual_1 + 2])*(-1.77778e-2F*px*py + 1.33333e-1F*px)*residual[time][p_residual];
          #pragma omp atomic update
          v[t1][ii_residual_3 + 4][ii_residual_1 + 4] += r4;
        }
        if (ii_residual_2 >= y_m - 1 && ii_residual_3 >= x_m - 1 && ii_residual_2 <= y_M + 1 && ii_residual_3 <= x_M + 1)
        {
          float r5 = 1.69763539167411e-2F*px*py*(vp[ii_residual_3 + 2][ii_residual_2 + 2]*vp[ii_residual_3 + 2][ii_residual_2 + 2])*residual[time][p_residual];
          #pragma omp atomic update
          v[t1][ii_residual_3 + 4][ii_residual_2 + 4] += r5;
        }
      }
    }
    STOP_TIMER(section1,timers)
    /* End section1 */
  }

  /* Begin close section */
  START_TIMER(close)
  for(int i=0; i < nthreads; i++){
    close(files[i]);
  }
  STOP_TIMER(close, iop)
  /* End close section */

  long int read_size = (time_M - time_m+1) * u_vec->size[1] * u_size;

  save(nthreads, timers, iop, read_size);

  free(iop);
  free(files);
  free(counters);

  return 0;
}
