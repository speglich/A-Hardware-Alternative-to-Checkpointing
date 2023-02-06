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

#ifdef CACHE
#define OPEN_FLAGS O_RDONLY
#else
#define OPEN_FLAGS O_DIRECT | O_RDONLY
#endif

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

    if ((files[i] = open(name, OPEN_FLAGS,
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
  printf("[REV] Section2 %.2lf s\n", timers->section2);

  printf("[IO] Open %.2lf s\n", iop->open);
  printf("[IO] Read %.2lf s\n", iop->read);
  printf("[IO] Close %.2lf s\n", iop->close);

  char name[100];
  sprintf(name, "rev_disks_%d_threads_%d.csv", NDISKS, nthreads);

  FILE *fpt;
  fpt = fopen(name, "w");

  fprintf(fpt,"Disks, Threads, Bytes, [REV] Section0, [REV] Section1, [REV] Section2, [IO] Open, [IO] Read, [IO] Close\n");

  fprintf(fpt,"%d, %d, %ld, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf\n", NDISKS, nthreads, read_size,
        timers->section0, timers->section1, timers->section2, iop->open, iop->read, iop->close);

  fclose(fpt);
}

int Gradient(struct dataobj *restrict damp_vec, const float dt, struct dataobj *restrict grad_vec, const float o_x, const float o_y, const float o_z, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, struct dataobj *restrict vp_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int p_rec_M, const int p_rec_m, const int time_M, const int time_m, const int x0_blk0_size, const int x1_blk0_size, const int y0_blk0_size, const int y1_blk0_size, const int nthreads, const int nthreads_nonaffine, struct profiler * timers)
{
  float (*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]][damp_vec->size[2]]) damp_vec->data;
  float (*restrict grad)[grad_vec->size[1]][grad_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[grad_vec->size[1]][grad_vec->size[2]]) grad_vec->data;
  float (*restrict rec)[rec_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_vec->size[1]]) rec_vec->data;
  float (*restrict rec_coords)[rec_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_coords_vec->size[1]]) rec_coords_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;
  float (*restrict v)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]]) v_vec->data;
  float (*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]][vp_vec->size[2]]) vp_vec->data;

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

  size_t u_size = u_vec->size[2]*u_vec->size[3]*sizeof(float);

  for (int time = time_M, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3); time >= time_m; time -= 1, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3))
  {
    /* Begin section0 */
    START_TIMER(section0)
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
              #pragma omp simd aligned(damp,v,vp:64)
              for (int z = z_m; z <= z_M; z += 1)
              {
                float r10 = 1.0F/(vp[x + 6][y + 6][z + 6]*vp[x + 6][y + 6][z + 6]);
                v[t1][x + 6][y + 6][z + 6] = (r1*damp[x + 1][y + 1][z + 1]*v[t0][x + 6][y + 6][z + 6] + r10*(-r0*(-2.0F*v[t0][x + 6][y + 6][z + 6]) - r0*v[t2][x + 6][y + 6][z + 6]) + 1.77777773e-5F*(v[t0][x + 3][y + 6][z + 6] + v[t0][x + 6][y + 3][z + 6] + v[t0][x + 6][y + 6][z + 3] + v[t0][x + 6][y + 6][z + 9] + v[t0][x + 6][y + 9][z + 6] + v[t0][x + 9][y + 6][z + 6]) + 2.39999994e-4F*(-v[t0][x + 4][y + 6][z + 6] - v[t0][x + 6][y + 4][z + 6] - v[t0][x + 6][y + 6][z + 4] - v[t0][x + 6][y + 6][z + 8] - v[t0][x + 6][y + 8][z + 6] - v[t0][x + 8][y + 6][z + 6]) + 2.39999994e-3F*(v[t0][x + 5][y + 6][z + 6] + v[t0][x + 6][y + 5][z + 6] + v[t0][x + 6][y + 6][z + 5] + v[t0][x + 6][y + 6][z + 7] + v[t0][x + 6][y + 7][z + 6] + v[t0][x + 7][y + 6][z + 6]) - 1.30666663e-2F*v[t0][x + 6][y + 6][z + 6])/(r0*r10 + r1*damp[x + 1][y + 1][z + 1]);
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
        if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1 && ii_rec_2 <= z_M + 1)
        {
          float r2 = (dt*dt)*(vp[ii_rec_0 + 6][ii_rec_1 + 6][ii_rec_2 + 6]*vp[ii_rec_0 + 6][ii_rec_1 + 6][ii_rec_2 + 6])*(-6.4e-5F*px*py*pz + 1.6e-3F*px*py + 1.6e-3F*px*pz - 4.0e-2F*px + 1.6e-3F*py*pz - 4.0e-2F*py - 4.0e-2F*pz + 1)*rec[time][p_rec];
          #pragma omp atomic update
          v[t1][ii_rec_0 + 6][ii_rec_1 + 6][ii_rec_2 + 6] += r2;
        }
        if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= z_M + 1)
        {
          float r3 = (dt*dt)*(vp[ii_rec_0 + 6][ii_rec_1 + 6][ii_rec_3 + 6]*vp[ii_rec_0 + 6][ii_rec_1 + 6][ii_rec_3 + 6])*(6.4e-5F*px*py*pz - 1.6e-3F*px*pz - 1.6e-3F*py*pz + 4.0e-2F*pz)*rec[time][p_rec];
          #pragma omp atomic update
          v[t1][ii_rec_0 + 6][ii_rec_1 + 6][ii_rec_3 + 6] += r3;
        }
        if (ii_rec_0 >= x_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_2 <= z_M + 1 && ii_rec_4 <= y_M + 1)
        {
          float r4 = (dt*dt)*(vp[ii_rec_0 + 6][ii_rec_4 + 6][ii_rec_2 + 6]*vp[ii_rec_0 + 6][ii_rec_4 + 6][ii_rec_2 + 6])*(6.4e-5F*px*py*pz - 1.6e-3F*px*py - 1.6e-3F*py*pz + 4.0e-2F*py)*rec[time][p_rec];
          #pragma omp atomic update
          v[t1][ii_rec_0 + 6][ii_rec_4 + 6][ii_rec_2 + 6] += r4;
        }
        if (ii_rec_0 >= x_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_3 <= z_M + 1 && ii_rec_4 <= y_M + 1)
        {
          float r5 = (dt*dt)*(vp[ii_rec_0 + 6][ii_rec_4 + 6][ii_rec_3 + 6]*vp[ii_rec_0 + 6][ii_rec_4 + 6][ii_rec_3 + 6])*(-6.4e-5F*px*py*pz + 1.6e-3F*py*pz)*rec[time][p_rec];
          #pragma omp atomic update
          v[t1][ii_rec_0 + 6][ii_rec_4 + 6][ii_rec_3 + 6] += r5;
        }
        if (ii_rec_1 >= y_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_2 <= z_M + 1 && ii_rec_5 <= x_M + 1)
        {
          float r6 = (dt*dt)*(vp[ii_rec_5 + 6][ii_rec_1 + 6][ii_rec_2 + 6]*vp[ii_rec_5 + 6][ii_rec_1 + 6][ii_rec_2 + 6])*(6.4e-5F*px*py*pz - 1.6e-3F*px*py - 1.6e-3F*px*pz + 4.0e-2F*px)*rec[time][p_rec];
          #pragma omp atomic update
          v[t1][ii_rec_5 + 6][ii_rec_1 + 6][ii_rec_2 + 6] += r6;
        }
        if (ii_rec_1 >= y_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= z_M + 1 && ii_rec_5 <= x_M + 1)
        {
          float r7 = (dt*dt)*(vp[ii_rec_5 + 6][ii_rec_1 + 6][ii_rec_3 + 6]*vp[ii_rec_5 + 6][ii_rec_1 + 6][ii_rec_3 + 6])*(-6.4e-5F*px*py*pz + 1.6e-3F*px*pz)*rec[time][p_rec];
          #pragma omp atomic update
          v[t1][ii_rec_5 + 6][ii_rec_1 + 6][ii_rec_3 + 6] += r7;
        }
        if (ii_rec_2 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_2 <= z_M + 1 && ii_rec_4 <= y_M + 1 && ii_rec_5 <= x_M + 1)
        {
          float r8 = (dt*dt)*(vp[ii_rec_5 + 6][ii_rec_4 + 6][ii_rec_2 + 6]*vp[ii_rec_5 + 6][ii_rec_4 + 6][ii_rec_2 + 6])*(-6.4e-5F*px*py*pz + 1.6e-3F*px*py)*rec[time][p_rec];
          #pragma omp atomic update
          v[t1][ii_rec_5 + 6][ii_rec_4 + 6][ii_rec_2 + 6] += r8;
        }
        if (ii_rec_3 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_3 <= z_M + 1 && ii_rec_4 <= y_M + 1 && ii_rec_5 <= x_M + 1)
        {
          float r9 = 6.4e-5F*px*py*pz*(dt*dt)*(vp[ii_rec_5 + 6][ii_rec_4 + 6][ii_rec_3 + 6]*vp[ii_rec_5 + 6][ii_rec_4 + 6][ii_rec_3 + 6])*rec[time][p_rec];
          #pragma omp atomic update
          v[t1][ii_rec_5 + 6][ii_rec_4 + 6][ii_rec_3 + 6] += r9;
        }
      }
    }
    STOP_TIMER(section1,timers)
    /* End section1 */

    /* Begin read section */
    START_TIMER(read)
    #pragma omp parallel for schedule(static,1) num_threads(nthreads)
    for(int i= u_vec->size[1]-1;i>=0;i--)
    {
      int tid = i%nthreads;

      off_t offset = counters[tid] * u_size;
      lseek(files[tid], -1 * offset, SEEK_END);

      int ret = read(files[tid], u[t0][i], u_size);

      if (ret != u_size) {
          printf("%d", ret);
          perror("Cannot open output file");
          exit(1);
      }

      counters[tid]++;
    }
    STOP_TIMER(read, iop);
    /* End read section */

    /* Begin section2 */
    START_TIMER(section2)
    #pragma omp parallel num_threads(nthreads)
    {
      #pragma omp for collapse(2) schedule(static,1)
      for (int x1_blk0 = x_m; x1_blk0 <= x_M; x1_blk0 += x1_blk0_size)
      {
        for (int y1_blk0 = y_m; y1_blk0 <= y_M; y1_blk0 += y1_blk0_size)
        {
          for (int x = x1_blk0; x <= MIN(x1_blk0 + x1_blk0_size - 1, x_M); x += 1)
          {
            for (int y = y1_blk0; y <= MIN(y1_blk0 + y1_blk0_size - 1, y_M); y += 1)
            {
              #pragma omp simd aligned(grad,u,v:64)
              for (int z = z_m; z <= z_M; z += 1)
              {
                grad[x + 1][y + 1][z + 1] += -(r0*(-2.0F*v[t0][x + 6][y + 6][z + 6]) + r0*v[t1][x + 6][y + 6][z + 6] + r0*v[t2][x + 6][y + 6][z + 6])*u[t0][x + 6][y + 6][z + 6];
              }
            }
          }
        }
      }
    }
    STOP_TIMER(section2,timers)
    /* End section2 */
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
