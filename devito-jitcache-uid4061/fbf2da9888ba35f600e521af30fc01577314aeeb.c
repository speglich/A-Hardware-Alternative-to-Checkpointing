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
#include "pthread.h"
#include "semaphore.h"
#include "stdio.h"
#include "unistd.h"

struct dataobj
{
  void *restrict data;
  unsigned long * size;
  unsigned long * npsize;
  unsigned long * dsize;
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


struct writter_struct {
    int begin;
    int end;
    FILE* fdes;
    void* vec;
} ;

sem_t mutex;
int to_write = 0;

void *Writter(void* arguments){

  struct writter_struct *args = (struct writter_struct *) arguments;

  struct dataobj *restrict u_vec = (struct dataobj *restrict) args->vec;

  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;

  size_t u_size = u_vec->size[1]*u_vec->size[2]*u_vec->size[3];

  FILE* f = args->fdes;

  int begin = args->begin;
  int end = args->end + 1;

  int pos_write = begin;
  int len = 0;

  while(pos_write != end) {

    sem_wait(&mutex);
    len = to_write;
    sem_post(&mutex);

    int end = pos_write + len;

    for(int i = pos_write; i < end; i++) {
      fwrite(u[i%3], sizeof(float), u_size, f);
      pos_write++;
      printf("Thread write %d\n", i);
    }

    sem_wait(&mutex);
    to_write -= len;
    sem_post(&mutex);

  }
  //printf("Thread escreveu %d...\n", begin);

  return 0;
}

int Forward(struct dataobj *restrict damp_vec, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const float dt, const float o_x, const float o_y, const float o_z, const int p_rec_M, const int p_rec_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, const int x0_blk0_size, const int y0_blk0_size, const int nthreads, const int nthreads_nonaffine, struct profiler * timers)
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

  float r8 = 1.0F/(dt*dt);
  float r9 = 1.0F/dt;

  FILE *file = fopen("example.data", "wb");
  struct writter_struct args;
  args.vec = (void *) u_vec;
  args.fdes = file;
  args.begin = time_m;
  args.end = time_M;

  sem_init(&mutex, 0, 1);

  pthread_t thread;
  int iret1 = pthread_create(&thread, NULL, Writter, (void *) &args);

  for (int time = time_m, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3); time <= time_M; time += 1, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3))
  {
    int busy = 1;
    while(busy)
    {
      if(to_write < 2) {
        sem_wait(&mutex);
        to_write++;
        sem_post(&mutex);
        busy = 0;
        printf("Already to write %d\n", time);
      } else {
        printf("Writter Busy...\n");
        sleep(1);
      }
    }

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
              #pragma omp simd aligned(damp,u,vp:64)
              for (int z = z_m; z <= z_M; z += 1)
              {
                float r10 = 1.0F/(vp[x + 6][y + 6][z + 6]*vp[x + 6][y + 6][z + 6]);
                u[t2][x + 6][y + 6][z + 6] = (r9*damp[x + 1][y + 1][z + 1]*u[t0][x + 6][y + 6][z + 6] + r10*(-r8*(-2.0F*u[t0][x + 6][y + 6][z + 6]) - r8*u[t1][x + 6][y + 6][z + 6]) + 1.77777773e-5F*(u[t0][x + 3][y + 6][z + 6] + u[t0][x + 6][y + 3][z + 6] + u[t0][x + 6][y + 6][z + 3] + u[t0][x + 6][y + 6][z + 9] + u[t0][x + 6][y + 9][z + 6] + u[t0][x + 9][y + 6][z + 6]) + 2.39999994e-4F*(-u[t0][x + 4][y + 6][z + 6] - u[t0][x + 6][y + 4][z + 6] - u[t0][x + 6][y + 6][z + 4] - u[t0][x + 6][y + 6][z + 8] - u[t0][x + 6][y + 8][z + 6] - u[t0][x + 8][y + 6][z + 6]) + 2.39999994e-3F*(u[t0][x + 5][y + 6][z + 6] + u[t0][x + 6][y + 5][z + 6] + u[t0][x + 6][y + 6][z + 5] + u[t0][x + 6][y + 6][z + 7] + u[t0][x + 6][y + 7][z + 6] + u[t0][x + 7][y + 6][z + 6]) - 1.30666663e-2F*u[t0][x + 6][y + 6][z + 6])/(r8*r10 + r9*damp[x + 1][y + 1][z + 1]);
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
          float r0 = (dt*dt)*(vp[ii_src_0 + 6][ii_src_1 + 6][ii_src_2 + 6]*vp[ii_src_0 + 6][ii_src_1 + 6][ii_src_2 + 6])*(-6.4e-5F*px*py*pz + 1.6e-3F*px*py + 1.6e-3F*px*pz - 4.0e-2F*px + 1.6e-3F*py*pz - 4.0e-2F*py - 4.0e-2F*pz + 1)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_0 + 6][ii_src_1 + 6][ii_src_2 + 6] += r0;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
        {
          float r1 = (dt*dt)*(vp[ii_src_0 + 6][ii_src_1 + 6][ii_src_3 + 6]*vp[ii_src_0 + 6][ii_src_1 + 6][ii_src_3 + 6])*(6.4e-5F*px*py*pz - 1.6e-3F*px*pz - 1.6e-3F*py*pz + 4.0e-2F*pz)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_0 + 6][ii_src_1 + 6][ii_src_3 + 6] += r1;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r2 = (dt*dt)*(vp[ii_src_0 + 6][ii_src_4 + 6][ii_src_2 + 6]*vp[ii_src_0 + 6][ii_src_4 + 6][ii_src_2 + 6])*(6.4e-5F*px*py*pz - 1.6e-3F*px*py - 1.6e-3F*py*pz + 4.0e-2F*py)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_0 + 6][ii_src_4 + 6][ii_src_2 + 6] += r2;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r3 = (dt*dt)*(vp[ii_src_0 + 6][ii_src_4 + 6][ii_src_3 + 6]*vp[ii_src_0 + 6][ii_src_4 + 6][ii_src_3 + 6])*(-6.4e-5F*px*py*pz + 1.6e-3F*py*pz)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_0 + 6][ii_src_4 + 6][ii_src_3 + 6] += r3;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r4 = (dt*dt)*(vp[ii_src_5 + 6][ii_src_1 + 6][ii_src_2 + 6]*vp[ii_src_5 + 6][ii_src_1 + 6][ii_src_2 + 6])*(6.4e-5F*px*py*pz - 1.6e-3F*px*py - 1.6e-3F*px*pz + 4.0e-2F*px)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_5 + 6][ii_src_1 + 6][ii_src_2 + 6] += r4;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r5 = (dt*dt)*(vp[ii_src_5 + 6][ii_src_1 + 6][ii_src_3 + 6]*vp[ii_src_5 + 6][ii_src_1 + 6][ii_src_3 + 6])*(-6.4e-5F*px*py*pz + 1.6e-3F*px*pz)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_5 + 6][ii_src_1 + 6][ii_src_3 + 6] += r5;
        }
        if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r6 = (dt*dt)*(vp[ii_src_5 + 6][ii_src_4 + 6][ii_src_2 + 6]*vp[ii_src_5 + 6][ii_src_4 + 6][ii_src_2 + 6])*(-6.4e-5F*px*py*pz + 1.6e-3F*px*py)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_5 + 6][ii_src_4 + 6][ii_src_2 + 6] += r6;
        }
        if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r7 = 6.4e-5F*px*py*pz*(dt*dt)*(vp[ii_src_5 + 6][ii_src_4 + 6][ii_src_3 + 6]*vp[ii_src_5 + 6][ii_src_4 + 6][ii_src_3 + 6])*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_5 + 6][ii_src_4 + 6][ii_src_3 + 6] += r7;
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
  }
  pthread_join(thread, NULL);
  fclose(file);
  return 0;
}
/* Backdoor edit at Thu Jun 23 10:11:55 2022*/
/* Backdoor edit at Thu Jun 23 10:17:19 2022*/
/* Backdoor edit at Thu Jun 23 10:17:53 2022*/
/* Backdoor edit at Thu Jun 23 10:35:24 2022*/
/* Backdoor edit at Thu Jun 23 10:36:32 2022*/
/* Backdoor edit at Thu Jun 23 11:08:32 2022*/
/* Backdoor edit at Thu Jun 23 11:11:45 2022*/
/* Backdoor edit at Thu Jun 23 11:12:40 2022*/
/* Backdoor edit at Thu Jun 23 11:22:53 2022*/
/* Backdoor edit at Thu Jun 23 11:46:58 2022*/
/* Backdoor edit at Thu Jun 23 12:10:28 2022*/
/* Backdoor edit at Thu Jun 23 12:18:03 2022*/
/* Backdoor edit at Thu Jun 23 14:52:32 2022*/
/* Backdoor edit at Thu Jun 23 15:02:23 2022*/
/* Backdoor edit at Thu Jun 23 15:56:31 2022*/
/* Backdoor edit at Thu Jun 23 15:58:52 2022*/
/* Backdoor edit at Thu Jun 23 16:52:49 2022*/
/* Backdoor edit at Thu Jun 23 18:45:14 2022*/ 
