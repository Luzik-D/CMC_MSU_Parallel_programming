/* Include benchmark-specific header. */
#include "gemver.h"
#include "mpi.h"
#include <stdio.h>

#define MIN(x, y) ((x) > (y) ? (y) : (x))


double bench_t_start, bench_t_end;

int using_proc; // num procs used in program
int active_proc[64]; // num active procs


static
double rtclock()
{
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0)
      printf ("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start()
{
  bench_t_start = rtclock ();
}

void bench_timer_stop()
{
  bench_t_end = rtclock ();
}

void bench_timer_print()
{
  printf ("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}


/// Main calc function
static
void kernel_gemver(int proc_group,
     int n,
     int size,
     float alpha,
     float beta,
     float A[ size][n],
     float u1[ size],
     float v1[ n],
     float u2[ size],
     float v2[ n],
     float w[ size],
     float x[ n],
     float y[ size],
     float z[ n]);


/// Handler for spare proc
/// Spare proc waiting for information from main proc
/// If main proc send 0 --> error didn't cause
/// If main proc send 0 not --> replace broken proc
static
void spare_proc_process(int k,
                        int m);


static
void init_array (int n,
   int size, // size of data for proc
   int start, // start of data for proc
   int end, // end of data for proc
   float *alpha,
   float *beta,
   float A[ n][n],
   float u1[ n],
   float v1[ n],
   float u2[ n],
   float v2[ n],
   float w[ n],
   float x[ n],
   float y[ n],
   float z[ n])
{
  int i, j;
  *alpha = 1.5;
  *beta = 1.2;

  float fn = (float)n;

  for (i = 0; i < n; i++)
    {
      u1[i] = i;
      u2[i] = ((i+1)/fn)/2.0;
      v1[i] = ((i+1)/fn)/4.0;
      v2[i] = ((i+1)/fn)/6.0;
      y[i] = ((i+1)/fn)/8.0;
      z[i] = ((i+1)/fn)/9.0;
      x[i] = 0.0;
      w[i] = 0.0;
      for (j = 0; j < n; j++)
        A[i][j] = (float) (i*j % n) / n;
    }
}

static
void print_array(int n,
   float w[ n])
{
  int i;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "w");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (stderr, "\n");
    fprintf (stderr, "%0.2f ", w[i]);
  }
  fprintf(stderr, "\nend   dump: %s\n", "w");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

/// ------------------------ MAIN ----------------------------- ///
int num_workers, proc_rank;

int main(int argc, char** argv)
{ 
  int n = N;

  // init MPI parallel working
  int err_code = MPI_Init(&argc, &argv);
  if(err_code) {
    printf("MPI_Init start error\n");
    MPI_Abort(MPI_COMM_WORLD, err_code);
    return err_code;
  }

  // get cur proc 
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  // get all workers num
  MPI_Comm_size(MPI_COMM_WORLD, &num_workers);

  // set error environment
  char err_str[MPI_MAX_ERROR_STRING];
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  using_proc = num_workers - 1; // one proc for replacement
  int k = n / using_proc;
  int m = n % using_proc;

  // set active procs
  for(int i = 0; i < using_proc; i++) {
    active_proc[i] = i;
  }

  // put spare proc to handler
  if(proc_rank == using_proc) {
    spare_proc_process(k, m);
  }

  /* -------------------------- main calcs ------------------------------- */
  MPI_Request req[1];
  MPI_Status status[1];
  for(int i = 0; i < using_proc; i++) {
    if(proc_rank == i) {
      // get params for proc
      int start = MIN(i, m) + i * k;
      int size = k + (i < m);
      int end = start + size;

      // aloccate data for proc
      float alpha;
      float beta;
      float (*A)[size][n]; A = (float(*)[size][n])malloc ((size) * (n) * sizeof(float));
      float (*u1)[size]; u1 = (float(*)[size])malloc ((size) * sizeof(float));
      float (*v1)[n]; v1 = (float(*)[n])malloc ((n) * sizeof(float));
      float (*u2)[size]; u2 = (float(*)[size])malloc ((size) * sizeof(float));
      float (*v2)[n]; v2 = (float(*)[n])malloc ((n) * sizeof(float));
      float (*w)[size]; w = (float(*)[size])malloc ((size) * sizeof(float));
      float (*x)[n]; x = (float(*)[n])malloc ((n) * sizeof(float));
      float (*y)[size]; y = (float(*)[size])malloc ((size) * sizeof(float));
      float (*z)[n]; z = (float(*)[n])malloc ((n) * sizeof(float));

      // init data for proc
      init_array(n, 
                 size,
                 start, 
                 end, 
                 &alpha, 
                 &beta,
                 *A,
                 *u1,
                 *v1,
                 *u2,
                 *v2,
                 *w,
                 *x,
                 *y,
                 *z);


      // start time counter
      double time_1;
      if (proc_rank == 0) {
          bench_timer_start();
      }

      // call main calc function
      kernel_gemver(0, 
                    n, 
                    size, 
                    alpha, 
                    beta,
                    *A,
                     *u1,
                     *v1,
                     *u2,
                     *v2,
                     *w,
                     *x,
                     *y,
                     *z);


      /*------- Gather data from procs------ */
      /*------------ Main proc --------------*/
      if(proc_rank == 0) {
        int size = k + (0 < m);
  
        // allocate memory
        float (*cur)[size];
        cur = (float(*)[size])malloc ((size) * sizeof(float));
  
        // gather information from procs
        for(int j = 0; j < using_proc; j++) {
          int proc_size = k + (j < m);
          printf("Waiting message from process %d\n", active_proc[j]);
        
          // get proc info status
          int info_status = MPI_Recv(*cur, size, MPI_DOUBLE, active_proc[j], 13, MPI_COMM_WORLD, &status[0]);
          if(!info_status) {
            // receive data, no error
            printf("Get message from process %d\n", active_proc[j]);
          }
          else {
            // error in proc
            printf("Process %d has fault\n", active_proc[j]);

            // send message to activate spare proc
            MPI_Send(&active_proc[j], 1, MPI_INT, using_proc, 13, MPI_COMM_WORLD);

            // set cur proc to spare proc
            active_proc[j] = using_proc;

            // reset last proc
            j--;
            continue;
          }
        }

        // send spare proc 0 (no errors)
        int send_spare = 0;
        MPI_Send(&send_spare, 1, MPI_INT, using_proc, 13, MPI_COMM_WORLD);

        // stop time counter and print result time
        bench_timer_stop();
        bench_timer_print();
      }

      /*--------- Calc proc--------*/
      else{
        // send data to main proc
        MPI_Send(w, size, MPI_DOUBLE, 0, 13, MPI_COMM_WORLD);
      }

      // free allocated data
      free((void*)A);
      free((void*)u1);
      free((void*)v1);
      free((void*)u2);
      free((void*)v2);
      free((void*)w);
      free((void*)x);
      free((void*)y);
      free((void*)z);
      break;
    }
  }

  MPI_Finalize();

  return 0;
}




  /*float alpha;
  float beta;
  float (*A)[n][n]; A = (float(*)[n][n])malloc ((n) * (n) * sizeof(float));
  float (*u1)[n]; u1 = (float(*)[n])malloc ((n) * sizeof(float));
  float (*v1)[n]; v1 = (float(*)[n])malloc ((n) * sizeof(float));
  float (*u2)[n]; u2 = (float(*)[n])malloc ((n) * sizeof(float));
  float (*v2)[n]; v2 = (float(*)[n])malloc ((n) * sizeof(float));
  float (*w)[n]; w = (float(*)[n])malloc ((n) * sizeof(float));
  float (*x)[n]; x = (float(*)[n])malloc ((n) * sizeof(float));
  float (*y)[n]; y = (float(*)[n])malloc ((n) * sizeof(float));
  float (*z)[n]; z = (float(*)[n])malloc ((n) * sizeof(float));

  init_array (n,
       size,
       start,
       end, 
       &alpha, 
       &beta,
       *A,
       *u1,
       *v1,
       *u2,
       *v2,
       *w,
       *x,
       *y,
       *z);

  bench_timer_start();

  kernel_gemver (n, alpha, beta,
   *A,
   *u1,
   *v1,
   *u2,
   *v2,
   *w,
   *x,
   *y,
   *z);

  bench_timer_stop();
  bench_timer_print();

  if (argc > 42 && ! strcmp(argv[0], "")) print_array(n, *w);

  free((void*)A);
  free((void*)u1);
  free((void*)v1);
  free((void*)u2);
  free((void*)v2);
  free((void*)w);
  free((void*)x);
  free((void*)y);
  free((void*)z);
 */



/// Main calc function
/*static
void kernel_gemver(int n,
     float alpha,
     float beta,
     float A[ n][n],
     float u1[ n],
     float v1[ n],
     float u2[ n],
     float v2[ n],
     float w[ n],
     float x[ n],
     float y[ n],
     float z[ n])
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];

  for (i = 0; i < n; i++)
    x[i] = x[i] + z[i];

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      w[i] = w[i] + alpha * A[i][j] * x[j];

}*/
