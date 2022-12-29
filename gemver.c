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
     double alpha,
     double beta,
     double A[ size][n],
     double u1[ size],
     double v1[ n],
     double u2[ size],
     double v2[ n],
     double w[ size],
     double x[ n],
     double y[ size],
     double z[ n]);


/// Handler for spare proc
/// Spare proc waiting for information from main proc
/// If main proc send 0 --> error didn't cause
/// If main proc send 0 not --> replace broken proc
static
void spare_proc_process(int k,
                        int m);

/// Initialize array data
static
void init_array (int n,
   int size,
   int start,
   int end,
   double *alpha,
   double *beta,
   double A[ n][n],
   double u1[ n],
   double v1[ n],
   double u2[ n],
   double v2[ n],
   double w[ n],
   double x[ n],
   double y[ n],
   double z[ n]);

static
void print_array(int n,
   double w[ n])
{
  int i;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "w");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (stderr, "\n");
    fprintf (stderr, "%0.2lf ", w[i]);
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
      double alpha;
      double beta;
      double (*A)[size][n]; A = (double(*)[size][n])malloc ((size) * (n) * sizeof(double));
      double (*u1)[size]; u1 = (double(*)[size])malloc ((size) * sizeof(double));
      double (*v1)[n]; v1 = (double(*)[n])malloc ((n) * sizeof(double));
      double (*u2)[size]; u2 = (double(*)[size])malloc ((size) * sizeof(double));
      double (*v2)[n]; v2 = (double(*)[n])malloc ((n) * sizeof(double));
      double (*w)[size]; w = (double(*)[size])malloc ((size) * sizeof(double));
      double (*x)[n]; x = (double(*)[n])malloc ((n) * sizeof(double));
      double (*y)[size]; y = (double(*)[size])malloc ((size) * sizeof(double));
      double (*z)[n]; z = (double(*)[n])malloc ((n) * sizeof(double));

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
        double (*cur)[size];
        cur = (double(*)[size])malloc ((size) * sizeof(double));
  
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


static
void init_array (int n,
   int size,
   int start,
   int end,
   double *alpha,
   double *beta,
   double A[ n][n],
   double u1[ n],
   double v1[ n],
   double u2[ n],
   double v2[ n],
   double w[ n],
   double x[ n],
   double y[ n],
   double z[ n])
{
  int i, j;
  *alpha = 1.5;
  *beta = 1.2;

  double fn = (double)n;

  for (i = start; i < end; i++)
    {
      u1[i - start] = i;
      u2[i - start] = ((i+1)/fn)/2.0;
      v1[i - start] = ((i+1)/fn)/4.0;
      v2[i - start] = ((i+1)/fn)/6.0;
      y[i - start] = ((i+1)/fn)/8.0;
      z[i - start] = ((i+1)/fn)/9.0;
      x[i - start] = 0.0;
      w[i - start] = 0.0;
      for (j = 0; j < n; j++)
        A[i - start][j] = (double) (i*j % n) / n;
    }
}

/// Create and set breakpoint
static
void set_breakpoint(int proc_group,
     int n,
     int size,
     double alpha,
     double beta,
     double A[ size][n],
     double u1[ size],
     double v1[ n],
     double u2[ size],
     double v2[ n],
     double w[ size],
     double x[ n],
     double y[ size],
     double z[ n])
{
  int proc_rank; // broken proc
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  // create and open file with breakpoint data
  char file_name[64];
  snprintf(file_name, sizeof(file_name), "_process_%d_breakpoint.txt", proc_rank);
  FILE *filename = fopen(file_name, "w");

  // write data to file
  fprintf(filename, "%d\n", proc_group);
  fprintf(filename, "%d\n", n);
  fprintf(filename, "%d\n", size);
  fprintf(filename, "%0.2lf\n", alpha);
  fprintf(filename, "%0.2lf\n", beta);

  for (int i = 0; i < size; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(filename, "%0.2lf ", A[i][j]);
        }
        fprintf(filename, "\n");
  }

  for (int i = 0; i < size; i++) {
      fprintf(filename, "%0.2lf ", u1[i]);
  }
  fprintf(filename, "\n");

  for (int i = 0; i < n; i++) {
      fprintf(filename, "%0.2lf ", v1[i]);
  }
  fprintf(filename, "\n");

  for (int i = 0; i < size; i++) {
      fprintf(filename, "%0.2lf ", u2[i]);
  }
  fprintf(filename, "\n");

  for (int i = 0; i < n; i++) {
      fprintf(filename, "%0.2lf ", v2[i]);
  }
  fprintf(filename, "\n");

  for (int i = 0; i < size; i++) {
      fprintf(filename, "%0.2lf ", w[i]);
  }
  fprintf(filename, "\n");

  for (int i = 0; i < n; i++) {
      fprintf(filename, "%0.2lf ", x[i]);
  }
  fprintf(filename, "\n");

  for (int i = 0; i < size; i++) {
      fprintf(filename, "%0.2lf ",y[i]);
  }
  fprintf(filename, "\n");

  for (int i = 0; i < n; i++) {
      fprintf(filename, "%0.2lf ", z[i]);
  }
  fprintf(filename, "\n");

  fclose(filename);
}


/// Handler for spare proc
/// Spare proc waiting for information from main proc
/// If main proc send 0 --> error didn't cause
/// If main proc send 0 not --> replace broken proc
static
void spare_proc_process(int k,
                        int m)
{
  MPI_Status status[1]; // for MPI funcs calls

  int proc; // this func is only for sparse proc and its value no interesting for us

  // receive message from main proc
  MPI_Recv(&proc, 1, MPI_INT, 0, 13, MPI_COMM_WORLD, status);

  if(proc != 0) {
    // ! Main proc send error code !
    printf("One of procs was broken\n");

    // open file from last break point of broken proc
    char file_name[64];
    snprintf(file_name, sizeof(file_name), "process_%d_breakpoint.txt", proc);
    FILE *filename = fopen(file_name, "r");

    // create variables and fill data from break point
    int group;
    int n;
    int size;
    double alpha;
    double beta;

    /*----------------------- FILL DATA PART START --------------------------*/
    fscanf(filename, "%d", &group);
    fscanf(filename, "%d", &n);
    fscanf(filename, "%d", &size);
    fscanf(filename, "%lf", &alpha);
    fscanf(filename, "%lf", &beta);


    double (*A)[size][n];
    A = (double(*)[size][n])malloc ((size) * (n) * sizeof(double));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < n; j++) {
            fscanf(filename, "%lf", &((*A)[i][j]));
        }
    }

    double (*u1)[size];
    u1 = (double(*)[size])malloc ((size) * sizeof(double));
    for (int i = 0; i < size; i++) {
        fscanf(filename, "%lf", &((*u1)[i]));
    }

    double (*v1)[n];
    v1 = (double(*)[n])malloc ((n) * sizeof(double));
    for (int i = 0; i < n; i++) {
        fscanf(filename, "%lf", &((*v1)[i]));
    }

    double (*u2)[size];
    u2 = (double(*)[size])malloc ((size) * sizeof(double));
    for (int i = 0; i < size; i++) {
        fscanf(filename, "%lf", &((*u2)[i]));
    }

    double (*v2)[n];
    v2 = (double(*)[n])malloc ((n) * sizeof(double));
    for (int i = 0; i < n; i++) {
        fscanf(filename, "%lf", &((*v2)[i]));
    }

    double (*w)[size];
    w = (double(*)[size])malloc ((size) * sizeof(double));
    for (int i = 0; i < size; i++) {
        fscanf(filename, "%lf", &((*w)[i]));
    }

    double (*x)[n];
    x = (double(*)[n])malloc ((n) * sizeof(double));
    for (int i = 0; i < n; i++) {
        fscanf(filename, "%lf", &((*x)[i]));
    }

    double (*y)[size];
    y = (double(*)[size])malloc ((size) * sizeof(double));
    for (int i = 0; i < size; i++) {
        fscanf(filename, "%lf", &((*y)[i]));
    }

    double (*z)[n];
    z = (double(*)[n])malloc ((n) * sizeof(double));
    for (int i = 0; i < n; i++) {
        fscanf(filename, "%lf", &((*z)[i]));
    }

    fclose(filename);


    /*----------------------- FILL DATA PART END --------------------------*/

    /* Now continue calc of broken proc */
    kernel_gemver (group, 
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


    // send main proc calculated data
    size = k + (0 < m);
    MPI_Send(w, size, MPI_DOUBLE, 0, 13, MPI_COMM_WORLD);

    // free allocated memory
    free((void*)A);
    free((void*)u1);
    free((void*)v1);
    free((void*)u2);
    free((void*)v2);
    free((void*)w);
    free((void*)x);
    free((void*)y);
    free((void*)z);
  }
  else {
    // no error
    printf("No error while calc\n");
  }
}



/// Main calc function
static
void kernel_gemver(int proc_group,
     int n,
     int size,
     double alpha,
     double beta,
     double A[ size][n],
     double u1[ size],
     double v1[ n],
     double u2[ size],
     double v2[ n],
     double w[ size],
     double x[ n],
     double y[ size],
     double z[ n])
{
  MPI_Status status[1]; // for MPI funcs calls

  int proc;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);

  if(proc_group <= 1) {
    // create breakpoint for 1-st group
    set_breakpoint (1, 
                 n, 
                 size, 
                 alpha, 
                 beta,
                 A,
                 u1,
                 v1,
                 u2,
                 v2,
                 w,
                 x,
                 y,
                 z);
    
    // calc 1-st group
    for (int i = 0; i < size; i++) {
      // kill one proc
      //*******
      for (int j = 0; j < n; j++)
        A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
    }
    
  }

  if(proc_group <= 2) {
    // create breakpoint for 2-nd group
    set_breakpoint (2, 
                 n, 
                 size, 
                 alpha, 
                 beta,
                 A,
                 u1,
                 v1,
                 u2,
                 v2,
                 w,
                 x,
                 y,
                 z);

    // calc 2-nd group
    for (int i = 0; i < size; i++)
      for (int j = 0; j < n; j++)
        x[i] = x[i] + beta * A[j][i] * y[j];
  }

  if(proc_group <= 3) {
    // create breakpoint for 3-rd group
    set_breakpoint (3, 
               n, 
               size, 
               alpha, 
               beta,
               A,
               u1,
               v1,
               u2,
               v2,
               w,
               x,
               y,
               z);

    // calc 3-rd group
    for (int i = 0; i < size; i++)
      x[i] = x[i] + z[i];

    // gather information in main proc for last calc
    // allocate memory for info gathering
    double (*cur)[n];
    cur = (double(*)[n])malloc(n * sizeof(double));

    /*------Main proc------*/
    if(proc == 0) {
      // gather info
      for(int i = 1; i < using_proc; i++) {
        printf("Waiting message from process %d in kernel\n", active_proc[i]);

        // get proc info status
        int info_status = MPI_Recv(*cur, size, MPI_INT, active_proc[i], 13, MPI_COMM_WORLD, status);

        if(info_status == 0) {
          // no error
          printf("Get message from process %d in kernel\n", active_proc[i]);
        }
        else {
          // error
          printf("Process %d has fault in kernel\n", active_proc[i]);

          // send message to spare proc
          MPI_Send(&active_proc[i], 1, MPI_INT, using_proc, 13, MPI_COMM_WORLD);

          // set cur proc to spare proc
          active_proc[i] = using_proc;

          // reset last proc
          i--;
          continue;        
        } 
      }

      // sent gathered data to all procs
      for(int i = 0; i < using_proc; i++) {
        MPI_Send(x, n, MPI_DOUBLE, active_proc[i], 13, MPI_COMM_WORLD);
      }
    }
    /*------Calc proc------*/
    else {
      // send data to main proc
      MPI_Send(x, n, MPI_DOUBLE, 0, 13, MPI_COMM_WORLD);

      // receive info from main proc
      MPI_Recv(x, n, MPI_DOUBLE, 0, 13, MPI_COMM_WORLD, status);
    }
  }

  // last calcs
  if(proc_group <= 4) {
    // create breakpoint for 4-th group
    set_breakpoint (4, 
               n, 
               size, 
               alpha, 
               beta,
               A,
               u1,
               v1,
               u2,
               v2,
               w,
               x,
               y,
               z);

    // calc 4-th group
    for (int i = 0; i < size; i++)
      for (int j = 0; j < n; j++)
        w[i] = w[i] + alpha * A[i][j] * x[j];
  }
  
}