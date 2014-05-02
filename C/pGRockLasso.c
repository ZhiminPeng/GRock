/*----------------------------------------------------------
 * Solve a distributed lasso problem, i.e.,
 *
 *   minimize lambda ||x||_1 + 0.5 * ||Ax - b||_2^2
 *
 * The implementation uses MPI for distributed communication
 * and the GNU Scientific Library (GSL) for math.
 *
 * Compile: make
 * run code: mpiexec -n 1 ./lasso $dataDirectory$
 *
 * Copyright (c)     Zhimin Peng @ Math, UCLA
 * Created Date:     01/11/2013
 * Modified Date:    05/02/2013
 *                   05/02/2014, 
 *                   add dynamic update P, 
 *                   fixed some bugs
 * --------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mmio.h"
#include <mpi.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>


struct value {
  int    ID;
  double data;
};

typedef int (*compfn)(const void*, const void*);
int compare(struct value *, struct value *);
void shrink(gsl_vector *v, double sigma);
double objective(gsl_vector *x, double lambda, gsl_vector *z);
gsl_vector *pointwise(gsl_vector *a, gsl_vector *b, double n);

int main(int argc, char **argv) {

  const int MAX_ITER  = 2000;
  const double TOL = 1e-6;
  
  int rank;
  int size;
  int P = 8; // number of blocks to update P <= size

  /* -----------------------------------
     mode controls the selection schemes, 
     mode =1, GRock
     mode =0, Mixed CD
     ----------------------------------*/
  int mode=1; // number of processors used to update each time
  double lambda = 0.1;
  srand (time(NULL));
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Determine current running process
  MPI_Comm_size(MPI_COMM_WORLD, &size); // Total number of processes
  
  // data directory (you need to change the path to your own data directory)
  char* dataCenterDir = "./Data/Gaussian";
  char* big_dir;
  if(argc==2)
    big_dir = argv[1];
  else
    big_dir = "big1";
  
  
  /* Read in local data */
  
  FILE *f, *test;
  int m, n, j;
  int row, col;
  double entry, startTime, endTime;
  /*
   * Subsystem n will look for files called An.dat and bn.dat
   * in the current directory; these are its local data and do not need to be
   * visible to any other processes. Note that
   * m and n here refer to the dimensions of the *local* coefficient matrix.
   */
  
  /* ------------
     Read in A 
     ------------*/
  char s[100];
  sprintf(s, "%s/%s/A%d.dat",dataCenterDir,big_dir, rank + 1);
  printf("[%d] reading %s\n", rank, s);
  f = fopen(s, "r");
  if (f == NULL) {
    printf("[%d] ERROR: %s does not exist, exiting.\n", rank, s);
    exit(EXIT_FAILURE);
  }
  mm_read_mtx_array_size(f, &m, &n);
  gsl_matrix *A = gsl_matrix_calloc(m, n);
  for (int i = 0; i < m*n; i++) {
    row = i % m;
    col = floor(i/m);
    fscanf(f, "%lf", &entry);
    gsl_matrix_set(A, row, col, entry);
  }
  fclose(f);
  
  /* ------------
     Read in b 
     -------------*/
  sprintf(s, "%s/%s/b.dat", dataCenterDir, big_dir);
  f = fopen(s, "r");
  if (f == NULL) {
    printf("[%d] ERROR: %s does not exist, exiting.\n", rank, s);
    exit(EXIT_FAILURE);
  }
  mm_read_mtx_array_size(f, &m, &n);
  gsl_vector *b = gsl_vector_calloc(m);
  for (int i = 0; i < m; i++) {
    fscanf(f, "%lf", &entry);
    gsl_vector_set(b, i, entry);
  }
  fclose(f);
  
  /* ------------
     Read in xs 
     ------------*/
  sprintf(s, "%s/%s/xs%d.dat", dataCenterDir, big_dir, rank + 1);
  printf("[%d] reading %s\n", rank, s);
  f = fopen(s, "r");
  if (f == NULL) {
    printf("[%d] ERROR: %s does not exist, exiting.\n", rank, s);
    exit(EXIT_FAILURE);
  }
  mm_read_mtx_array_size(f, &m, &n);
  gsl_vector *xs = gsl_vector_calloc(m);
  
  for (int i = 0; i < m; i++) {
    fscanf(f, "%lf", &entry);
    gsl_vector_set(xs, i, entry);
  }
  fclose(f);
  
  m = A->size1;
  n = A->size2;
  MPI_Barrier(MPI_COMM_WORLD);
  
  /*----------------------------------------
   * These are all variables related to BCD
   ----------------------------------------*/
  
  struct value table[size];
  gsl_vector *x      = gsl_vector_calloc(n);
  gsl_vector *As     = gsl_vector_calloc(n);
  gsl_vector *invAs  = gsl_vector_calloc(n);
  gsl_vector *local_b= gsl_vector_calloc(m);
  gsl_vector *beta   = gsl_vector_calloc(n);
  gsl_vector *tmp    = gsl_vector_calloc(n);
  gsl_vector *d      = gsl_vector_calloc(n);
  gsl_vector *oldx   = gsl_vector_calloc(n);
  gsl_vector *z      = gsl_vector_calloc(m);
  gsl_vector *Ax     = gsl_vector_calloc(m);
  gsl_vector *xdiff  = gsl_vector_calloc(n);
  double send[1]; 
  double recv[1]; 
  double err;
  double xs_local_nrm[1], xs_nrm[1];
  
  //calculate the 2 norm of xs
  xs_local_nrm[0] = gsl_blas_dnrm2(xs);
  xs_local_nrm[0] *=xs_local_nrm[0];
  MPI_Allreduce(xs_local_nrm, xs_nrm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  xs_nrm[0] = sqrt(xs_nrm[0]);
  
  // evaluate the two norm of the columns of A
  for(j=0;j<n;j++){
    gsl_vector_view column = gsl_matrix_column(A, j);
    double d;
    d = gsl_blas_dnrm2(&column.vector);
    gsl_vector_set(As, j, d*d);
    gsl_vector_set(invAs, j, 1./(d*d));
  }
  
  if (rank == 0) {
    printf("%3s %10s %10s %10s\n", "#", "relative error", "obj", "time per iteration");
    startTime = MPI_Wtime();
    sprintf(s, "results/test%d.m", size);
    test = fopen(s, "w");
    fprintf(test,"res = [ \n");
  }
  
  /* Main BCD loop */
  int iter = 0, list[size], vektor[P];
  while (iter < MAX_ITER) {
    startTime = MPI_Wtime();
    /* ---------------------------------------------
       This mode randomly select P blocks to update
       ---------------------------------------------*/
    if(mode == 0){
      if(size>P){
	for (int i = 0; i < size; i++) {
	  list[i] = i;
	}
	for (int i = 0; i < size; i++) {
	  int j = i + rand() % (size - i);
	  int temp = list[i];
	  list[i] = list[j];
	  list[j] = temp;
	}
	for (int i = 0; i < P; i++) {
	  vektor[i] = list[i];
	}
      }
    }

    /*---------- restore the old x ------------*/
    gsl_vector_memcpy(oldx, x);
    
    /*------- calculate local_b = b - sum_{j \neq i} Aj*xj--------- */ 
    gsl_blas_dgemv(CblasNoTrans, 1, A, x, 0, Ax); // Ax = A * x
    MPI_Allreduce(Ax->data, z->data,  m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    gsl_vector_sub(z, b); // z = Ax - b
    gsl_vector_memcpy(local_b, Ax);
    gsl_vector_sub(local_b, z);
    
    /* -------calculate beta ------------------*/
    gsl_blas_dgemv(CblasTrans, -1, A, z, 0, beta); // beta = A'(b - Ax) + ||A.s||^2 * xs
    tmp = pointwise(As, x, n);
    gsl_vector_add(beta, tmp);
    shrink(beta, lambda);
    // x = 1/|xs|^2 * shrink(beta, lambda)
    x = pointwise(beta, invAs, n); 
  
    /* ------calcuate proposed decrease -------- */
    gsl_vector_memcpy(d,x);
    gsl_vector_sub(d, oldx);
    CBLAS_INDEX_t idx = gsl_blas_idamax(d);

    /* ---------------------------------------------
       This mode greedily select P blocks to update
       ---------------------------------------------*/
    if(mode == 1){
      double *store = (double*)calloc(size, sizeof(double));
      double foo[1];
      foo[0] = gsl_vector_get(d,idx);
      MPI_Allgather(foo, 1, MPI_DOUBLE, store, 1, MPI_DOUBLE, MPI_COMM_WORLD);
      for(int i=0;i<size;i++){
	table[i].ID   = i;
	table[i].data = fabs(store[i]);
      }
      // quick sort to decide which block to update
      qsort((void *) & table, size, sizeof(struct value), (compfn)compare );
    }
      
    gsl_vector_memcpy(x, oldx);
  
    /*----------------------------------
      determine which block to update
      ---------------------------------*/
    if(size>P && mode ==1){
      for(int i=0;i<P;i++){
	if(rank == table[i].ID)
	  gsl_vector_set(x, idx, gsl_vector_get(oldx, idx) + gsl_vector_get(d, idx));
      }
    }
    else if(size > P && mode ==0){
      for(int i=0;i<P;i++){
	if(rank == vektor[i])
	  gsl_vector_set(x, idx, gsl_vector_get(oldx, idx) + gsl_vector_get(d, idx));
      }
    }
    else{
      gsl_vector_set(x, idx, gsl_vector_get(oldx, idx) + gsl_vector_get(d, idx));
    }

    /*------------------------------
      calculate the relative error
      ------------------------------*/
    gsl_vector_memcpy(xdiff,xs);
    gsl_vector_sub(xdiff, x);
    err = gsl_blas_dnrm2(xdiff);
    send[0] = err*err;
    MPI_Allreduce(send, recv, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    recv[0] = sqrt(recv[0])/xs_nrm[0];
 
    endTime = MPI_Wtime();
    
    if (rank == 0) {
      printf("%3d %e %10.4f %e\n", iter,
	     recv[0],  objective(x, lambda, z), endTime - startTime);
      fprintf(test, "%e \n",recv[0]);
    }

    /* termination check */
    if(recv[0] < TOL){
      break;
    }
    iter++;
  }
  
  /* Have the master write out the results to disk */
  if (rank == 0) {
    fprintf(test,"] \n");
    fprintf(test,"semilogy(1:length(res),res); \n");
    fprintf(test,"xlabel('# of iteration'); ylabel('||x - xs||');\n");
    fclose(test);
    f = fopen("results/solution.dat", "w");
    fprintf(f,"x = [ \n");
    gsl_vector_fprintf(f, x, "%lf");
    fprintf(f,"] \n");
    fclose(f);
    endTime = MPI_Wtime();
  }
  
  MPI_Finalize(); /* Shut down the MPI execution environment */
  
  /* Clear memory */
  gsl_matrix_free(A);
  gsl_vector_free(b);
  gsl_vector_free(x);
  gsl_vector_free(z);
  gsl_vector_free(xdiff);
  gsl_vector_free(Ax);
  gsl_vector_free(As);
  gsl_vector_free(invAs);
  gsl_vector_free(oldx);
  gsl_vector_free(local_b);
  gsl_vector_free(beta);
  
  return 0;
}


/* ----------- evaluate the objective function --------------*/
double objective(gsl_vector *x, double lambda, gsl_vector *z) {
  double obj = 0;
  double temp =0.0;
  temp = gsl_blas_dnrm2(z);
  temp = temp*temp/2;
  double foo;
  foo = gsl_blas_dasum(x);
  obj = lambda*foo + temp;
  return obj;
}

/*----------- shrinkage function ---------------------- */
void shrink(gsl_vector *v, double sigma) {
  double vi;
  for (int i = 0; i < v->size; i++) {
    vi = gsl_vector_get(v, i);
    if (vi > sigma)       { gsl_vector_set(v, i, vi-sigma); }
    else if (vi < -sigma) { gsl_vector_set(v, i, vi+sigma); }
    else              { gsl_vector_set(v, i, 0); }
  }
}


/* ----------- point wise product function ----------------- */
gsl_vector *pointwise(gsl_vector *a, gsl_vector *b, double n){
  gsl_vector *c      = gsl_vector_calloc(n);
  for(int i=0; i<n; i++)
    gsl_vector_set(c, i, gsl_vector_get(a, i) * gsl_vector_get(b, i));
  
  return c;
  
}


int compare(struct value *tab1, struct value *tab2){

  if(tab1->data < tab2->data)
    return 1;
  else if (tab1->data > tab2->data)
    return -1;
  else
    return 0;
}
