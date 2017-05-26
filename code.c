

/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>


void local_matrix_mult(double *a, double *b, double *c, int size, int rank);
int assert_mult(double *Cglobal, double *CglobalTest, int size);

int verify=0;

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  srand(0);

  int i,j,k,n_local_rows,n;
  int rank, nprocs, sqnprocs;
  double *A, *Aglobal, *current_A, *B, *Bglobal, *C, *Cglobal, *current_B, *CglobalTest;
  double range, time_init, time_end;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);   /* get current process id */
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* get number of processes */

  MPI_Request request[nprocs];
  MPI_Datatype strided; 
  MPI_Status status[nprocs];

  n = 288;
  range = 1.0;
  
  sqnprocs = sqrt(nprocs);

  if(argc>1) {
    n=atoi(argv[1]);
  }

  if(verify==1) n=4;

  // set up number of rows per proc
  // NOTE: ONLY WORKS WHEN n IS MULTIPLE OF nprocs
  n_local_rows = n/nprocs;

  //////////////////////////////////////////////////////////////////////////////////

  A = malloc(n*n_local_rows*sizeof(double));
  B = malloc(n*n_local_rows*sizeof(double));
  C = malloc(n*n_local_rows*sizeof(double));
  current_A = malloc((n*n/nprocs)*sizeof(double));
  current_B = malloc((n*n/nprocs)*sizeof(double));

  for(i=0;i<n*n_local_rows;i++){
    C[i] = 0;
  }

  if (rank==0)
    {
      printf("\n Matrix size = %d \n", n);
      Aglobal = malloc(n*n*sizeof(double));
      Bglobal = malloc(n*n*sizeof(double));

      if(verify==0) {
	for (i=0; i<n; i++) {
	  for (j=0; j<n; j++) {
	    Aglobal[i*n+j]=range*(1.0-2.0*(double)rand()/RAND_MAX);
	    Bglobal[i*n+j]=range*(1.0-2.0*(double)rand()/RAND_MAX);
	  }
	}        
      }
      else {
	CglobalTest = malloc(n*n*sizeof(double));
	// For debugging
	for (i=0; i<n; i++){
	  for (j=0; j<n; j++){
	    Aglobal[i*n+j]=(i+j+1)*1.0;
	    Bglobal[i*n+j]=(i+j+1)*1.0;
	  }
	}              
	CglobalTest[0]=30;
	CglobalTest[1]=40;
	CglobalTest[2]=50;
	CglobalTest[3]=60;
	CglobalTest[4]=40;
	CglobalTest[5]=54;
	CglobalTest[6]=68;
	CglobalTest[7]=82;
	CglobalTest[8]=50;
	CglobalTest[9]=68;
	CglobalTest[10]=86;
	CglobalTest[11]=104;
	CglobalTest[12]=60;
	CglobalTest[13]=82;
	CglobalTest[14]=104;
	CglobalTest[15]=126;
      }

      if(n <11) {
	printf("\n Matrix A (generated randomly):\n");
      }
    }

  /*Set up cartesian communicator */
  MPI_Barrier(MPI_COMM_WORLD);
  int row_rank, col_rank,grid_rank, ndims, reorder, numA, numB, bonk2;
  int dims[2], coords[2], cyclic[2];
  MPI_Comm proc_grid, proc_row, proc_col;
  ndims = 2;
  reorder = 1;
  dims[0] = sqnprocs;
  dims[1] = sqnprocs;
  cyclic[0] = 0;
  cyclic[1] = 0;
  MPI_Cart_create(MPI_COMM_WORLD,ndims,dims,cyclic, reorder,&proc_grid);
  MPI_Comm_rank(proc_grid,&grid_rank);
  MPI_Cart_coords(proc_grid,rank,ndims,coords);
  MPI_Comm_split(proc_grid,coords[0],coords[1],&proc_row);
  MPI_Comm_rank(proc_row,&row_rank);
  MPI_Comm_split(proc_grid,coords[1],coords[0],&proc_col);
  MPI_Comm_rank(proc_col,&col_rank);

    printf("\nCommunicator set up\n");
  //Distribution of A and B to all processes
  if(rank == 0) {
    time_init = MPI_Wtime(); //start timer  
    MPI_Type_vector(n/sqnprocs, n/sqnprocs, n, MPI_DOUBLE, &strided);
    MPI_Type_commit(&strided);

    for (i=0; i<sqnprocs; i++) {
      for (j=0; j<sqnprocs; j++){ 
        MPI_Cart_rank(proc_grid,coords,&rank);
        MPI_Isend(&Aglobal[(n*n/nprocs)*2*i+n/sqnprocs*j], 1, strided, (j+i*sqnprocs), 1, proc_grid, &request[i*sqnprocs+j]); //[A,mytype]
        MPI_Isend(&Bglobal[(n*n/nprocs)*2*i+n/sqnprocs*j], 1, strided, (j+i*sqnprocs), 2, proc_grid, &request[i*sqnprocs+j]); //[B,mytype]
      }
    }
  }
 
  MPI_Barrier(MPI_COMM_WORLD);
    printf("\n A och B skickade\n");
  MPI_Recv(A,n*n/nprocs,MPI_DOUBLE,0,1,proc_grid,&status[rank]); //[A,double]   //Ta emot en fyrkant
  MPI_Recv(B,n*n/nprocs,MPI_DOUBLE,0,2,proc_grid,&status[rank]); //[B,double]  


        
        MPI_Get_elements(&status[rank],MPI_DOUBLE,&bonk2);
        printf("\n rank %d got %d elements of type MPI_DOUBLE\n",rank, bonk2);


    printf("\n A och B mottagna\n");
  memcpy(current_A, A,(n*n/nprocs)*sizeof(double)); 
  MPI_Barrier(proc_grid);
  //Fox Algo
  for (k=0; k<sqnprocs; k++) {
 
    MPI_Bcast(current_A,n*n/nprocs,MPI_DOUBLE,(k+coords[0])%sqnprocs, proc_row);

    /* Start shifting B blocks upwards */
    MPI_Isend(&B[0],n*n/nprocs,MPI_DOUBLE,(coords[0]-1+nprocs)%sqnprocs,3,proc_col,&request[rank]);

    local_matrix_mult(current_A, B, C, n/sqnprocs, rank);

    /*Receive B blocks from below */
    MPI_Recv(current_B,n*n/nprocs,MPI_DOUBLE,(coords[0]+1+nprocs)%sqnprocs,3,proc_col,&status[rank]);
    MPI_Barrier(MPI_COMM_WORLD);

    /*update current_A and B */
    memcpy(current_A, A,(n*n/nprocs)*sizeof(double));
    memcpy(B, current_B,(n*n/nprocs)*sizeof(double));

  }
    printf("\nC ska skickas\n");

  //Collect blocks of C
  MPI_Isend(C, n*n/nprocs, MPI_DOUBLE, 0, 4, proc_grid, &request[rank]);  //[C,double] 
  printf("\nC har skickas, %d element \n", n*n/nprocs);
    // MPI_Barrier(proc_grid);

  if(rank==0){
    free(Aglobal);
    free(Bglobal);

    Cglobal = malloc((n*n+4*n)*sizeof(double));
    for (i=0; i<nprocs; i++) {
      MPI_Probe(i,4,proc_grid, &status[i]);
      MPI_Cart_coords(proc_grid, i,2,coords);
      printf("\nC ska tas emot till plats: %d \n", coords[1]*n/sqnprocs+coords[0]*n*n/sqnprocs);      
      MPI_Recv(&Cglobal[coords[1]*n/sqnprocs+coords[0]*n*n/sqnprocs],1, strided, i,4,proc_grid, &status[i]); //[C,type]
      printf("\nC har tagits emot till plats: %d \n", coords[1]*n/sqnprocs+coords[0]*n*n/sqnprocs);
    }

  }

  MPI_Wait(request, status);

  if(rank == 0){
    time_end = MPI_Wtime() - time_init;
    printf("\n Multiplication done!\n Elapsed time %.16f s\n", time_end);
    if(verify==1){
      for(j=0; j<n*n;j++){
	printf("\nCglobal[%d] =  %g", j, Cglobal[j]);
      }
      printf("\n");
      if(assert_mult(Cglobal, CglobalTest, n)){
	printf("\nTest ok!\n");}
      else{
	printf("\n Assertion failed\n");
      }
      free(CglobalTest);
    }

    free(Cglobal);
  }
  free(A);
  free(B);
  free(C);

  MPI_Finalize(); 
}

void local_matrix_mult(double *a, double *b, double *c, int size, int rank){
  int i, j, l;
  double temp;
  for (i = 0; i < size; i++){
    for (j = 0; j < size; j++) {
      temp=0;
      for (l = 0; l < size; l++) {
	temp = temp + a[i*size+l]*b[l*size+j];
      }
      c[i*size+j] = c[i*size+j]+temp;
    }
  }
}

int assert_mult(double *Cglobal, double *CglobalTest, int size){
  int i;
  for(i=0; i<size; i++){
    if(Cglobal[i] != CglobalTest[i]){
      return 0;
    }
  }
  return 1;
}





