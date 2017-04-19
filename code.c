


/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>


//void print_mat(const double *A, int n);

int verify=1;

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  srand(0);

  int i,j,k,m,n_local_rows,n;
  int rank, nprocs, sqnprocs;
  double *A, *Aglobal, *current_A, *B, *Bglobal, *C, *Cglobal, *current_B, *CglobalTest;
  double range, begin, end;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);   /* get current process id */
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* get number of processes */

  MPI_Request request[nprocs];
  MPI_Datatype strided; 
  MPI_Status status;

  n = 40;
  range = 1.0;
  
  sqnprocs = sqrt(nprocs);
  printf("nprocs: %d sqnprocs: %d rank: %d", nprocs, sqnprocs, rank);

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

  if (rank==0)
    {
      Aglobal = malloc(n*n*sizeof(double));
      Bglobal = malloc(n*n*sizeof(double));
      Cglobal = malloc(n*n*sizeof(double));
      CglobalTest = malloc(n*n*sizeof(double));

      if(verify==0) {
	for (i=0; i<n; i++) {
	  for (j=0; j<n; j++) {
	    Aglobal[i*n+j]=range*(1.0-2.0*(double)rand()/RAND_MAX);
	    Bglobal[i*n+j]=range*(1.0-2.0*(double)rand()/RAND_MAX);}
	}        
      }
      else {
	printf("I am in else!!! n: %d", n);
	// For debugging
	for (i=0; i<n; i++){
	  for (j=0; j<n; j++){
	    Aglobal[i*n+j]=i+j+1;
	    Bglobal[i*n+j]=i+j+1;
	    printf("Aglobal filled with %d ", i+j+1);
	  }
	}
                
	//double Aglobal[16]= {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
	//  double Bglobal[16]= {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
	//	for(k=0;k<16;k++) {
	  //    printf("\n Aglobal is_: %g \n",Aglobal[k]);
          //printf("\n Bglobal is_: %g \n",Bglobal[k]);}
	//    Aglobal[0]=1;
	//Bglobal[0]=1;
	//Bglobal[1]=2;
	//Bglobal[2]=2;
	//Bglobal[3]=3;
	//Aglobal[1]=2;
	//Aglobal[2]=2;
	//Aglobal[3]=3;
	CglobalTest[0]=5;
	CglobalTest[1]=8;
	CglobalTest[2]=8;
	CglobalTest[3]=13;
      }

      if(n <11) {

	printf("\n Matrix A (generated randomly):\n");
	// print_mat(Aglobal,n);
	printf("\n\n");
      }
    }

  MPI_Barrier(MPI_COMM_WORLD);
  int row_rank, col_rank, ndims,  reorder,numA,numB;
  int dims[2], coords[2], cyclic[2];
  MPI_Comm proc_grid, proc_row, proc_col;
  ndims = 2;
  reorder = 1;
  dims[0] = sqnprocs;
  dims[1] = sqnprocs;
  cyclic[0] = 0;
  cyclic[1] = 0;
  printf("\n before coords create \n");
  //MPI_Dims_create(nprocs,ndims,dims);
  MPI_Cart_create(MPI_COMM_WORLD,ndims,dims,cyclic, reorder,&proc_grid);
  MPI_Comm_rank(proc_grid,&rank);
  MPI_Cart_coords(proc_grid,rank,ndims,coords);
  MPI_Comm_split(proc_grid,coords[0],coords[1],&proc_row);
  MPI_Comm_rank(proc_row,&row_rank);
  MPI_Comm_split(proc_grid,coords[1],coords[0],&proc_col);
  MPI_Comm_rank(proc_col,&col_rank);
  printf("\n after coords create \n");
  //Distribution
  if(rank == 0) {
    MPI_Type_vector(n/sqnprocs, n/sqnprocs, n, MPI_DOUBLE, &strided);
    MPI_Type_commit(&strided);
    printf("nprocs: %d sqnprocs: %d rank: %d", nprocs, sqnprocs, rank);

    for (i=0; i<sqnprocs; i++) {
      // coords[0] = i;
      for (j=0; j<sqnprocs; j++){
        // coords[1] = j; 
        printf("\n i did not send yet \n");
	//   for(k=0;k<16;k++) {
	// printf("\n Aglobal is_: %g \n",Aglobal[k]);
	// printf("\n Bglobal is_: %g \n",Bglobal[k]);}
        MPI_Cart_rank(proc_grid,coords,&rank);
        MPI_Isend(&Aglobal[i*sqnprocs+j], 1, strided, (j+i*sqnprocs), 1, proc_grid, &request[i*sqnprocs+j]); //[A,mytype] var tidigare request[i][j]
        MPI_Isend(&Bglobal[i*sqnprocs+j], 1, strided, (j+i*sqnprocs), 2, proc_grid, &request[i*sqnprocs+j]); //[B,mytype] borde ha en annan tag än A
        // MPI_Isend(&ndims, 1, MPI_INT, (j+i*sqrt(nprocs)), 1, proc_grid, &request[i*sqnprocs+j]); //[A,mytype] var tidigare request[i][j]
        // MPI_Isend(&ndims, 1, MPI_INT, (j+i*sqrt(nprocs)), 2, proc_grid, &request[i*sqnprocs+j]); //[B,mytype] borde ha en annan tag än A
        printf("\n i did send! \n");
      }
    }
  }
  printf("nprocs: %d sqnprocs: %d rank: %d", nprocs, sqnprocs, rank);

  MPI_Recv(&A,n*n/nprocs,MPI_DOUBLE,0,1,proc_grid,&status); //[A,double]   //Ta emot en fyrkant


  //HÄR KAN DET VARA ETT PROBLEM!!!!!!!!!!

  MPI_Recv(&B,n*n/nprocs,MPI_DOUBLE,0,2,proc_grid,&status); //[B,double]  
  // MPI_Recv(&numA,1,MPI_INT,0,1,proc_grid,&status); //[A,double]   //Ta emot en fyrkant
  // MPI_Recv(&numB,1,MPI_INT,0,2,proc_grid,&status); //[B,double]  
  printf("processor %d got \n",rank);
  printf("processor %d got \n",rank);
     
  //wait
  MPI_Waitall(nprocs,request,&status);

  //Fox Algo
  for (k=0; k<sqnprocs; k++) {
    // for (i=0; i<sqnprocs; i++) {
    if(coords[1]==(k+coords[0])%sqnprocs){
      MPI_Bcast(&A,n*n/nprocs,MPI_DOUBLE,0,proc_row);}
    else {
      MPI_Bcast(&current_A,n*n/nprocs,MPI_DOUBLE,0,proc_row);}
     


    //m=(i+k)%sqnprocs;
    // memcpy(current_A,&A,n/sqnprocs);
    // memcpy(current_col[i],&Aglobal[m]);
    //   MPI_Bcast(&current_A,n*n/nprocs,MPI_DOUBLE,0,proc_row);
    // }
    // for (j=0; j<n*n/nprocs;j++){
    // for(m=0;m<n*n/nprocs;m++){
    //   C[
    //}

    MPI_Isend(&B,n/sqnprocs,MPI_DOUBLE,(coords[0]-1)%sqnprocs,3,proc_col,&request[k]);
    //SKRIV RIKTIG MATRISKOD!
    for (i = 0; i < n/sqnprocs; i++){
      for (j = 0; j < n/sqnprocs; j++) {
        for (k = 0; k < n/nprocs; k++) {
	  C[i*n/sqnprocs+j] +=A[i*n/sqnprocs+k]*B[k*n/sqnprocs+j];// Entry(A,i,k)*Entry(B,k,j); }
        }
      }
    }


    // C += current_A*B;}//skapa block C

    MPI_Recv(&current_B,n/sqnprocs,MPI_DOUBLE,(coords[0]+1)%sqnprocs,3,proc_col,&status);
    double temp = *B;
    *B=*current_B;
    *current_B=temp;
    //   memcpy(B,&current_B,n/sqnprocs);
  }

  //Skift B 
  //Collect blocks of C
  MPI_Isend(&C, n*n/nprocs, MPI_DOUBLE, 0, 4, proc_grid, &request[k]);  //[C,double] 
  if(rank==0){
    for (i=0; i<nprocs; i++) {
      MPI_Probe(i,4,proc_grid, &status);
      MPI_Cart_coords(proc_grid, i,2,&coords[2]);//vad använda coords till?
      MPI_Recv(&Cglobal[coords[0]*sqnprocs+coords[1]],1, strided, i,4,proc_grid, &status); //[C,type]
    }
  }   
  MPI_Wait(request, &status);
}





