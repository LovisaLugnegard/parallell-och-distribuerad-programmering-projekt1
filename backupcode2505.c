


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
void local_matrix_mult(double *a, double *b, double *c, int size, int rank);
int verify=1;

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  srand(0);

  int i,j,k,l,m,n_local_rows,n;
  int rank, nprocs, sqnprocs,bonk2;
  double *A, *Aglobal, *current_A, *B, *Bglobal, *C, *Cglobal, *current_B, *CglobalTest;
  double range, begin, end;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);   /* get current process id */
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* get number of processes */

  MPI_Request request[nprocs];
  MPI_Datatype strided; 
  MPI_Status status[nprocs];

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

  for(i=0;i<n*n_local_rows;i++){
    C[i] = 0;
  }

  if (rank==0)
    {
      Aglobal = malloc(n*n*sizeof(double));
      Bglobal = malloc(n*n*sizeof(double));
      Cglobal = malloc((n*n+2*n)*sizeof(double));
      CglobalTest = malloc(n*n*sizeof(double));

      if(verify==0) {
	for (i=0; i<n; i++) {
	  for (j=0; j<n; j++) {
	    Aglobal[i*n+j]=range*(1.0-2.0*(double)rand()/RAND_MAX);
	    Bglobal[i*n+j]=range*(1.0-2.0*(double)rand()/RAND_MAX);}
	}        
      }
      else {
	//	printf("I am in else!!! n: %d", n);
	// For debugging
	for (i=0; i<n; i++){
	  for (j=0; j<n; j++){
	    Aglobal[i*n+j]=i+j+1;
	    Bglobal[i*n+j]=i+j+1;
	    printf("\nAglobal filled with %d \n", i+j+1);
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
	//	printf("\n\n");
      }
    }

  MPI_Barrier(MPI_COMM_WORLD);
  int row_rank, col_rank,grid_rank, ndims,  reorder,numA,numB;
  int dims[2], coords[2], cyclic[2];
  MPI_Comm proc_grid, proc_row, proc_col;
  ndims = 2;
  reorder = 1;
  dims[0] = sqnprocs;
  dims[1] = sqnprocs;
  cyclic[0] = 0;
  cyclic[1] = 0;
  // printf("\n before coords create \n");
  //MPI_Dims_create(nprocs,ndims,dims);
  MPI_Cart_create(MPI_COMM_WORLD,ndims,dims,cyclic, reorder,&proc_grid);
  MPI_Comm_rank(proc_grid,&grid_rank);
  MPI_Cart_coords(proc_grid,rank,ndims,coords);
  MPI_Comm_split(proc_grid,coords[0],coords[1],&proc_row);
  MPI_Comm_rank(proc_row,&row_rank);
  MPI_Comm_split(proc_grid,coords[1],coords[0],&proc_col);
  MPI_Comm_rank(proc_col,&col_rank);

  // printf("\n after coords create \n");

  //Distribution
  if(rank == 0) {
    MPI_Type_vector(n/sqnprocs, n/sqnprocs, n, MPI_DOUBLE, &strided);
    MPI_Type_commit(&strided);
    printf("nprocs: %d sqnprocs: %d rank: %d", nprocs, sqnprocs, rank);

    for (i=0; i<sqnprocs; i++) {
      for (j=0; j<sqnprocs; j++){ 
	//   printf("\n i did not send yet \n");
	//   for(k=0;k<16;k++) {
	// printf("\n Aglobal is_: %g \n",Aglobal[k]);
	// printf("\n Bglobal is_: %g \n",Bglobal[k]);}
        MPI_Cart_rank(proc_grid,coords,&rank);
        MPI_Isend(&Aglobal[(n*n/nprocs)*2*i+n/sqnprocs*j], 1, strided, (j+i*sqnprocs), 1, proc_grid, &request[i*sqnprocs+j]); //[A,mytype]
        MPI_Isend(&Bglobal[(n*n/nprocs)*2*i+n/sqnprocs*j], 1, strided, (j+i*sqnprocs), 2, proc_grid, &request[i*sqnprocs+j]); //[B,mytype]
        printf("\n i did send! \n");
      }
    }
    // MPI_Waitall(nprocs - 1,request,status);
  }
 
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Recv(A,n*n/nprocs,MPI_DOUBLE,0,1,proc_grid,&status[rank]); //[A,double]   //Ta emot en fyrkant
  MPI_Recv(B,n*n/nprocs,MPI_DOUBLE,0,2,proc_grid,&status[rank]); //[B,double]  

  MPI_Get_elements(&status[rank],MPI_DOUBLE,&bonk2);
  printf("proc %d recv %d elements of type MPI_DOUBLE\n",rank,bonk2);
  for (i=0;i<bonk2;i++)
    if(A[i] != -1)
      printf("proc: %d A[%d] =  %g\n",rank, i, A[i]);
  printf("\n");

  for (i=0;i<bonk2;i++)
    if(B[i] != -1)
      printf("proc: %d B[%d] =  %g\n",rank, i, B[i]);
  printf("\n");
  MPI_Barrier(MPI_COMM_WORLD);

  printf("Processor %d is before fox algo. \n",rank);

  //hit verkar allt fungera :) (25/5-17)

  //Fox Algo
  for (k=0; k<sqnprocs; k++) {

    printf("\nrank %d, k = %d \n", rank,k);
    MPI_Barrier(MPI_COMM_WORLD);
    // for (i=0; i<sqnprocs; i++) {
    //broadcast A radvis, current_A verkar inte användas senare, det ger problem! det är ju det vi ska multiplicera med?
    //Jag tror att vi har tänkt fel med MPI_Bcast, dvs hur den fungerar roten ska vara processen som skickar /L

    /* if(coords[1]==(k+coords[0])%sqnprocs){ */
    /*   MPI_Barrier(proc_row); */
    /*   MPI_Bcast(A,n*n/nprocs,MPI_DOUBLE,0,proc_row); */
    /*   MPI_Barrier(proc_row); */
    /* } */
    /* else { */
    /*   MPI_Barrier(proc_row); */
    /*   MPI_Bcast(A,n*n/nprocs,MPI_DOUBLE,0,proc_row); */
    /*   MPI_Barrier(proc_row); */

    /*   //print curret_A */
    /*   MPI_Get_elements(&status[rank],MPI_DOUBLE,&bonk2); */
    /*   printf("andra ggn proc %d recv %d elements of type MPI_DOUBLE\n",rank,bonk2); */
    /*   for (i=0;i<bonk2;i++) */
    /* 	if(current_A[i] != -1) */
    /* 	  printf("proc: %d current_A[%d] =  %g\n",rank, i, current_A[i]); */
    /* } */  

    current_A = A;

    //Bcast verkar ok
    printf("\n bcast nr: %d\n", (k+coords[0])%sqnprocs);
    MPI_Bcast(current_A,n*n/nprocs,MPI_DOUBLE,(k+coords[0])%sqnprocs, proc_row);
    MPI_Barrier(MPI_COMM_WORLD);

    for (i=0;i<n*n/nprocs;i++)
      printf("\nproc: %d current_A[%d] =  %g\n",rank, i, current_A[i]);
    printf("\n");

    MPI_Barrier(MPI_COMM_WORLD);
    /*   //m=(i+k)%sqnprocs; */
    /*   // memcpy(current_A,&A,n/sqnprocs); */
    /*   // memcpy(current_col[i],&Aglobal[m]); */
    /*   //   MPI_Bcast(&current_A,n*n/nprocs,MPI_DOUBLE,0,proc_row); */
    /*   // } */
    /*   // for (j=0; j<n*n/nprocs;j++){ */
    /*   // for(m=0;m<n*n/nprocs;m++){ */
    /*   //   C[ */
    /*   //} */

    printf("\n Hi! I'm before Isend in fox algo :) k= %d, rank = %d, B's dest is: %d, coords[0] = %d \n", k, rank, ((coords[0]-1+nprocs)%nprocs)%sqnprocs, coords[0]);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Isend(&B[0],n/sqnprocs,MPI_DOUBLE,(coords[0]-1+nprocs)%sqnprocs,3,proc_col,&request[rank]);
    printf("\n Hi! Process %d after Isend in fox algo :)", rank );
  

    local_matrix_mult(current_A, B, C, n/sqnprocs, rank);


    // MPI_Wait(&request[rank], &status[rank]);
    MPI_Barrier(MPI_COMM_WORLD);

    printf("\n Process %d before receiving current B from: %d", rank, ((coords[0]+1+nprocs)%nprocs)%sqnprocs);
    MPI_Recv(current_B,n/sqnprocs,MPI_DOUBLE,(coords[0]+1+nprocs)%sqnprocs,3,proc_col,&status[rank]);
    MPI_Barrier(proc_col);
    printf("\n Proc: %d after receiving current B", rank);
    int temp = *B;
    *B=*current_B;
    *current_B=temp;

    /* MPI_Sendrecv_replace(B, n*n/nprocs,MPI_DOUBLE, */
    /* 			 (row_rank+sqnprocs-1)%sqnprocs , 0, (coords[0]+1+sqnprocs)%sqnprocs, 0, proc_col, status); */

      MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  printf("\nProcess : %d, C:", rank);
  for(j=0; j<n*n/nprocs;j++){
    printf(" C[%d] =  %g ", j, C[j]);
  }
  printf("\n");



  //Collect blocks of C
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Isend(C, n*n/nprocs, MPI_DOUBLE, 0, 4, proc_grid, &request[rank]);  //[C,double] Varför har vi &request[k]???/L 
  printf("\n Process %d after MPI_Isend C", rank);
  MPI_Barrier(proc_grid);
  if(rank==0){
    printf("\n rank=0");
    for (i=0; i<nprocs; i++) {
      MPI_Probe(i,4,proc_grid, &status[i]);

      /* int number_amount; */
      /* MPI_Get_count(&status[i], MPI_INT, &number_amount); */
      /* printf("0 dynamically received %d numbers from %d.\n", */
      /* 	     number_amount, i); */

      MPI_Cart_coords(proc_grid, i,2,coords);//vad använda coords till?
      printf("\n proc %d plats Cglob %d Coords[0] = %d coords[1]= %d\n",i, coords[1]*n/sqnprocs+coords[0]*n*n/sqnprocs, coords[0], coords[1]);
      MPI_Recv(&Cglobal[coords[1]*n*sqnprocs+coords[0]*n*n/sqnprocs],1, strided, i,4,proc_grid, &status[i]); //[C,type]
      MPI_Waitall(nprocs,request, status);
    }
    //  MPI_Waitall(nprocs,request, status);


  }

  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0){
    for(j=0; j<n*n;j++){
      printf("\nCglobal[%d] =  %g", j, Cglobal[j]);
    }
    printf("\n");}
  //  MPI_Wait(&request[nprocs], &status[nprocs]);
  MPI_Finalize(); 
}



void local_matrix_mult(double *a, double *b, double *c, int size, int rank){
  int i, j, l;
  double temp;
    for (i = 0; i < size; i++){
      for (j = 0; j < size; j++) {
	temp=0;
	for (l = 0; l < size; l++) {
	  //  printf("\n Process %d in matrix loop C= %g, A = %g, B is: %g \n",rank, C[i*n/sqnprocs+j] , A[i*n/sqnprocs+k], B[k*n/sqnprocs+j]);
    	  temp = temp + a[i*size+l]*b[l*size+j];
	  printf("\n Process %d in matrix loop C= %g, A = %g, B is: %g \n",rank, c[i*size+j] , a[i*size+l], b[l*size+j]);
	}
	c[i*size+j] = c[i*size+j]+temp;
      }
    }

}



