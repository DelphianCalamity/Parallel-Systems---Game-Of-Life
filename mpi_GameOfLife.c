#include <stdio.h>
#include <stdlib.h>
#include <math.h>


int main(int argc, char **argv)
{
	int numOftasks, taskid;
	char myGrid[2][Xdimension][Xdimension];
	Neighborhood neighbors;
		
	//Initialization
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numOftasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

	//Make a new communicator with Cartesian Topology								//RE-ORDER of tasks!!
	MPI_Comm myCom;
	int myCoords[2];

	//Each process will determine its own neighbors
	int dims[2], periodic[2];
	dims[0] = dims[1] = sqrtTasks;
	periodic[0] = periodic[1] = 1;													//Periodic Table - each cell has 8 neighbors

	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 1, &myCom);
	MPI_Cart_shift(myCom, 0, 1, &neighbors.north, &neighbors.south);				//Get the ids of the tasks with whom the current process will need to exchange information
	MPI_Cart_shift(myCom, 1, 1, &neighbors.west, &neighbors.east);

	MPI_Comm_rank(myCom, &taskid);													//Get new taskid of current process
	MPI_Cart_coords(myCom, taskid, 2, myCoords);
	
	restNeighbors(myCom, myCoords, sqrtTasks, &neighbors);							//Get the ids of the 'diagonal' neighbors

	/* Distribute work to workers. */
	int sqrtTasks = (int)sqrt(numtasks);
	int x = Xdimension/sqrtTasks;													//An x*x subarray will be assigned to each task/worker


	if (taskid == MASTER) {				//MASTER CODE

		argsChecking(numOftasks, Xdimension);

		printf ("Starting the Game Of Life with %d worker tasks.\n", numOftasks);
		/***** Initialize grid *****/
		printf("Grid size: X= %d  Y= %d  Time steps= %d\n", Xdimension, Xdimension, STEPS);
		printf("Initializing grid and writing initial.dat file...\n");
		inidat(Xdimension, *myGrid);
		prtdat(Xdimension, *myGrid, "initial.dat");


		/****** GET TIME ******/
	    double time = MPI_Wtime();


	    /******* Send the subarrays to their tasks *******/
		int id;
		int temp[2], init[2], xs[2], coords[2];

		MPI_Datatype *myDatatype = malloc(numOftasks*sizeof(MPI_Datatype));
		MPI_Request *requests = malloc(numOftasks*sizeof(MPI_Request));

		for (id=0; id<numOftasks; id++)
		{	
			MPI_Cart_coords(myCom, id, 2, coords);

			xs[0] = xs[1] = x;
			temp[0] = coords[0]*x; temp[1] = coords[1]*x;
			init[0] = init[1] = Xdimension;

			MPI_Type_create_subarray(2, init, xs, temp, MPI_ORDER_C, MPI_CHAR, myDatatype+id);					//Datatype creation - in order to avoid data copying
			MPI_Type_commit(myDatatype+id);

			MPI_Isend(&myGrid[0][0][0], 1, myDatatype[id], id, BEGIN, myCom, NULL);								//Non blocking SEND
		}

	    /* Wait for results from all worker tasks */
		for (i=0; i<numOftasks; i++)
			MPI_Irecv(&myGrid[0][0][0], 1, myDatatype[i], i, DONE, myCom, &requests[id]);

		MPI_Waitall(numOftasks, requests, MPI_STATUSES_IGNORE);
		
		//GET TIME
		time = MPI_Wtime() - time;

		/* Write final output, call X graph and finalize MPI */
		printf("Finished after %f seconds\n", time);
		printf("Writing final.dat file \n");
		prtdat(Xdimension, *(myGrid+1), "final.dat");

		//Free Memory
		for (id=0; id<numOftasks; id++)	
			MPI_Type_free(&myDatatype[i]);

		free(myDatatype);
		free(requests);
	}


	if (taskid != MASTER){ 				//WORKER CODE

		workerJob(myCom, x);
	}

	MPI_Comm_free(&myCom);
	MPI_Finalize();

	return 0;
}



/******************* DEFINITIONS ********************/

void workerJob(MPI_Comm myCom, int x){
	
	char *grid = malloc(x*x*sizeof(char));

    /******* Receive data from Master *******/
	MPI_Request request;
	MPI_Irecv(grid, x*x, MPI_CHAR, MASTER, BEGIN, myCom, request);

	DiagonalNeighbors diagonalNeighbors;
	MyDatatype myDatatype;
	create_DataType(myCom, 0, 0, x, x, 1, x, &myDatatype.north);
	create_DataType(myCom, x-1, 0, x, x, 1, x, &myDatatype.south);
	create_DataType(myCom, 0, 0, x, x, x, 1, &myDatatype.west);
	create_DataType(myCom, x-1, 0, x, x, x, 1, &myDatatype.east);

	//Send to neighbors
  	MPI_Sendrecv(, 1, MPI_INT, dest, tag,    &datain, 1, MPI_INT, so

	MPI_Wait(request, MPI_STATUS_IGNORE);


	free(grid);
}


void argsChecking(int numOfWorkers, int Xdimension){

  	if ((numOfWorkers > MAXWORKER) || (numOfWorkers < MINWORKER)) {
    	printf("ERROR: the number of tasks must be between %d and %d.\n", MINWORKER+1,MAXWORKER+1);
    	printf("Quitting...\n");
	    MPI_Abort(MPI_COMM_WORLD, rc);
	    exit(1);
    }
	//Simplifying the problem
	//Demanding that number of workers is a square number whose square root is perfectly divided by Grid's X & Y dimensions
	//Plus 1 task -> MASTER
	int sqrtTasks = sqrt(numOfWorkers);
	if(sqrtTasks*sqrtTasks != numOfWorkers || Xdimension % sqrtTasks != 0){
    	printf("ERROR: number of WORKERS must be a square number whose square root is perfectly divided by %d\n.	Total tasks must be #workers+1.\n", Xdimension);
		printf("Quitting...\n");
	    MPI_Abort(MPI_COMM_WORLD, rc);
	    exit(1);
	}  	
}

void restNeighbors(MPI_Comm myCom, int *myCoords, int sqrtTasks, Neighbor *neighbors){

	int temp[2], rank;
	int x = myCoords[0], y = myCoords[1];

	if(y != sqrtTasks-1 && x != 0){
		temp[0] = x-1; temp[1] = y+1;
		MPI_Cart_rank(myCom, temp, &rank);
		neighbors->northeast = rank;
	}
	else neighbors->northeast = MPI_PROC_NULL;


	if(y != 0 && x != 0){
		temp[0] = x-1; temp[1] = y-1;
		MPI_Cart_rank(myCom, temp, &rank);
		neighbors->northeast = rank;
	}
	else neighbors->northeast = MPI_PROC_NULL;


	if(y != 0 && x != sqrtTasks-1){
		temp[0] = x-1; temp[1] = y-1;
		MPI_Cart_rank(myCom, temp, &rank);
		neighbors->southwest = rank;
	}
	else neighbors->southwest = MPI_PROC_NULL;


	if(y != sqrtTasks-1 && x != sqrtTasks-1){
		temp[0] = x-1; temp[1] = y-1;
		MPI_Cart_rank(myCom, temp, &rank);
		neighbors->southwest = rank;
	}
	else neighbors->southwest = MPI_PROC_NULL;

}


/**************************************************************************
*  subroutine update
****************************************************************************/
void update(int start, int end, int ny, float *u1, float *u2)
{
	int ix, iy;
	for (ix = start; ix <= end; ix++) 
	  for (iy = 1; iy <= ny-2; iy++) 
	     *(u2+ix*ny+iy) = *(u1+ix*ny+iy)  + parms.cx * (*(u1+(ix+1)*ny+iy) + *(u1+(ix-1)*ny+iy) - 2.0 * *(u1+ix*ny+iy)) + parms.cy * (*(u1+ix*ny+iy+1) + *(u1+ix*ny+iy-1) - 2.0 * *(u1+ix*ny+iy));
}

/*****************************************************************************
*  subroutine inidat
*****************************************************************************/
void inidat(int nx, int ny, float *u) {
	int ix, iy;
	for (ix = 0; ix <= nx-1; ix++) 
	for (iy = 0; iy <= ny-1; iy++)
	 *(u+ix*ny+iy) = (float)(ix * (nx - ix - 1) * iy * (ny - iy - 1));
}

/**************************************************************************
* subroutine prtdat
**************************************************************************/
void prtdat(int nx, int ny, float *u1, char *fnam) {
	int ix, iy;
	FILE *fp;

	fp = fopen(fnam, "w");
	for (iy = ny-1; iy >= 0; iy--) {
	for (ix = 0; ix <= nx-1; ix++) {
	fprintf(fp, "%6.1f", *(u1+ix*ny+iy));
	if (ix != nx-1) 
	  fprintf(fp, " ");
	else
	  fprintf(fp, "\n");
	}
	}
	fclose(fp);
}
