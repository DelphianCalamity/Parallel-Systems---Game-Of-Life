#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mpi.h"

#define Xdimension  20						//We demand that the grid is square in order to simplify the problem, so only its one dimension needs to be defined
#define STEPS       600000                /* number of time steps */
#define MAXWORKER   50                  	//maximum number of worker tasks
#define MINWORKER   1                  		//minimum number of worker tasks
#define BEGIN       1 
#define DONE        4 
#define MASTER      0 

typedef struct {
	int north;
	int south;
	int west;
	int east;
	int eastnorth;
	int westnorth;
	int eastsouth;
	int westsouth;
} Neighborhood;

void inidat();
void prtdat();
void update();
void argsChecking(int numOfWorkers);
restNeighbors(MPI_Comm myCom, int *myCoords, int sqrtTasks, Neighbor *neighbors);


int main(int argc, char *argv[])
{
	int numOftasks, taskid;

	char myGrid[2][Xdimension][Xdimension];        /* array for grid */
	Neighborhood neighbors;
		
	//Initialization
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numOftasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

	//Make a new communicator with Cartesian topology
	MPI_Comm myCom;
	int myCoords[2];

	//Each process will determine its own neighbors
	int dims[2], periodic[2];
	dims[0] = dims[1] = sqrtTasks;
	periodic[0] = periodic[1] = 0;

	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 1, &myCom);
	MPI_Cart_shift(myCom, 0, 1, &neighbors.north, &neighbors.south);				//Get the ids of the tasks with whom the current process will need to exchange information
	MPI_Cart_shift(myCom, 1, 1, &neighbors.west, &neighbors.east);

	MPI_Comm_rank(myCom, &taskid);													//Get new taskid
	MPI_Cart_coords(myCom, taskid, 2, myCoords);
	
	restNeighbors(myCom, myCoords, sqrtTasks, &neighbors);							//Finds 'diagonal' neighbors

	/* Distribute work to workers. */
	int sqrtTasks = (int)sqrt(numtasks);
	int x = Xdimension/sqrtTasks;					//An x*x subarray will be assigned to each task/worker


	if (taskid == MASTER) {				//MASTER CODE

		argsChecking(numOftasks, Xdimension);

		printf ("Starting the Game Of Life with %d worker tasks.\n", numOftasks);

		/***** Initialize grid *****/
		printf("Grid size: X= %d  Y= %d  Time steps= %d\n", Xdimension, Xdimension, STEPS);
		printf("Initializing grid and writing initial.dat file...\n");
		inidat(Xdimension, *myGrid);
		prtdat(Xdimension, *myGrid, "initial.dat");


		//GET TIME
	    double time = MPI_Wtime();

	    /******* Send data to tasks *******/

		int id;
		int coords[2];
		MPI_Datatype *myDatatype = malloc(numOftasks*sizeof(MPI_Datatype));
		MPI_Request *requests = malloc(numOftasks*sizeof(MPI_Request));

		for (id=0; id<numOftasks; id++)
		{	
			MPI_Cart_coords(myCom, id, 2, coords);
			create_DataType(myCom, id, coords[0]*x, coords[1]*x, Xdimension, Xdimension, x, x){
			MPI_Isend(&myGrid[0][0][0], 1, myDatatype[id], id, BEGIN, myCom, NULL);
		}
	    
	    

	    /**********************************/

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

		for (id=0; id<numOftasks; id++)	
			MPI_Type_free(&myDatatype[i]);

		free(myDatatype);
		free(requests);
	}


	if (taskid != MASTER) 				//WORKER CODE

		workerJob();
	}

	MPI_Comm_free(&myCom);
	MPI_Finalize();

	return 0;
}


void create_DataType(MPI_Comm myCom, int id, int b1, int b2, int init1, int init2, int xs1, xs2){
	
	int temp[2], init[2], xs[2];

	xs[0] = xs1; xs[1] = x2;
	temp[0] = b1; temp[1] = b2;
	init[0] = init1; init[1] = init2;

	MPI_Type_create_subarray(2, init, xs, temp, MPI_ORDER_C, MPI_CHAR, &myDatatype[id]);
	MPI_Type_commit(&myDatatype[id]);
}


	void workerJob(int sqrtTasks, int x){
		
		char *grid = malloc(sqrtTasks*sqrtTasks*sizeof(char));

		MPI_Request request;

	    /******* Receive data from Master *******/

		MPI_Irecv(grid, x*x, MPI_CHAR, MASTER, BEGIN, myCom, request);

		MPI_Datatype *myDatatype = malloc(8*sizeof(MPI_Datatype));



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
		//Demanding that number of tasks is a square number whose square root is perfectly divided by Grid's X & Y dimensions
		int sqrtTasks = sqrt(numOfWorkers);
		if(sqrtTasks*sqrtTasks != numOfWorkers || Xdimension % sqrtTasks != 0){
	    	printf("ERROR: number of tasks must be a square number whose square root is perfectly divided by %d\n", Xdimension);
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
