#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <mpi.h>

#include "GameOfLife.h"

int main(int argc, char **argv)
{
	int numOftasks, numOfworkers, taskid;

	char myGrid[2][XDIMENSION][XDIMENSION];
	MPI_Comm cartesianComm, workersComm;

	MPI_Init(&argc, &argv);								//Initialization
	MPI_Comm_size(MPI_COMM_WORLD, &numOftasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	numOfworkers = numOftasks-1;

	MPI_Group workersGroup, initialGroup;
	MPI_Comm_group(MPI_COMM_WORLD, &initialGroup);

	int excludedRanks[1]={MASTER};
	MPI_Group_excl(initialGroup, 1, excludedRanks,&workersGroup);			//Exclude Master from new group
	MPI_Comm_create(MPI_COMM_WORLD, workersGroup, &workersComm);			//Creating a new communicator for workers

	int sqrtWorkers = (int)sqrt(numOfworkers);					//Distribute amount of work to workers
	int x = XDIMENSION/sqrtWorkers;							//An x*x subarray will be assigned to each task/worker



	if(taskid == MASTER){				//MASTER CODE

		validateInput(numOfworkers);
	
		printf("\nStarting the Game Of Life with %d worker tasks and 1 Master.\n", numOfworkers);
		printf("Grid size: X=%d, Y=%d  Generations=%d\n\n\n", XDIMENSION, XDIMENSION, GENERATIONS);

		/****** GET TIME ******/
		double time = MPI_Wtime();

		    /* Wait for results from all worker tasks */
/*		for (i=0; i<numOftasks; i++)
			MPI_Irecv(&myGrid[0][0][0], 1, myDatatype[i], i, DONE, cartesianComm, &requests[id]);
*/
		/****** GET TIME ******/
		time = MPI_Wtime() - time;

		/* Write final output, call X graph and finalize MPI */
/*		printf("Finished after %f seconds\n", time);
		printf("Writing final.dat file \n");
		prtdat(XDIMENSION, *(myGrid+1), "final.dat");

		//Free Memory
		for (id=0; id<numOftasks; id++)
			MPI_Type_free(&myDatatype[i]);

		free(myDatatype);
		free(requests);
*/
	}

	else {						//WORKER CODE
		
		srand(time(NULL));

		int cartid, reorder=1, i, j, id, myCoords[2], dims[2], periodic[2];
		dims[0] = dims[1] = sqrtWorkers;
		periodic[0] = periodic[1] = 1;							//Periodic Table - each cell has 8 neighbors
		MPI_Cart_create(workersComm, 2, dims, periodic, reorder, &cartesianComm);	//Make a new communicator with Cartesian Topology

		/*
		MPI_Comm_rank(cartesianComm, &id);
			MPI_Cart_coords(cartesianComm, id, 2, myCoords);
			printf("old Rank %d coordinates are %d %d - ", taskid, myCoords[0], myCoords[1]);
		MPI_Cart_rank(cartesianComm, myCoords, &id);
		printf("new rank is %d\n", id);
		fflush(stdout);
		*/

		//Each worker initializes its own subgrid randomly

		char **myGrids = malloc(2*sizeof(char*));
		for(i=0; i<2; i++)
			myGrids[i] = malloc(x*x*sizeof(char));	

		for(i=0; i<x*x; i++)
			myGrids[0][i] = (rand()%2 == 0)?DEAD:ALIVE;

		Neighborhood neighbors;

		//Get North, South, West, East neighbors' ids
		MPI_Cart_shift(cartesianComm, 0, 1, &neighbors.north, &neighbors.south);	//Get North and South "ids"
		MPI_Cart_shift(cartesianComm, 1, 1, &neighbors.west, &neighbors.east);		//Get West and East "ids"

		MPI_Comm_rank(cartesianComm, &cartid);						//Get new taskid of current process -- (tasks may have been reordered)
		MPI_Cart_coords(cartesianComm, cartid, 2, myCoords);

		//Get northWest, northEast, southWest, southEast neighbors' ids
		getRestNeighbors(cartesianComm, myCoords, sqrtWorkers, &neighbors);	

		//Datatypes
		MPI_Datatype leftCol, rightCol;
		int array_start[2];
		int array_sizes[] = {x, x};
		int array_subsizes[] = {x, 1};

		array_start[0] = 0; array_start[1] = 0;
		//Create a subarray datatype for left column
		MPI_Type_create_subarray(2, array_sizes, array_subsizes, array_start, MPI_ORDER_C, MPI_CHAR, &leftCol);
		MPI_Type_commit(&leftCol);

		array_start[0] = 0; array_start[1] = x-1;
		//Create a subarray datatype for right column
		MPI_Type_create_subarray(2, array_sizes, array_subsizes, array_start, MPI_ORDER_C, MPI_CHAR, &rightCol);
		MPI_Type_commit(&rightCol);

		MPI_Request **sendRequests = malloc(2*sizeof(MPI_Request*));
		for(i=0; i<2; i++){
			sendRequests[i] = malloc(8*sizeof(MPI_Request));

			//Create persistent send requests
			MPI_Send_init(myGrids[i], 1, leftCol, neighbors.west, TAG, cartesianComm, &sendRequests[i][0]);				//Send Left Column
			MPI_Send_init(myGrids[i], 1, rightCol, neighbors.east, TAG, cartesianComm, &sendRequests[i][1]);			//Send Right Column

			MPI_Send_init(myGrids[i], x, MPI_CHAR, neighbors.north, TAG, cartesianComm, &sendRequests[i][2]);			//Send Up Line
			MPI_Send_init(myGrids[i]+(x*(x-1)), x, MPI_CHAR, neighbors.south, TAG, cartesianComm, &sendRequests[i][3]);		//Send Down Line

			MPI_Send_init(myGrids[i], 1, MPI_CHAR, neighbors.northwest, TAG, cartesianComm, &sendRequests[i][4]);			//Send nortWest																				//Send northWest
			MPI_Send_init(myGrids[i]+x-1, 1, MPI_CHAR, neighbors.northeast, TAG, cartesianComm, &sendRequests[i][5]);		//Send northEast																//Send northEast
			MPI_Send_init(myGrids[i]+(x*(x-1)), 1, MPI_CHAR, neighbors.southwest, TAG, cartesianComm, &sendRequests[i][6]);		//Send southWest
			MPI_Send_init(myGrids[i]+(x*(x-1)+(x-1)), 1, MPI_CHAR, neighbors.southeast, TAG, cartesianComm, &sendRequests[i][7]);	//Send southEast
		}

		ReceiveBuffer buffer;
		buffer.north = malloc(x*sizeof(char));
		buffer.south = malloc(x*sizeof(char));
		buffer.east = malloc(x*sizeof(char));
		buffer.west = malloc(x*sizeof(char));

		MPI_Request *receiveRequests = malloc(8*sizeof(MPI_Request));	

		//Create persistent receive requests
		MPI_Recv_init(buffer.west, x, MPI_CHAR, neighbors.west, TAG, cartesianComm, &receiveRequests[0]);			//Receive Left Column
		MPI_Recv_init(buffer.east, x, MPI_CHAR, neighbors.east, TAG, cartesianComm, &receiveRequests[1]);			//Receive Right Column

		MPI_Recv_init(buffer.north, x, MPI_CHAR, neighbors.north, TAG, cartesianComm, &receiveRequests[2]);			//Receive Up Line
		MPI_Recv_init(buffer.south, x, MPI_CHAR, neighbors.south, TAG, cartesianComm, &receiveRequests[3]);			//Receive Down Line

		MPI_Recv_init(&buffer.northwest, 1, MPI_CHAR, neighbors.northwest, TAG, cartesianComm, &receiveRequests[4]);		//Receive northWest
		MPI_Recv_init(&buffer.northeast, 1, MPI_CHAR, neighbors.northeast, TAG, cartesianComm, &receiveRequests[5]);		//Receive northEast
		MPI_Recv_init(&buffer.southeast, 1, MPI_CHAR, neighbors.southeast, TAG, cartesianComm, &receiveRequests[6]);		//Receive southWest
		MPI_Recv_init(&buffer.southwest, 1, MPI_CHAR, neighbors.southwest, TAG, cartesianComm, &receiveRequests[7]);		//Receive southEast

		int round=0;
		MPI_Status *statuses = malloc(8*sizeof(MPI_Status));

		for(i=0; i<GENERATIONS; i++){
			round = i%2;
			
			for (j=0; j<8; j++) {
				MPI_Start(sendRequests[round]+j);		//Trigger send requests
				MPI_Start(receiveRequests + j);			//Trigger receive requests
			}

			nextGenerationInsideCells(myGrids[round], myGrids[round+1], x);

			for(j=0; j<8; j++)
				MPI_Wait(receiveRequests+j, &statuses[j]);
			//MPI_Waitall(8, sendRequests[round], MPI_STATUSES_IGNORE);
		
			nextGenerationOutsideCells(myGrids[round], myGrids[round+1], x);
    		}

/*
		for(i=0; i<2; i++) 
 			MPI_Request_free(requests[i]);
*/
		MPI_Comm_free(&cartesianComm);

	}

//MPI_Isend(&myGrid[0][0][0], 1, myDatatype[id], id, BEGIN, cartesianComm, NULL);								//Non blocking SEND


	MPI_Finalize();

	return 0;
}
