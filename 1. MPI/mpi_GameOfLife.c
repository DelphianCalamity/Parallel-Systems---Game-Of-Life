#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <mpi.h>

#include "GameOfLife.h"

int main(int argc, char **argv)
{
	int numOftasks, i, j, numOfworkers, taskid;
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

	int sqrtWorkers = (int)sqrt(numOfworkers);					//Distribute amount of work to workers equally
	int x = XDIMENSION/sqrtWorkers;							//An x*x subarray will be assigned to each task/worker
	int subgrid_size = x*x*sizeof(char);



	if(taskid == MASTER){				//MASTER CODE

		validateInput(numOfworkers);

		printf("\nStarting the Game Of Life with %d worker tasks and 1 Master.\n", numOfworkers);
		printf("Grid size: X=%d, Y=%d, Generations=%d\n\n\n", XDIMENSION, XDIMENSION, GENERATIONS);

		MPI_Request *requests[2];
		char ***subarraysInOrder[2];
		char **subarrays[2];

		//Initialize data structures
		for(i=0; i<2; i++){
			requests[i] = malloc(numOfworkers*sizeof(MPI_Request));
			subarrays[i] = malloc(numOfworkers*sizeof(char*));
			subarraysInOrder[i] = malloc(sqrtWorkers*sizeof(char**));
			for(j=0; j<sqrtWorkers; j++)
				subarraysInOrder[i][j] = malloc(sqrtWorkers*sizeof(char*));
		}
		
		/********************************************************************************************************/
		/****** GET TIME ******/
		double time = MPI_Wtime();
		//Receive initial and final subgrids from workers - subarrays will not be in order
		for(i=0; i<2; i++){
			for(j=0; j<numOfworkers; j++){
				subarrays[i][j] = malloc(subgrid_size+2*sizeof(int));		//Allocate Space for 2 integers too- worker sends his coordinates too along with his subgrid
				MPI_Irecv(subarrays[i][j], subgrid_size+2*sizeof(int), MPI_CHAR, j+1, i, MPI_COMM_WORLD, &requests[i][j]);		//Tags: 0 for initial, 1 for final
			}
		}
		MPI_Waitall(numOfworkers, requests[0], MPI_STATUSES_IGNORE);
		MPI_Waitall(numOfworkers, requests[1], MPI_STATUSES_IGNORE);

		/****** GET TIME ******/
		time = MPI_Wtime() - time;
		/********************************************************************************************************/

		//Map subarraysInOrder's elements to their correspondant subarrays 	--This is a way to have the subarrays 'sorted' in linear time
		int coords[2]={0,0};
		for(i=0; i<2; i++){
			for(j=0; j<numOfworkers; j++){		
				memcpy(&coords[0], &subarrays[i][j][subgrid_size], sizeof(int));
				memcpy(&coords[1], &subarrays[i][j][subgrid_size+sizeof(int)], sizeof(int));
				subarraysInOrder[i][coords[0]][coords[1]] = subarrays[i][j];
			}
		}
		
		printf("Finished after %f seconds\n\n", time);
		
		//Writing initial grid
		printf("Writing initial.txt file.. \n");
		printData(x, sqrtWorkers, subarraysInOrder[0], "initial.txt");
		
		//Writing final grid
		printf("Writing final.txt file.. \n\n");
		printData(x, sqrtWorkers, subarraysInOrder[1], "final.txt");
		
		//Free Allocated Space
		for(i=0; i<2; i++){
			free(requests[i]);
			for(j=0; j<sqrtWorkers; j++)
				free(subarraysInOrder[i][j]);
			free(subarraysInOrder[i]);
			free(subarrays[i]);
		
		}
	}

	else {						//WORKER CODE

		int cartid, reorder=1, seed, id, myCoords[2], dims[2], periodic[2];
		MPI_Request request[2];

		dims[0] = dims[1] = sqrtWorkers;
		periodic[0] = periodic[1] = 1;							//Periodic Table - each cell has 8 neighbors
		MPI_Cart_create(workersComm, 2, dims, periodic, reorder, &cartesianComm);	//Make a new communicator with Cartesian Topology
		MPI_Comm_rank(cartesianComm, &cartid);						//Get new taskid of current process -- (tasks may have been reordered)
		MPI_Cart_coords(cartesianComm, cartid, 2, myCoords);

		seed = time(NULL)+cartid;
		srand(seed);

		/********************************************************************************************************/
		//Allocate space for two subgrids, the initial and the temporary one.
		//At the end of each grid we store the task's coords in order to send them to Master along with the main subgrid
		char **myGrids = malloc(2*sizeof(char*));		
		for(i=0; i<2; i++){
			myGrids[i] = malloc(subgrid_size + 2*sizeof(int));
			memcpy(&myGrids[i][subgrid_size], &myCoords[0], sizeof(int));
			memcpy(&myGrids[i][subgrid_size+sizeof(int)], &myCoords[1], sizeof(int));
		}

		//Each worker initializes his own subgrid randomly
		for(i=0; i<x*x; i++)
			myGrids[0][i] = (rand()%2 == 0)?DEAD:ALIVE;

		//Must send the initial grid to Master along with tha task's coordinates
		MPI_Isend(myGrids[0], subgrid_size+2*sizeof(int), MPI_CHAR, MASTER, INITIAL, MPI_COMM_WORLD, &request[0]);

		
		/********************************************************************************************************/
		//Get Neighbors
		Neighborhood neighbors;

		//Get North, South, West, East neighbors' ids
		MPI_Cart_shift(cartesianComm, 0, 1, &neighbors.north, &neighbors.south);	//Get North and South "ids"
		MPI_Cart_shift(cartesianComm, 1, 1, &neighbors.west, &neighbors.east);		//Get West and East "ids"

		//Get northWest, northEast, southWest, southEast neighbors' ids
		getRestNeighbors(cartesianComm, myCoords, sqrtWorkers, &neighbors);

		/********************************************************************************************************/
		
		//Create Datatypes
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

		MPI_Request *sendRequests[2];
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


		/********************************************************************************************************/
		int round=0, max;
		MPI_Status *statuses = malloc(8*sizeof(MPI_Status));

		for(i=0; i<GENERATIONS; i++) {
			round = i%2;

			for (j=0; j<8; j++) {
				MPI_Start(sendRequests[round]+j);		//Trigger send requests
				MPI_Start(receiveRequests + j);			//Trigger receive requests
			}

			nextGenerationInsideCells(myGrids[round], myGrids[2-(round+1)], x);

			MPI_Waitall(8, receiveRequests, MPI_STATUSES_IGNORE);

			//for(j=0; j<8; j++)	MPI_Wait(receiveRequests+j, &statuses[j]);

			nextGenerationOutsideCells(myGrids[round], myGrids[2-(round+1)], x, buffer);

			MPI_Waitall(8, sendRequests[0], MPI_STATUSES_IGNORE);
			MPI_Waitall(8, sendRequests[1], MPI_STATUSES_IGNORE);

			/**********************************************************************************************************/
			//-- COMMENT the block if you don't want to perform checks
/*
			if (i % FREQUENCY == 0) {				//Perform Reduce Checks every FREQUENCY times	

				max = isDifferent(myGrids[round], myGrids[2-(round+1)], x);
				MPI_Allreduce(MPI_IN_PLACE, &max, 1, MPI_INT, MPI_MAX, cartesianComm);
			
				if (max == 0) {
					printf("Stopping..\nEither cells are dead or generations remain the same\n");
					break;
				}
			}
*/
			/**********************************************************************************************************/
    		}


    		/********************************************************************************************************/
		//Must send the final grid to MASTER
		MPI_Isend(myGrids[2-(round+1)], subgrid_size+2*sizeof(int), MPI_CHAR, MASTER, FINAL, MPI_COMM_WORLD, &request[1]);

    		/********************************************************************************************************/

    		MPI_Waitall(2, request, MPI_STATUSES_IGNORE);

    		//Free Allocated Space
		MPI_Type_free(&leftCol);
		MPI_Type_free(&rightCol);
		
		for(i=0; i<2; i++){
			free(myGrids[i]);
			free(sendRequests[i]);
		}

		free(buffer.north);
		free(buffer.south);
		free(buffer.east);
		free(buffer.west);

		free(myGrids);
		free(receiveRequests);
		free(statuses);
		MPI_Comm_free(&cartesianComm);

	}

	MPI_Finalize();

	return 0;
}
