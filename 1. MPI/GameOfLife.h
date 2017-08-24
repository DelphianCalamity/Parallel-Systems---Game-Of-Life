#ifndef GAMEOFLIFE_H
#define GAMEOFLIFE_H

#include "mpi.h"

#define XDIMENSION	20						//We demand that the grid is square in order to simplify the problem, so only its one dimension needs to be defined
#define GENERATIONS 	1	                	//number of generations
#define MAXWORKER   	50                  	//maximum number of worker tasks
#define MINWORKER   	1                  		//minimum number of worker tasks
#define INITIAL    	0
#define FINAL     	1
#define TAG       	2
#define MASTER      	0
#define ALIVE		't'
#define DEAD 		'f'

typedef struct {
	int north;
	int south;
	int west;
	int east;
	int northeast;
	int northwest;
	int southeast;
	int southwest;
} Neighborhood;


typedef struct {
	char *north;
	char *south;
	char *west;
	char *east;
	char northeast;
	char northwest;
	char southeast;
	char southwest;
} ReceiveBuffer;

void validateInput(int numOfworkers);
void printData(int x, int sqrtWorkers, char ***subarraysptr, char* fnam);
void nextGenerationInsideCells(char *fromGrid, char *toGrid, int x);
void nextGenerationOutsideCells(char *fromGrid, char *toGrid, int x, ReceiveBuffer neighbors);
void getRestNeighbors(MPI_Comm cartesianComm, int *myCoords, int sqrtWorkers, Neighborhood *neighbors);

#endif
