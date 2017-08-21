#ifndef GAMEOFLIFE_H
#define GAMEOFLIFE_H

#include "mpi.h"

#define XDIMENSION	5						//We demand that the grid is square in order to simplify the problem, so only its one dimension needs to be defined
#define STEPS       6000                	//number of time steps
#define MAXWORKER   50                  	//maximum number of worker tasks
#define MINWORKER   1                  		//minimum number of worker tasks
#define BEGIN       1
#define TAG       	2
#define DONE        4
#define MASTER      0
#define ALIVE		1
#define DEAD 		0

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

/*
typedef struct {
	MPI_s north;
	MPI_Datatype south;
	MPI_Datatype west;
	MPI_Datatype east;

} MyDatatype;
*/
/*
typedef struct {
	char northwest;
	char northeast;
	char southwest;
	char southeast;
} DiagonalNeighbor;
*/
void prtdat();
void nextGenerationInsideCells(char *fromGrid, char *toGrid, int x, int offset);
void nextGenerationOutsideCells(char *fromGrid, char *toGrid, int x, int offset);
void validateInput(int numOfworkers);
void getRestNeighbors(MPI_Comm cartesianComm, int *myCoords, int sqrtWorkers, Neighborhood *neighbors);

#endif
