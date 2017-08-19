#include "mpi.h"

#define Xdimension  20						//We demand that the grid is square in order to simplify the problem, so only its one dimension needs to be defined
#define STEPS       600000                	/* number of time steps */
#define MAXWORKER   50                  	//maximum number of worker tasks
#define MINWORKER   1                  		//minimum number of worker tasks
#define BEGIN       1 
#define DONE        4 
#define MASTER      0 
#define ALIVE		1
#define DEAD 		0

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

typedef struct {
	MPI_Datatype north;
	MPI_Datatype south;
	MPI_Datatype west;
	MPI_Datatype east;
} MyDatatype;


typedef struct {
	char northwest;
	char northeast;
	char southwest;
	char southeast;
} DiagonalNeighbors;

void inidat();
void prtdat();
void update();
void argsChecking(int numOfWorkers);
void restNeighbors(MPI_Comm myCom, int *myCoords, int sqrtTasks, Neighbor *neighbors);
void workerJob(MPI_Comm myCom, int x);
