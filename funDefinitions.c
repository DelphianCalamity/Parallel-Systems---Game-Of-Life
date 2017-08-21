#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "GameOfLife.h"

/**************************************************************************
	.validateInput
****************************************************************************/

void validateInput(int numOfWorkers){

	//Simplifying the problem
	//Demanding that number of workers is a square number whose square root is perfectly divided by Grid's X & Y dimensions
	//Plus 1 task -> MASTER

	int sqrtWorkers = sqrt(numOfWorkers);
  	if ((numOfWorkers > MAXWORKER) || (numOfWorkers < MINWORKER) || sqrtWorkers*sqrtWorkers != numOfWorkers  || XDIMENSION%sqrtWorkers != 0) {
    	printf("\n\n ERROR: The number of tasks must be between %d and %d.\n 	number of WORKERS must be a square number whose square root is perfectly divided by %d.\n	Total #tasks must be #workers+1.\n", MINWORKER+1,MAXWORKER, XDIMENSION);
    	printf(" Quitting...\n\n");
	    MPI_Abort(MPI_COMM_WORLD, 0);
	    exit(1);
    }
}


/**************************************************************************
	.getRestNeighbors
****************************************************************************/

void getRestNeighbors(MPI_Comm cartesianComm, int *myCoords, int sqrtWorkers, Neighborhood *neighbors){

	int temp[2], rank;
	int x = myCoords[0], y = myCoords[1];

	if(y != sqrtWorkers-1 && x != 0){						//NorthEast
		temp[0] = x-1; temp[1] = y+1;
		MPI_Cart_rank(cartesianComm, temp, &rank);
		neighbors->northeast = rank;
	}
	else if(x==0 && y!=sqrtWorkers-1){
		temp[0] = sqrtWorkers-1; temp[1] = y+1;
		MPI_Cart_rank(cartesianComm, temp, &rank);
		neighbors->northeast = rank;
	}
	else if(x!=0 && y==sqrtWorkers-1){
		temp[0] = x-1; temp[1] = 0;
		MPI_Cart_rank(cartesianComm, temp, &rank);
		neighbors->northeast = rank;
	}
	else{
		temp[0] = sqrtWorkers-1; temp[1] = 0;
		MPI_Cart_rank(cartesianComm, temp, &rank);
		neighbors->northeast = rank;
	}

/************************************************/
	if(y!=0 && x!=0){										//NorthWest
		temp[0] = x-1; temp[1] = y-1;
		MPI_Cart_rank(cartesianComm, temp, &rank);
		neighbors->northwest = rank;
	}
	else if(y!=0 && x==0){
		temp[0] = sqrtWorkers-1; temp[1] = y-1;
		MPI_Cart_rank(cartesianComm, temp, &rank);
		neighbors->northwest = rank;
	}
	else if(y==0 && x!=0){
		temp[0] = x-1; temp[1] = sqrtWorkers-1;
		MPI_Cart_rank(cartesianComm, temp, &rank);
		neighbors->northwest = rank;
	}
	else{
		temp[0] = sqrtWorkers-1; temp[1] = sqrtWorkers-1;
		MPI_Cart_rank(cartesianComm, temp, &rank);
		neighbors->northwest = rank;
	}

/************************************************/			//SouthWest
	if(y!=0 && x!=sqrtWorkers-1){
		temp[0] = x+1; temp[1] = y-1;
		MPI_Cart_rank(cartesianComm, temp, &rank);
		neighbors->southwest = rank;
	}
	else if(y!=0 && x==sqrtWorkers-1){
		temp[0] = 0; temp[1] = y-1;
		MPI_Cart_rank(cartesianComm, temp, &rank);
		neighbors->southwest = rank;
	}
	else if(y==0 && x!=sqrtWorkers-1){
		temp[0] = x+1; temp[1] = sqrtWorkers-1;
		MPI_Cart_rank(cartesianComm, temp, &rank);
		neighbors->southwest = rank;
	}
	else {
		temp[0] = 0; temp[1] = sqrtWorkers-1;
		MPI_Cart_rank(cartesianComm, temp, &rank);
		neighbors->southwest = rank;
	}

/************************************************/			//SouthEast
	if(y!=sqrtWorkers-1 && x!=sqrtWorkers-1){
		temp[0] = x+1; temp[1] = y+1;
		MPI_Cart_rank(cartesianComm, temp, &rank);
		neighbors->southeast = rank;
	}
	else if(y!=sqrtWorkers-1 && x==sqrtWorkers-1){
		temp[0] = 0; temp[1] = y+1;
		MPI_Cart_rank(cartesianComm, temp, &rank);
		neighbors->southeast = rank;
	}
	else if(y==sqrtWorkers-1 && x!=sqrtWorkers-1){
		temp[0] = x+1; temp[1] = 0;
		MPI_Cart_rank(cartesianComm, temp, &rank);
		neighbors->southeast = rank;
	}
	else{
		temp[0] = 0; temp[1] = 0;
		MPI_Cart_rank(cartesianComm, temp, &rank);
		neighbors->southeast = rank;
	}
}


/**************************************************************************
	.nextGenerationInsideCells
****************************************************************************/
void nextGenerationInsideCells(char *fromGrid, char *toGrid, int x){

	/*Determine which of the cells from fromGrid array survive and which don't  according to the 
	 game of life rules. The image of the new generation will be stored in toGrid array

	 Important: function is called "InsideCells" cause you will determine the state of all the cells 
	 except those in the 'perimeter'. those cells need to receive the state of their neighbors from 
	 other processes. we ll determine the state of those cells in the function below (outsideCells) later
	*/


/*
	fromGrid and toGrid are two 2-dimensional arrays represented as 1-dimensional arrays.
	only way to pass a 2-dim array as an argument to a function and being able to manipulate it
	with the usual convenient way: array[i][j] is when you define that array with mallocs.

	char **array = malloc(..);
	for(i=0; i<X; i++)
		array[i] = malloc(..);

	... but that way the array won't be in continuous cells in memory and that will be a problem
		when creating the column subarray datatype (in mpi_GameOfLife.c).

	so the fromGrid,toGrid arrays must be considered as 2-dimensional
	and unfortunately must be manipulated with the non convenient way. 

	so when you want to access the element in position (2,3) you must 
	refer to it as : 2*x + 3*x  where x is the X and Y dimension-size (matrix is square).

	Use these definitions maybe :
		#define ALIVE		't'
		#define DEAD 		'f'


*/	int i,j;

	for(i=1; i<x-1; i++){
		for(j=1; j<x-1; j++){

		}
	}
}


/**************************************************************************
	.nextGenerationOutsideCells
****************************************************************************/
void nextGenerationOutsideCells(char *fromGrid, char *toGrid, int x){

	/*	EYAGGELIA */
}


/**************************************************************************
	.printData
**************************************************************************/
void printData(int nx, int ny, float *u1, char *fnam) {
/*	int ix, iy;
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
*/
}
