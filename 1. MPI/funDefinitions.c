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

	int i, j, k, aliveCounter=0;

	for(i=1; i<x-1; i++) {
		for(j=1; j<x-1; j++) {
			if(fromGrid[(i-1)*x+j]==ALIVE)        aliveCounter++;         // north neighbor
			if(fromGrid[i*x+(j+1)]==ALIVE)        aliveCounter++;         // east neighbor
			if(fromGrid[(i+1)*x+j]==ALIVE)        aliveCounter++;         // south neighbor
			if(fromGrid[i*x+(j-1)]==ALIVE)        aliveCounter++;         // west neighbor
			if(fromGrid[(i-1)*x+(j+1)]==ALIVE)    aliveCounter++;         // northeast neighbor
			if(fromGrid[(i+1)*x+(j+1)]==ALIVE)    aliveCounter++;         // southeast neighbor
			if(fromGrid[(i+1)*x+(j-1)]==ALIVE)    aliveCounter++;         // southwest neighbor
			if(fromGrid[(i-1)*x+(j-1)]==ALIVE)    aliveCounter++;         // northwest neighbor
			if(fromGrid[i*x+j]==ALIVE) {
				if(aliveCounter==2 || aliveCounter==3)          toGrid[i*x+j]=ALIVE;
				else            toGrid[i*x+j]=DEAD;
			}
			else {
				if(aliveCounter==3)             toGrid[i*x+j]=ALIVE;
				else            toGrid[i*x+j]=DEAD;
			}
			aliveCounter=0;
		}
	}
}



/**************************************************************************
	.nextGenerationOutsideCells
****************************************************************************/
void nextGenerationOutsideCells(char *fromGrid, char *toGrid, int x,  ReceiveBuffer neighbors){
	int i, j, k, aliveCounter=0;

	for(i=0; i<x; i++) {
		for(j=0; j<x; j++) {
			if(i==0 || i==x-1 || j==0 || j==x-1) {

				// north neighbor
				if(i-1<0) {
					if(neighbors.north[j]==ALIVE)			aliveCounter++;
				}
				else {
					if(fromGrid[(i-1)*x+j]==ALIVE)		aliveCounter++;
				}

				// east neighbor
				if(j+1>=x) {
					if(neighbors.east[i]==ALIVE)			aliveCounter++;
				}
				else {
					if(fromGrid[i*x+(j+1)]==ALIVE)		aliveCounter++;
				}

				// south neighbor
				if(i+1>=x) {
					if(neighbors.south[j]==ALIVE)			aliveCounter++;
				}
				else {
					if(fromGrid[(i+1)*x+j]==ALIVE)		aliveCounter++;
				}

				// west neighbor
				if(j-1<0) {
					if(neighbors.west[i]==ALIVE)			aliveCounter++;
				}
				else {
					if(fromGrid[i*x+(j-1)]==ALIVE)		aliveCounter++;
				}
			
				// northeast neighbor
				if(i==0 && j==x-1) {
					if(neighbors.northeast==ALIVE)		aliveCounter++;
				}
				else if(i==0 && j!=x-1) {
					if(neighbors.north[j+1]==ALIVE)		aliveCounter++;
				}
				else if(i==x-1 && j==x-1) {
					if(neighbors.east[i+1]==ALIVE)		aliveCounter++;
				}
				else {
					if(fromGrid[(i-1)*x+(j+1)]==ALIVE)	aliveCounter++;
				}

				// southeast neighbor
				if(i==0 && j==x-1) {
					if(neighbors.east[i+1]==ALIVE)		aliveCounter++;
				}
				else if(i==x-1 && j==x-1) {
					if(neighbors.southeast==ALIVE)		aliveCounter++;
				}
				else if(i==x-1 && j!=x-1) {
					if(neighbors.south[j+1]==ALIVE)		aliveCounter++;
				}
				else {
					if(fromGrid[(i+1)*x+(j+1)]==ALIVE)	aliveCounter++;
				}

				// southwest neighbor
				if(i==0 && j==0) {
					if(neighbors.west[i-1]==ALIVE)		aliveCounter++;
				}
				else if(i==x-1 && j==0) {
					if(neighbors.southwest==ALIVE)		aliveCounter++;
				}
				else if(i==x-1 && j!=0) {
					if(neighbors.south[j-1]==ALIVE)		aliveCounter++;
				}
				else {
					if(fromGrid[(i+1)*x+(j-1)]==ALIVE)	aliveCounter++;
				}

				// northwest neighbor
				if(i==0 && j==0) {
					if(neighbors.northwest==ALIVE)		aliveCounter++;
				}
				else if(i==0 && j!=0) {
					if(neighbors.north[j-1]==ALIVE)		aliveCounter++;
				}
				else if(i==x-1 && j==0) {
					if(neighbors.east[i-1]==ALIVE)		aliveCounter++;
				}
				else {
					if(fromGrid[(i-1)*x+(j-1)]==ALIVE)	aliveCounter++;
				}


				// Check if it will be alive or dead at next generation
				if(fromGrid[i*x+j]==ALIVE) {
					if(aliveCounter==2 || aliveCounter==3)		toGrid[i*x+j]=ALIVE;
					else		toGrid[i*x+j]=DEAD;
				}
				else {
					if(aliveCounter==3)		toGrid[i*x+j]=ALIVE;
					else		toGrid[i*x+j]=DEAD;
				}
				aliveCounter=0;
			}
		}
	}
}



/**************************************************************************
	.printData
**************************************************************************/
void printData(int x, int sqrtWorkers, char ***subarraysptr, char* fnam) {

	FILE *fp;
	int i, j, z, subline;
	char *subarray;

	fp = fopen(fnam, "w");

	for(i=0; i<sqrtWorkers; i++){					//subarray of task (i,j)
		for(subline=0; subline<x; subline++){		//subline line of subarray
			for(j=0; j<sqrtWorkers; j++){
				subarray = subarraysptr[i][j];
				for(z=0; z<x; z++){
					fprintf(fp, "%c ", subarray[subline*x+z]);	
				}
			}		
			fprintf(fp, "\n");
		}
	}
	fclose(fp);
}
