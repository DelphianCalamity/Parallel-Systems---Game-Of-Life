#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_GameOfLife.cuh"


/**************************************************************************
	.nextGenerationCells
****************************************************************************/
__global__ void nextGenerationCells(char *t, char *t1, int width, int height) {

	int i, j, k, aliveCounter=0;
	
	size_t x = blockIdx.x * blockIdDim.x + threadIdx.x; 

	if (0 < x && x < height-1 && 0 < x && x < width-1) {		// inside cells
		for(i=1; i<height-1; i++) {
			for(j=1; j<width-1; j++) {
				if(t[i-1][j]==ALIVE)      aliveCounter++;         // north neighbor
				if(t[i][j+1]==ALIVE)      aliveCounter++;         // east neighbor
				if(t[i+1][j]==ALIVE)      aliveCounter++;         // south neighbor
				if(t[i][j-1]==ALIVE)      aliveCounter++;         // west neighbor
				if(t[i-1][j+1]==ALIVE)    aliveCounter++;         // northeast neighbor
				if(t[i+1][j+1]==ALIVE)    aliveCounter++;         // southeast neighbor
				if(t[i+1][j-1]==ALIVE)    aliveCounter++;         // southwest neighbor
				if(t[i-1][j-1]==ALIVE)    aliveCounter++;         // northwest neighbor
				if(t[i][j]==ALIVE) {
					if(aliveCounter==2 || aliveCounter==3)          t1[i][j]=ALIVE;
					else            t1[i][j]=DEAD;
				}
				else {
					if(aliveCounter==3)             t1[i][j]=ALIVE;
					else            t1[i][j]=DEAD;
				}
				aliveCounter=0;
			}
		}
	}

	
	if ((0 == x || x == height-1) && (0 == x || width-1)) {		// outside cells
	for(i=0; i<height; i++) {
		for(j=0; j<width; j++) {
			if(i==0 || i==x-1 || j==0 || j==x-1) {

				// north neighbor
				if(i-1<0) {
					if(t[height-1][j]==ALIVE)		aliveCounter++;
				}
				else {
					if(t[i-1][j]==ALIVE)			aliveCounter++;
				}

				// east neighbor
				if(j+1>=x) {
					if(t[i][0]==ALIVE)			aliveCounter++;
				}
				else {
					if(t[i][j+1]==ALIVE)			aliveCounter++;
				}

				// south neighbor
				if(i+1>=x) {
					if(t[0][j]==ALIVE)			aliveCounter++;
				}
				else {
					if(t[i+1][j]==ALIVE)			aliveCounter++;
				}

				// west neighbor
				if(j-1<0) {
					if(t[i][width-1]==ALIVE)		aliveCounter++;
				}
				else {
					if(t[i][j-1]==ALIVE)			aliveCounter++;
				}
			
				// northeast neighbor
				if(i==0 && j==x-1) {
					if(t[height-1][0]==ALIVE)		aliveCounter++;
				}
				else if(i==0 && j!=x-1) {
					if(t[height-1][j+1]==ALIVE)		aliveCounter++;
				}
				else if(i!=0 && j==x-1) {
					if(t[i-1][0]==ALIVE)		aliveCounter++;
				}
				else {
					if(t[i-1][j+1]==ALIVE)			aliveCounter++;
				}

				// southeast neighbor
				if(i!=x-1 && j==x-1) {
					if(t[i+1][0]==ALIVE)		aliveCounter++;
				}
				else if(i==x-1 && j==x-1) {
					if(t[0][0]==ALIVE)		aliveCounter++;
				}
				else if(i==x-1 && j!=x-1) {
					if(t[0][j+1]==ALIVE)			aliveCounter++;
				}
				else {
					if(t[i+1][j+1]==ALIVE)			aliveCounter++;
				}

				// southwest neighbor
				if(i!=x-1 && j==0) {
					if(t[i+1][width-1]==ALIVE)		aliveCounter++;
				}
				else if(i==x-1 && j==0) {
					if(t[0][width-1]==ALIVE)		aliveCounter++;
				}
				else if(i==x-1 && j!=0) {
					if(t[0][j-1]==ALIVE)			aliveCounter++;
				}
				else {
					if(t[i+1][j-1]==ALIVE)			aliveCounter++;
				}

				// northwest neighbor
				if(i==0 && j==0) {
					if(t[height-1][width-1]==ALIVE)		aliveCounter++;
				}
				else if(i==0 && j!=0) {
					if(t[height-1][j-1]==ALIVE)		aliveCounter++;
				}
				else if(i!=0 && j==0) {
					if(t[i-1][width-1]==ALIVE)		aliveCounter++;
				}
				else {
					if(t[i-1][j-1]==ALIVE)			aliveCounter++;
				}


				// Check if it will be alive or dead at next generation
				if(t[i][j]==ALIVE) {
					if(aliveCounter==2 || aliveCounter==3)		toGrid[i][j]=ALIVE;
					else		t1[i][j]=DEAD;
				}
				else {
					if(aliveCounter==3)		t1[i][j]=ALIVE;
					else		t1[i][j]=DEAD;
				}
				aliveCounter=0;
			}
		}
	}
}


/**************************************************************************
	.isDifferent
**************************************************************************/
int isDifferent(char *t, char *t1, int height, int width) {

	int i, j;
	for(i=0; i<height; i++) {
		for(j=0; j<width; j++) {
			if(t[i][j] != t1[i][j])
				return 1;
	}

	return 0;
}


/**************************************************************************
	.printData
**************************************************************************/
void printData(int height, int width, char *t, char* fnam) {

	FILE *fp;
	int i, j, z, subline;
	char *subarray;

	fp = fopen(fnam, "w");

	for(i=0; i<height; i++) {
		for(j=0; j<width; j++) {
			fprintf(fp, "%c ", t[i][j]);	
		}		
			fprintf(fp, "\n");
	}
	fclose(fp);
}
