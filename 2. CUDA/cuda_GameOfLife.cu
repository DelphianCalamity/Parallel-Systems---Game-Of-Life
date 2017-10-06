#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "cuda_runtime.h"

#include "device_launch_parameters.h"

#include "header.cuh"

#include "cuda_GameOfLife.cuh"


int main (int argc, char **argv) {

	/* for Broadcast */

	int width = XDIMENSION;
	int height = XDIMENSION;
	int loops = GENERATIONS;

	/* Initialize arrays */

	
	ReceiveBuffer buffer;
	buffer.north = malloc(x*sizeof(char));
	buffer.south = malloc(x*sizeof(char));
	buffer.east = malloc(x*sizeof(char));
	buffer.west = malloc(x*sizeof(char));
	
	char** t;

	char** t1;

	char* gpu_t;

	char* gpu_t1;

	char* gpu_temp;
	

	t = malloc(sizeof(char*)*width);

	t1 = malloc(sizeof(char*)*width);

	for(i = 0; i < width; i++){

		t[i] = malloc(sizeof(char)*height);

		t1[i] = malloc(sizeof(char)*height);

	}

	cudaMalloc(&gpu_t, width*height*sizeof(char));

	cudaMalloc(&gpu_t1, width*height*sizeof(char));


	int blocks = BLOCKS; 

	
	/* Initialization of t array */
   
	for(i=0; i<x*x; i++)
		t[i] = (rand()%2 == 0)?DEAD:ALIVE;

	cudaMemcpy(gpu_t, t, width*height, cudaMemcpyHostToDevice);

	cudaMemcpy(gpu_t1, t1, width*height, cudaMemcpyHostToDevice);



	/* Update the grid STEP times */

	for (timer = 0 ; timer < loops ; timer++){

		dim3 blockSize(blocks, blocks);

		dim3 dimGrid(width, heigt);

		nextGenerationInsideCells<<<blockSize, dimGrid>>>(gpu_t, gpu_t1, width, height);
		nextGenerationOutsideCells<<<blockSize, dimGrid>>>(gpu_t, gpu_t1, width, height, neighbors)

		/* swap arrays */

		gpu_temp = gpu_t;

		gpu_t = gpu_t1;

		gpu_t1 = gpu_temp;

	}



	cudaMemcpy(t, gpu_t, width*height, cudaMemcpyDeviceToHost);

	cudaMemcpy(t1, gpu_t1, width*height, cudaMemcpyDeviceToHost);



	/* Print final */
	
	printData();			// needs to be done
	
	
	/* Free allocated space */
	
	for(i = 0; i < width; i++){

		free(t[i]);

		free(t1[i]);

	}
	free(t);
	free(t1);
	cudaFree(gpu_t);
	cudaFree(gpu_t1);
	
	free(buffer.north);
	free(buffer.south);
	free(buffer.east);
	free(buffer.west);
	
	return 0;

}


