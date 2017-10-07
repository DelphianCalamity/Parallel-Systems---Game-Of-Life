#ifndef GAMEOFLIFE_H
#define GAMEOFLIFE_H


#define XDIMENSION	576				//We demand that the grid is square in order to simplify the problem, so only its one dimension needs to be defined
#define GENERATIONS 	100	                	//number of generations
#define INITIAL    	0
#define FINAL     	1
#define TAG       	2
#define MASTER      	0
#define ALIVE		't'
#define DEAD 		'f'
#define FREQUENCY 	5

#define THREADS_PER_BLOCK 8
#define BLOCKS 16


void printData(int height, int width, char *t, char* fnam);
void nextGenerationCells(char *t, char *t1, int height, int width);
int isDifferent(char *t, char *t1, int height, int width);


#endif
