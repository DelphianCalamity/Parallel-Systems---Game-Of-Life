
/**************************************************************************
	.validateInput
****************************************************************************/

void validateInput(int numOfWorkers, int Xdimension){

	//Simplifying the problem
	//Demanding that number of workers is a square number whose square root is perfectly divided by Grid's X & Y dimensions
	//Plus 1 task -> MASTER

	int sqrtTasks = sqrt(numOfWorkers);
  	if ((numOfWorkers > MAXWORKER) || (numOfWorkers < MINWORKER) || sqrtTasks*sqrtTasks != numOfWorkers  || Xdimension % sqrtTasks != 0) {
    	printf("ERROR: The number of tasks must be between %d and %d.\n 	number of WORKERS must be a square number whose square root is perfectly divided by %d.\n	Total #tasks must be #workers+1.\n", MINWORKER+1,MAXWORKER+1, Xdimension);
    	printf("Quitting...\n");
	    MPI_Abort(MPI_COMM_WORLD, rc);
	    exit(1);
    }
}

/**************************************************************************
	.nextGeneration
****************************************************************************/
void nextGeneration(int start, int end, int ny, float *u1, float *u2)
{
	int ix, iy;
	for (ix = start; ix <= end; ix++) 
	  for (iy = 1; iy <= ny-2; iy++) 
	     *(u2+ix*ny+iy) = *(u1+ix*ny+iy)  + parms.cx * (*(u1+(ix+1)*ny+iy) + *(u1+(ix-1)*ny+iy) - 2.0 * *(u1+ix*ny+iy)) + parms.cy * (*(u1+ix*ny+iy+1) + *(u1+ix*ny+iy-1) - 2.0 * *(u1+ix*ny+iy));
}

/*****************************************************************************
	.initializeData
*****************************************************************************/
void initializeData(int nx, int ny, float *u) {
	int ix, iy;
	for (ix = 0; ix <= nx-1; ix++) 
	for (iy = 0; iy <= ny-1; iy++)
	 *(u+ix*ny+iy) = (float)(ix * (nx - ix - 1) * iy * (ny - iy - 1));
}

/**************************************************************************
	.printData
**************************************************************************/
void printData(int nx, int ny, float *u1, char *fnam) {
	int ix, iy;
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
}
