#include <stdio.h>
#include <stdlib.h>
#include "curand_kernel.h"
//#include <windows.h>
#define SIZE 128*128
#define MAX_STATE_SIZE 256
#define BLOCK_SIZE 256

__global__ void setup_kernel (curandStateMRG32k3a* state, unsigned long seed )//(curandState * state, unsigned long seed )

{
    int gid = threadIdx.x%MAX_STATE_SIZE; //+ blockIdx.x * blockDim.x;
    curand_init ( seed, gid, 0, &state[gid] );
} 

__global__ void simple_setup(curandStateMRG32k3a* state, int size )// (curandState * state, int size )
{
    int gid = threadIdx.x%MAX_STATE_SIZE;//+ blockIdx.x * blockDim.x;
    if(gid < size)curand_init ( 1937, gid, 0, &state[gid] );
} 

__global__ void generate( float* randArray, curandStateMRG32k3a* globalState, int size) //( float* randArray, curandState* globalState, int size) 
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if(gid < size){
		int chosenState = (blockIdx.x * 11 * threadIdx.x)%gridDim.x ;
		//int chosenState = gid;
		curandStateMRG32k3a localState = globalState[chosenState];
		randArray[gid] = curand_uniform( &localState );
		//saving back the state
		//globalState[chosenState] = localState;
	}
}

__global__ void generateHQ( float* randArray, curandStateMRG32k3a* globalState, int size) //( float* randArray, curandState* globalState, int size) 
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if(gid < size){
		//int chosenState = (blockIdx.x * 11 * threadIdx.x)%gridDim.x ;
		int chosenState = gid;
		curandStateMRG32k3a localState = globalState[chosenState];
		randArray[gid] = curand_uniform( &localState );
		//saving back the state
		globalState[chosenState] = localState;//every thread needs its own state...
	}
}


__host__ void full_rand_setup(float* randArray, int rand_size){
	curandStateMRG32k3a *allStates;
	cudaMalloc((void**) &allStates, rand_size*sizeof(curandStateMRG32k3a));
	simple_setup<<<(rand_size+255)/256,256>>>(allStates, rand_size);
	generate<<<(rand_size+255)/256,256>>>(randArray, allStates, rand_size);
	cudaFree(allStates);
}
int main(int argc,char **argv)
{

 printf("Starting the execution!\n");
 printf("Size of curandStateMRG32k3a state: %d\n", sizeof(curandStateMRG32k3a));

 curandStateMRG32k3a *d_states; 
 float *d_randArray, *d_output, *h_randArray;
 
 h_randArray=(float*)malloc(SIZE * sizeof(float));

 cudaEvent_t gstart,gstop;
 cudaEventCreate(&gstart);
 cudaEventCreate(&gstop); 

 int stateSize = MAX_STATE_SIZE;//((SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE);
/////////////////// STEP 1 - ALLOCATING DEVICE MEMORY /////////////////
 printf("Allocating device memory!\n");
 cudaMalloc((void**) &d_states, stateSize * sizeof(curandStateMRG32k3a));
 cudaMalloc((void**) &d_randArray, SIZE * sizeof(float));
 cudaMalloc((void**) &d_output, SIZE * sizeof(float));
 
 
//////////// STEP 3 - SETTING UP LAUNCH PARAMETERS /////////////////////
 dim3 dimGrid = dim3((SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE,1, 1);
 dim3 stateGrid = dim3((stateSize + BLOCK_SIZE - 1)/BLOCK_SIZE,1, 1);
 dim3 dimBlock = dim3(BLOCK_SIZE,1,1);
 printf("starting the calculation!\n");
 
 
 cudaEventRecord(gstart, 0);
////////////////// STEP 4 - LAUNCH THE KERNEL //////////////////////////
//printf("setting up the states\n");
 simple_setup<<<stateGrid, dimBlock>>>(d_states,stateSize);
 generate<<<dimGrid, dimBlock>>>(d_randArray, d_states,SIZE);
 
 cudaEventRecord(gstop, 0);
 cudaEventSynchronize(gstop); 
 float gpu_time; 
 cudaEventElapsedTime(&gpu_time, gstart, gstop);
 printf("GPU version has finished, it took %f ms\n",gpu_time );
//copy memory to print
cudaMemcpy( h_randArray, d_randArray, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
for(int i =0; i< SIZE; i++){
 	if(i%10 == 0) printf("\n");
	printf(" %f",  h_randArray[i]);
 }
  printf("\n");
///////////////// STEP 6 - DEALOCATE DEVICE MEMORY /////////////////////
 printf("Deallocating GPU memory \n");
 cudaFree(d_states);
 cudaFree(d_randArray);
 cudaFree(d_output); 
 
 free( h_randArray);

 
 cudaEventDestroy(gstart); //cleaning up a bit
 cudaEventDestroy(gstop);
 
 printf("That's all folks!\n");

 return 0;
}