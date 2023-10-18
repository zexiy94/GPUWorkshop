// nvcc -arch=sm_37 -run streams_exercise.cu cuda_helper.cu
#include "cuda_runtime.h"
#include "chTimer.h"
#include "cuda_helper.h"
#include <stdio.h>
#include <cfloat>

#define NSTREAMS 4

__global__ void any_kernel(float *output, float *arrayA, float* arrayB, int array_size){
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid >= array_size) return;
  output[gid] = arrayA[gid] * arrayB[gid];
}

void any_kernel_cpu(float *output, float *arrayA, float* arrayB, int array_size){
  for(int i = 0; i < array_size; i++)
  output[i] = arrayA[i] * arrayB[i];
}

/////////////////////////////////Diagnostic routines/////////////////////////////////////////////
int check_equal_float_vec(float *vec1,float *vec2,int size)
{
  	int numerrors = 0;
  	float dist;
  	float tolerance = 0.01f;
  	for(int i =0; i< size; i++){
  	    dist = (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
  		if(dist > tolerance) numerrors++;
  	}
  	if(numerrors ==0)printf("Congratulations you have 0 errors!\n");
  	if(numerrors >0)printf("Wrong results, you have %d errors!\n", numerrors);

  	return numerrors;
  }
  int check_equal_int_vec(int *vec1,int *vec2, int size){
  	int numerrors = 0;
  	for(int i =0; i< size; i++){
  		if(vec1[i] != vec2[i]) numerrors++;
  	}
  	if(numerrors ==0)printf("Congratulations you have 0 errors!\n");
  	if(numerrors >0)printf("Wrong results, you have %d errors!\n", numerrors);

  	return numerrors;
}
//////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	cudaDeviceReset();
	int size = 1024*1024;

	//performance measure
	cpuClock cpuck;
	cudaClock ck;
	printf("\nStarting program execution..\n\n");
	///////////////////////////////// STREAM CREATION //////////////////////////////
	std::cout << "Creating " << NSTREAMS << " streams" << std::endl;
	cudaStream_t *streams = (cudaStream_t *) malloc(NSTREAMS * sizeof(cudaStream_t));
	for(int is =0; is < NSTREAMS; is++){
		cudaStreamCreate(&(streams[is]));
	if(cudaSuccess != cudaGetLastError()) printf("Error creating the %d stream", is);
	}
	//IMPORTANT WE CREATE ARRAYs of POINTERS, to help maintain each stream data separated!!!
	float *h_inputA, *h_inputB, *h_output_gpu, *h_output_cpu;
	float **d_inputA = (float**)malloc(NSTREAMS * sizeof(float*));
	float **d_inputB = (float**)malloc(NSTREAMS * sizeof(float*));
	float **d_output = (float**)malloc(NSTREAMS * sizeof(float*));

	printf("Allocating and creating problem data..\n");

	//allocation of host memory
	h_inputA     = (float*)malloc(NSTREAMS * size * sizeof(float));
	h_inputB     = (float*)malloc(NSTREAMS * size * sizeof(float));
	h_output_gpu = (float*)malloc(NSTREAMS * size * sizeof(float));
	h_output_cpu = (float*)malloc(NSTREAMS * size * sizeof(float));

	for(int i =0; i < NSTREAMS * size; i++){
		h_inputA[i] = (rand()%1000) * 1.0f;
		h_inputB[i] = (rand()%1) * 1.0f;
	}
	////
	//------ Step 1: Allocate the memory-------
	std::cout << "Allocating Device Memory"<< std::endl;

	checkGPUMemory();
	std::cout << "Please no more memory allocations beyond this point!!"<< std::endl;
	//------ Step 3..before 2?: Prepare launch parameters-------
	//printf("preparing launch parameters..\n");
	dim3 dimGrid = dim3((size + 127)/128, 1, 1);
	dim3 dimBlock = dim3(128, 1, 1);

	std::cout << "Transfering data to the Device, Compute and Transfer back!" << std::endl;
	for(int is = 0 ; is < NSTREAMS; is++){
		//------ Step 2: Copy Memory to the device-------

		//------ Step 4: Compute kernels-------

		//------ Step 5: Copy Memory back to the host-------

	}
	CudaCheckError();
	//soem CPU verification
	any_kernel_cpu(h_output_cpu, h_inputA, h_inputB, NSTREAMS * size);
	check_equal_float_vec(h_output_gpu, h_output_cpu, NSTREAMS * size);

	//-----------Step 6: Free the memory --------------
	printf("Deallocating device memory\n");

	//important to free the arrays of pointers as well, small memory leaks can crash after few days of computation,
	free(d_inputA);
	free(d_inputB);
	free(d_output);
	free(h_inputA);
	free(h_inputB);
	free(h_output_gpu);
	free(h_output_cpu);

	return 42;
	}
