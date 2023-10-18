// nvcc -arch=sm_61 -run shuffle_exercise.cu cuda_helper.cu
#include "cuda_runtime.h"
#include "chTimer.h"
#include "cuda_helper.h"
#include <stdio.h>
#define WARP_SIZE 32

/*The objective of this exercise is to create a kernel, using shuffle instructions
to compute the average of many vectors. Can you think of another version using atomics?*/

__global__ void kernel_multi_naive_avg(float *vec_avg, float *vec_in, int vector_size, int number_of_vectors){
  int myVector = threadIdx.x + blockIdx.x * blockDim.x;
  if(myVector >= number_of_vectors)return;
  float mySum = 0.0f;
  for(int lid = 0 ; lid < vector_size; lid++) mySum +=  vec_in[myVector * vector_size + lid];
  vec_avg[myVector] = mySum / vector_size;
}

__global__ void kernel_multi_avg_shuffle(float *vec_avg, float *vec_in, int vector_size, int number_of_vectors){
  ////// YOUR CODE GOES HERE!////////////////////////////////////////////
  //step 1.. define lane and vector

  //step 2.. loop collecting values

	//step 3.. reduce in warp
  for (int i=16; i>0; i=i/2) mySum += __shfl_down(mySum, i);
  //step 4 let the first warp write the value
 if(myLane == 0) vec_avg[myVector] = mySum / vector_size;
}
/////////////////////////////////Serial version/////////////////////////////////////////////
void host_vector_multi_avg(float *vec_avg, float *vec_in, int vector_size, int number_of_vectors){

  for(int vector = 0 ; vector < number_of_vectors; vector++){
     float sum = 0.0f;
    	for(int elem = 0; elem < vector_size; elem++){
    		  sum += vec_in[vector * vector_size + elem];
    	}
      vec_avg[vector] = sum / vector_size;
  }
}
/////////////////////////////////Diagnostic routines/////////////////////////////////////////////
int check_equal_float_vec(float *vec1,float *vec2,int size){
	int numerrors = 0;
	float dist;
	float tolerance = 0.0001f;
	for(int i =0; i< size; i++){
	    dist = (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
		if(dist > tolerance) numerrors++;
	}
	if(numerrors ==0)printf("Congratulations you have 0 errors!\n");
	if(numerrors >0)printf("Wrong results, you have %d errors!\n", numerrors);

	return numerrors;
}
//////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
cpuClock cpuck;
cudaClock ck1, ck2;
float *d_vec_avg, *d_vec_avg2, *d_vec_in;
float *h_vec_avg_cpu, *h_vec_avg_gpu, *h_vec_avg_gpu2, *h_vec_in;
printf("\nStarting program execution..\n\n");
int vec_size = 512;
int number_of_vectors = 16192;

printf("\nAllocating and creating problem data..\n");

//allocation of host memory
h_vec_avg_cpu = (float*)malloc(number_of_vectors * sizeof(float));
h_vec_avg_gpu = (float*)malloc(number_of_vectors * sizeof(float));
h_vec_avg_gpu2 = (float*)malloc(number_of_vectors * sizeof(float));
h_vec_in = (float*)malloc(number_of_vectors * vec_size * sizeof(float));

for(int i =0; i < number_of_vectors * vec_size; i++){
	h_vec_in[i] = (rand()%10)/10.0f;
}
 ////
 //------ Step 1: Allocate the memory-------
 printf("\nAllocating Device Memory\n");
CudaSafeCall(cudaMalloc((void**)&d_vec_avg,  number_of_vectors * sizeof(float)));
CudaSafeCall(cudaMalloc((void**)&d_vec_avg2, number_of_vectors * sizeof(float)));
CudaSafeCall(cudaMalloc((void**)&d_vec_in,   number_of_vectors * vec_size* sizeof(float)));

//------ Step 2: Copy Memory to the device-------
printf("\nTransfering data to the Device..\n");
CudaSafeCall(cudaMemcpy(d_vec_in, h_vec_in, number_of_vectors * vec_size * sizeof(float), cudaMemcpyHostToDevice));
CudaCheckError();
//------ Step 3: Prepare launch parameters-------
printf("\npreparing launch parameters..\n");
//naive launch params
int tpbx = 128;
dim3 dimGrid = dim3((number_of_vectors + tpbx - 1)/tpbx, 1, 1);//.... CONFIGURE THE GRID IN BLOCKS OF 256 THREADS BLOCKS
dim3 dimBlock = dim3(tpbx,1,1);
//shuffle launch params
int tpby = 4;
dim3 shuffleGrid = dim3(1, (number_of_vectors + tpby - 1) / tpby, 1);//.... CONFIGURE THE GRID IN BLOCKS OF 256 THREADS BLOCKS
dim3 shuffleBlock = dim3(32,tpby,1);
//------ Step 4: Launch device kernel-------
printf("\nLaunch Device Kernels.\n");

cudaTick(&ck1);
kernel_multi_naive_avg<<<dimGrid, dimBlock>>>(d_vec_avg,
                                              d_vec_in,
                                              vec_size,
                                              number_of_vectors);
cudaTock(&ck1, "kernel_multi_naive_avg");
CudaCheckError();
// YOUR KERNEL LAUNCH GOES HERE------------------------>>>>>>>>>
cudaTick(&ck2);
kernel_multi_avg_shuffle<<<shuffleGrid, shuffleBlock>>>(d_vec_avg2,
                                                        d_vec_in,
                                                        vec_size,
                                                        number_of_vectors);
cudaTock(&ck2, "kernel_vector_avg_shuffle");
CudaCheckError();
std::cout << "shuffle version is " << ck1.elapsedMicroseconds/ck2.elapsedMicroseconds << " times faster" << std::endl;
std::cout << std::endl;
//------ Step 5: Copy Memory back to the host-------
printf("\nTransfering result data to the Host..\n");
CudaSafeCall(cudaMemcpy(h_vec_avg_gpu,  d_vec_avg, number_of_vectors * sizeof(float), cudaMemcpyDeviceToHost));
CudaSafeCall(cudaMemcpy(h_vec_avg_gpu2, d_vec_avg2, number_of_vectors * sizeof(float), cudaMemcpyDeviceToHost));
//
printf("\nCPU version...\n");
cpuTick(&cpuck);
host_vector_multi_avg(h_vec_avg_cpu, h_vec_in, vec_size, number_of_vectors);//serial version to compare
cpuTock(&cpuck, "host_vector_multi_avg");
std::cout << "the best gpu is " << cpuck.elapsedMicroseconds/ck2.elapsedMicroseconds << " times faster" << std::endl;
printf("\nChecking solutions..\n");
check_equal_float_vec(h_vec_avg_gpu, h_vec_avg_cpu, number_of_vectors);
check_equal_float_vec(h_vec_avg_gpu2, h_vec_avg_cpu, number_of_vectors);
check_equal_float_vec(h_vec_avg_gpu2, h_vec_avg_gpu, number_of_vectors);
// -----------Step 6: Free the memory --------------
printf("Deallocating device memory..\n");
cudaFree(d_vec_in);
cudaFree(d_vec_avg);
cudaFree(d_vec_avg2);

free(h_vec_in);
free(h_vec_avg_cpu);
free(h_vec_avg_gpu);
free(h_vec_avg_gpu2);

return 42;
}
