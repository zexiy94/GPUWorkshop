// nvcc -arch=sm_37 -run reduce_solution.cu cuda_helper.cu
#include "cuda_runtime.h"
#include "chTimer.h"
#include "cuda_helper.h"
#include <stdio.h>
//adding some thrust headers
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

void thrust_exclusive_prefix_sum_scan(int* d_output_prefix, int*d_input_prefix, int vector_size)
{
  thrust::plus<int> binary_op_plus;
  thrust::exclusive_scan(thrust::device,
                         d_input_prefix,
                         d_input_prefix + vector_size, //N ELEMENTS
                         d_output_prefix,
                         0,
                         binary_op_plus);
}

float thrust_reduce_sum(float *d_input_reduce, int vector_size)
{
   thrust::plus<float> binary_op_plus;
   return thrust::reduce(thrust::device,
                         d_input_reduce,
                         d_input_reduce + vector_size,
                         0,
                         binary_op_plus);
}
/////////////////////////////////Serial version/////////////////////////////////////////////
float sum_reduce(float* vector, int vector_size)
{
    float sum = 0.0f;
    for (int i=0; i< vector_size; i++)sum += vector[i];
    return sum;
}

void exclusivePrefixSumScanCPU(int *output, int *input, int vector_size)
{
    output[0] = 0;
    for(int i = 1; i < vector_size; i++){
        output[i] = input[i-1] + output[i-1];
    }
}
/////////////////////////////////Diagnostic routines/////////////////////////////////////////////
int check_equal_float_vec(float *vec1,float *vec2,int size){
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
//performance measure
cpuClock cpuck, cpuck2;
cudaClock ck, ck2;

float  *d_vec_in;
float *h_vec_in;
int *d_prefix, *d_count;
int *h_prefix_gpu, *h_prefix_cpu, *h_count;
printf("\nStarting program execution..\n\n");
int vec_size = 4 * 4096 * 4096;

printf("Allocating and creating problem data..\n");

//allocation of host memory
h_prefix_cpu = (int*)malloc(vec_size * sizeof(int));
h_prefix_gpu = (int*)malloc(vec_size * sizeof(int));
h_count = (int*)malloc(vec_size * sizeof(int));
h_vec_in = (float*)malloc(vec_size * sizeof(float));

for(int i =0; i < vec_size; i++){
	h_vec_in[i] = 1.0f;
  h_count[i] = rand()%10;
}
 ////
 //------ Step 1: Allocate the memory-------
 printf("Allocating Device Memory..\n");
CudaSafeCall(cudaMalloc((void**)&d_prefix, vec_size * sizeof(int)));
CudaSafeCall(cudaMalloc((void**)&d_count,  vec_size * sizeof(int)));
CudaSafeCall(cudaMalloc((void**)&d_vec_in, vec_size * sizeof(float)));
checkGPUMemory();
//------ Step 2: Copy Memory to the device-------
printf("Transfering data to the Device..\n");
CudaSafeCall(cudaMemcpy(d_vec_in, h_vec_in, vec_size * sizeof(float), cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpy(d_count,  h_count,  vec_size * sizeof(int),   cudaMemcpyHostToDevice));
//------ Step 3: Prepare launch parameters-------
//printf("preparing launch parameters..\n");
//WE DONT NEED TO CONFIGURE LAUNCH PARAMETERS TO USE THRUST
//------ Step 4: Launch device kernel-------
std::cout << std::endl << "--- prefix sum scan ---" << std::endl;
cudaTick(&ck);

cudaTock(&ck, "Thrust Reduce Sum");
CudaCheckError();

//------ Step 5: Copy Memory back to the host-------
printf("Transfering result data to the Host..\n");
CudaSafeCall(cudaMemcpy(h_prefix_gpu, d_prefix, vec_size * sizeof(int), cudaMemcpyDeviceToHost));
 //
printf("CPU version...\n");
cpuTick(&cpuck);
exclusivePrefixSumScanCPU(h_prefix_cpu, h_count, vec_size);//serial version to compare
cpuTock(&cpuck, "exclusivePrefixSumScanCPU");

std::cout << "thrust version gpu is " << cpuck.elapsedMicroseconds/ck.elapsedMicroseconds << " times faster" << std::endl;
printf("Checking solutions..\n");
check_equal_int_vec(h_prefix_cpu, h_prefix_gpu, vec_size);

std::cout << std::endl << "--- reduce sum ---" << std::endl;
cudaTick(&ck2);
float gpu_sum = ;
cudaTock(&ck2, "Thrust Reduce Sum");
CudaCheckError();

cpuTick(&cpuck2);
float cpu_sum = sum_reduce(h_vec_in, vec_size);//serial version to compare
cpuTock(&cpuck2, "sum_reduce");
std::cout << "thrust version gpu is " << cpuck2.elapsedMicroseconds/ck2.elapsedMicroseconds << " times faster" << std::endl;
std::cout <<"We added " << vec_size << " of 1s!" << std::endl;
std::cout <<"Result in the GPU: Sum = " << gpu_sum << std::endl;
std::cout <<"Result in the CPU: Sum = " << cpu_sum << std::endl;
//

// -----------Step 6: Free the memory --------------
printf("Deallocating device memory..\n");
CudaSafeCall(cudaFree(d_prefix));
CudaSafeCall(cudaFree(d_count));
CudaSafeCall(cudaFree(d_vec_in));

free(h_vec_in);
free(h_prefix_cpu);
free(h_prefix_gpu);
free(h_count);

return 0;
}
