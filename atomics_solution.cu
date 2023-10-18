// nvcc -arch=sm_37 -run atomics_solution.cu cuda_helper.cu
#include "cuda_runtime.h"
#include "chTimer.h"
#include "cuda_helper.h"
#include <stdio.h>
#include <cfloat>
//adding some thrust headers
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

__global__ void initialize_int_array_value(int *array, int val, int array_size){
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid >= array_size) return;
  array[gid] = val;
}

__global__ void gpu_atomics_histogram(int *histogram, float *input, float range_max, float range_min, int num_bins, int vector_size)
{
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid >= vector_size)return;
  float dist = range_max - range_min;
  float raw_val = input[gid];
  float norm_value = (raw_val - range_min)/dist;
  int my_bin = floor(norm_value * num_bins);
  if(my_bin >= num_bins) my_bin = num_bins - 1;
  if(my_bin < 0) my_bin = 0;
  atomicAdd(&histogram[my_bin], 1);
}

float thrust_reduce_max(float *d_input_reduce, int vector_size)
{
   thrust::maximum<float> binary_op_max;
   return thrust::reduce(thrust::device,
                         d_input_reduce,
                         d_input_reduce + vector_size,
                         0,
                         binary_op_max);
}
float thrust_reduce_min(float *d_input_reduce, int vector_size)
{
   thrust::minimum<float> binary_op_min;
   return thrust::reduce(thrust::device,
                         d_input_reduce,
                         d_input_reduce + vector_size,
                         0,
                         binary_op_min);
}
/////////////////////////////////Serial version/////////////////////////////////////////////
void host_histogram(int *histogram, float *input, int num_bins, int vector_size)
{
    for(int i = 0; i < num_bins; i++) histogram[i] = 0;//initialize histogram
    float range_min = FLT_MAX;
    float range_max = -FLT_MAX;
    for(int i = 0; i < vector_size; i++){
        float val = input[i];
        range_min = (range_min < val)? range_min:val;
        range_max = (range_max > val)? range_max:val;
    }
    std::cout << "max: " << range_max << std::endl;
    std::cout << "min: " << range_min << std::endl;
    float dist = range_max - range_min;
    //compute the bins
    for(int i = 0; i < vector_size; i++){
      float raw_val = input[i];
      float norm_value = (raw_val - range_min)/dist;
      int my_bin = floor(norm_value * num_bins);
      if(my_bin >= num_bins)my_bin = num_bins - 1;
      histogram[my_bin] += 1;
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
cpuClock cpuck;
cudaClock ck;

float  *d_vec_in;
float *h_vec_in;
int *d_histogram;
int *h_histogram_gpu, *h_histogram_cpu;
printf("\nStarting program execution..\n\n");
int vec_size = 4 * 4096 * 4096;
int num_bins = 1000;

printf("Allocating and creating problem data..\n");

//allocation of host memory
h_histogram_cpu = (int*)malloc(num_bins * sizeof(int));
h_histogram_gpu = (int*)malloc(vec_size * sizeof(int));
h_vec_in = (float*)malloc(vec_size * sizeof(float));

for(int i =0; i < vec_size; i++){
	h_vec_in[i] = (rand()%1000) * 1.0f;
}
 ////
 //------ Step 1: Allocate the memory-------
 printf("Allocating Device Memory..\n");
CudaSafeCall(cudaMalloc((void**)&d_histogram,  num_bins * sizeof(int)));
CudaSafeCall(cudaMalloc((void**)&d_vec_in,     vec_size * sizeof(float)));
checkGPUMemory();
//------ Step 2: Copy Memory to the device-------
printf("Transfering data to the Device..\n");
CudaSafeCall(cudaMemcpy(d_vec_in, h_vec_in, vec_size * sizeof(float), cudaMemcpyHostToDevice));
//------ Step 3: Prepare launch parameters-------
//printf("preparing launch parameters..\n");
dim3 binsGrid = dim3((num_bins + 127)/128, 1, 1);
dim3 binsBlock = dim3(128, 1, 1);
dim3 dimGrid = dim3((vec_size + 127)/128, 1, 1);
dim3 dimBlock = dim3(128, 1, 1);
//WE DONT NEED TO CONFIGURE LAUNCH PARAMETERS TO USE THRUST
//------ Step 4: Launch device kernel-------
std::cout << std::endl << "--- prefix sum scan ---" << std::endl;
cudaTick(&ck);
//we have to launch four subroutines
//1) initialize the histogram array
initialize_int_array_value<<<binsGrid, binsBlock>>>(d_histogram, 0, num_bins);CudaCheckError();
//2) compute reduce max for range max
float range_max = thrust_reduce_max(d_vec_in, vec_size);CudaCheckError();
//3) compute reduce min for range min
float range_min = thrust_reduce_max(d_vec_in, vec_size);CudaCheckError();
std::cout << "max: " << range_max << std::endl;
std::cout << "min: " << range_min << std::endl;
//4) compute histogram
gpu_atomics_histogram<<<binsGrid, binsBlock>>>(d_histogram, d_vec_in, range_max, range_min, num_bins, vec_size);
cudaTock(&ck, "GPU Histogram Compute");
CudaCheckError();

//------ Step 5: Copy Memory back to the host-------
printf("Transfering result data to the Host..\n");
CudaSafeCall(cudaMemcpy(h_histogram_gpu, d_histogram, num_bins * sizeof(int), cudaMemcpyDeviceToHost));
 //
printf("CPU version...\n");
cpuTick(&cpuck);
host_histogram(h_histogram_cpu, h_vec_in, num_bins, vec_size);//serial version to compare
cpuTock(&cpuck, "host_histogram");

std::cout << "thrust version gpu is " << cpuck.elapsedMicroseconds/ck.elapsedMicroseconds << " times faster" << std::endl;
printf("Checking solutions..\n");
check_equal_int_vec(h_histogram_cpu, h_histogram_gpu, num_bins);

//-----------Step 6: Free the memory --------------
printf("Deallocating device memory..\n");
CudaSafeCall(cudaFree(d_histogram));
CudaSafeCall(cudaFree(d_vec_in));

free(h_vec_in);
free(h_histogram_cpu);
free(h_histogram_gpu);

return 0;
}
