// nvcc -arch=sm_61 -run test_inf.cu cuda_helper.cu
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

#define CUDART_INF_F            __int_as_float(0x7f800000)
#define CUDART_INF              __longlong_as_double(0x7ff0000000000000ULL)

__global__ void add_inf(float *output, float *array, int array_size){
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid >= array_size) return;
  output[gid] = array[gid] + CUDART_INF_F + CUDART_INF_F;
}
__global__ void add_inf_as_param(float *output, float *array, float imInf ,int array_size){
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid >= array_size) return;
  if(gid==0)printf("%.3f %.3f \n",imInf, array[gid] + imInf);
  output[gid] = array[gid] + imInf;

}


int main(int argc, char **argv)
{
  cudaDeviceReset();
 int size = 1024*1024;

//performance measure
cpuClock cpuck;
cudaClock ck;
float *h_array, *d_array, *d_output, *h_output_gpu;
CudaSafeCall(cudaMallocHost((void**)&h_array, size * sizeof(float)));
CudaSafeCall(cudaMallocHost((void**)&h_output_gpu, size * sizeof(float)));

for(int i =0; i < size; i++){
	h_array[i] = (rand()%1000) * 1.0f;
}
 ////
 //------ Step 1: Allocate the memory-------
// std::cout << "Allocating Device Memory"<< std::endl;
CudaSafeCall(cudaMalloc((void**)&d_array,     size * sizeof(float)));
CudaSafeCall(cudaMalloc((void**)&d_output,    size * sizeof(float)));

checkGPUMemory();
 //std::cout << "Please no more memory allocations beyond this point!!"<< std::endl;
 //------ Step 3..before 2?: Prepare launch parameters-------
 //printf("preparing launch parameters..\n");
 dim3 dimGrid = dim3((size + 127)/128, 1, 1);
 dim3 dimBlock = dim3(128, 1, 1);

//std::cout << "Transfering data to the Device, Compute and Transfer back!" << std::endl;
  //------ Step 2: Copy Memory to the device-------
  cudaMemcpy(d_array, h_array, size * sizeof(float), cudaMemcpyHostToDevice);
  //------ Step 4: Compute kernels-------
  add_inf_as_param<<<dimGrid, dimBlock>>>(d_output, d_array, std::numeric_limits<float>::infinity(), size);
 //------ Step 5: Copy Memory back to the host-------
  cudaMemcpy(h_output_gpu, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
CudaCheckError();

for(int i =0; i < 10; i++){
	std::cout << " " << h_output_gpu[i] << std::endl;
}

//-----------Step 6: Free the memory --------------
//printf("Deallocating device memory\n");
CudaSafeCall(cudaFree(d_array));
CudaSafeCall(cudaFree(d_output));
//important to free the arrays of pointers as well, small memory leaks can crash after few days of computation,
cudaFreeHost(h_array);
cudaFreeHost(h_output_gpu);

//return 42;
return 0;
}
