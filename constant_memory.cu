// nvcc -arch=sm_60 -run constant_memory.cu
#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>

float __constant__ FACTORS[256];

__global__ void kernel_vector_constant(float *result_vec, float *vec_a, int vector_size){
  ////// YOUR CODE GOES HERE!////////////////////////////////////////////
  //step 1.. define the thread ID
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  //step 2.. make sure the thread works whithin the array bounds
  if(gid >= vector_size) return;
	//step 3.. compute the result
  result_vec[gid] = vec_a[gid] * FACTORS[blockIdx.x];
}


//////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
//device
float *d_vec_out, *d_vec_a;
//host
float *h_vec_out_gpu, *h_vec_a, *h_factors;
printf("\nStarting program execution..\n\n");
int vec_size = 256*256;

printf("Allocating and creating problem data..\n");
int vec_size_bytes = vec_size * sizeof(float);
//allocation of host memory
h_vec_out_gpu = (float*)malloc(vec_size_bytes);
h_vec_a = (float*)malloc(vec_size_bytes);
h_factors = (float*)malloc(256*sizeof(float));

for(int i =0; i < vec_size; i++){
	h_vec_a[i] = 1.0f + (0.0001 * (i%256));
}
for(int i =0; i < 256; i++){
	h_factors[i] = i*1.0f;
}
 ////
 //------ Step 1: Allocate the memory-------
 printf("Allocating Device Memory..\n");
cudaMalloc((void**)&d_vec_out, vec_size_bytes);
cudaMalloc((void**)&d_vec_a, vec_size_bytes);

//------ Step 2: Copy Memory to the device-------
printf("Transfering data to the Device..\n");
cudaMemcpy(d_vec_a, h_vec_a, vec_size_bytes, cudaMemcpyHostToDevice);
printf("Transfering __constant__ data to the Device..\n");
cudaMemcpyToSymbol(FACTORS, h_factors, 256*sizeof(float));

//------ Step 3: Prepare launch parameters-------
printf("preparing launch parameters..\n");

dim3 dimGrid = dim3((vec_size + 255)/ 256,1,1);//.... CONFIGURE THE GRID IN BLOCKS OF 256 THREADS BLOCKS
dim3 dimBlock = dim3(256,1,1);
//------ Step 4: Launch device kernel-------
printf("Launch Device Kernel.\n");

// YOUR KERNEL LAUNCH GOES HERE------------------------>>>>>>>>>
kernel_vector_constant<<<dimGrid, dimBlock>>>(d_vec_out, d_vec_a, vec_size);

//------ Step 5: Copy Memory back to the host-------
printf("Transfering result data to the Host..\n");
cudaMemcpy(h_vec_out_gpu, d_vec_out, vec_size_bytes, cudaMemcpyDeviceToHost);

// Print values to check
for(int i = 0; i < 256; i++){ //print few per block
	std::cout << "block " << i << ": " <<  h_vec_out_gpu[i*256] << " " << h_vec_out_gpu[i*256 + 1] << " ... " << h_vec_out_gpu[i*256 + 254] << " " << h_vec_out_gpu[i*256 + 255] << std::endl; 
}

// -----------Step 6: Free the memory --------------
printf("Deallocating device memory..\n");
cudaFree(d_vec_out);
//.... FREE THE REST OF THE VECTORS
cudaFree(d_vec_a);

free(h_factors);
free(h_vec_out_gpu);

return 0;
}

// cudaEvent_t gstart, gstop;
// cudaEventCreate(&gstart);
// cudaEventCreate(&gstop);
