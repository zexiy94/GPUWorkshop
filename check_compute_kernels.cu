// nvcc -arch=sm_37 -run check_compute_kernels.cu cuda_helper.cu
#include "cuda_runtime.h"
#include "chTimer.h"
#include "cuda_helper.h"
#include <stdio.h>

__global__ void check_compute_kernels(float* dummy_out, int value, int size)
{
   int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid < size){
  //lost of computations
      float useless = 0.0f;
      for(int i = 0; i < size*size ; i++) useless += (float)(i % threadIdx.x);
  //feel free to add more...
    if(value < threadIdx.x)dummy_out[gid]=useless;//check what happens if you comment this line
 }
}


int main()
{
 cudaClock ck;
 float *d_dummy_out, *h_dummy_out;
 int val = 0;
 int size = 4096;
 for(int i = 0; i < 1023; i++)val++;
 h_dummy_out = (float*)malloc(size*sizeof(float));
 cudaMalloc((void**)&d_dummy_out,size*sizeof(float));
 dim3 dimGrid = dim3((size + 255)/256, 1, 1);//.... CONFIGURE THE GRID IN BLOCKS OF 256 THREADS BLOCKS
 dim3 dimBlock = dim3(256,1,1);



 cudaTick(&ck);
 check_compute_kernels<<<dimGrid, dimBlock>>>(d_dummy_out, val, 4096);
 cudaTock(&ck, "check_compute_kernels");

 CudaCheckError();
 cudaMemcpy(h_dummy_out, d_dummy_out, size*sizeof(float), cudaMemcpyDeviceToHost);
  return 42;
}
