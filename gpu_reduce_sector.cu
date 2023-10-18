// Reduce SUM ... improved version, 27th May 2013
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"

__global__ void float_sector_reduce_add(float *output, float *input, int num_sectors, int sector_size, int warpsize){
//reduction by sectors... every warp works on one sector

	int laneid = threadIdx.x % warpsize;
	int warpid = threadIdx.x / warpsize;
	int sid = blockIdx.x * (blockDim.x / warpsize) + warpid;
	int size = num_sectors * sector_size;
	
	int start_elem = sid * sector_size;
	int gid;
	float myval = 0.0f; //initialize properly depending on the reduce operation.
	for(int k = 0; (k * warpsize + laneid ) < sector_size; k++){ //we do a lot of “serial work”
		gid = start_elem + k * warpsize + laneid;
		if(gid < size) myval += input[gid];//load & compute.
	}
	//__syncthreads(); //always
	//warp reduction part
	//#pragma unroll 
	for(int mask = warpsize / 2 ; mask > 0 ; mask >>= 1) myval += __shfl_xor(myval, mask);
	
	
	if(laneid == 0)  output[sid] = myval;
}

__global__ void int_sector_reduce_add(int *output, int *input, int num_sectors, int sector_size, int warpsize){
//reduction by sectors... every warp works on one sector

	int laneid = threadIdx.x % warpsize;
	int warpid = threadIdx.x / warpsize;
	int sid = blockIdx.x * (blockDim.x / warpsize) + warpid;
	int size = num_sectors * sector_size;
	
	int start_elem = sid * sector_size;
	int gid;
	int myval = 0; //initialize properly depending on the reduce operation.
	for(int k = 0; (k * warpsize + laneid ) < sector_size; k++){ //we do a lot of “serial work”
		gid = start_elem + k * warpsize + laneid;
		if(gid < size) myval += input[gid];//load & compute.
	}
	//__syncthreads(); //always
	//warp reduction part
	//#pragma unroll 
	for(int mask = warpsize / 2 ; mask > 0 ; mask >>= 1) myval += __shfl_xor(myval, mask);
	
	
	if(laneid == 0)  output[sid] = myval;
}

__global__ void int_sector_reduce_min(int *output, int *input, int num_sectors, int sector_size, int warpsize){
//reduction by sectors... every warp works on one sector

	int laneid = threadIdx.x % warpsize;
	int warpid = threadIdx.x / warpsize;
	int sid = blockIdx.x * (blockDim.x / warpsize) + warpid;
	int size = num_sectors * sector_size;
	
	int start_elem = sid * sector_size;
	int gid;
	int myval = 2147483647; //initialize properly depending on the reduce operation.
	for(int k = 0; (k * warpsize + laneid ) < sector_size; k++){ //we do a lot of “serial work”
		gid = start_elem + k * warpsize + laneid;
		if(gid < size) myval = min(input[gid], myval);//load & compute.
	}
	//__syncthreads(); //always
	//warp reduction part
	//#pragma unroll 
	for(int mask = warpsize / 2 ; mask > 0 ; mask >>= 1) myval = min(myval,__shfl_xor(myval, mask));
	
	if(laneid == 0)  output[sid] = myval;
}

__global__ void int_sector_reduce_max(int *output, int *input, int num_sectors, int sector_size, int warpsize){
//reduction by sectors... every warp works on one sector

	int laneid = threadIdx.x % warpsize;
	int warpid = threadIdx.x / warpsize;
	int sid = blockIdx.x * (blockDim.x / warpsize) + warpid;
	int size = num_sectors * sector_size;
	
	int start_elem = sid * sector_size;
	int gid;
	int myval = -2147483648; //initialize properly depending on the reduce operation.
	for(int k = 0; (k * warpsize + laneid ) < sector_size; k++){ //we do a lot of “serial work”
		gid = start_elem + k * warpsize + laneid;
		if(gid < size) myval = max(input[gid], myval);//load & compute.
	}
	//__syncthreads(); //always
	//warp reduction part
	//#pragma unroll 
	for(int mask = warpsize / 2 ; mask > 0 ; mask >>= 1) myval = max(myval,__shfl_xor(myval, mask));
	
	if(laneid == 0)  output[sid] = myval;
}

void host_sum_sector_reduce(int *d_output, int *d_input, int num_sectors, int sector_size, int warpsize){
	//int warpsblock = 4;
	dim3 dimGrid= dim3(num_sectors,1,1);
	dim3 dimBlock= dim3(warpsize,1,1);
	
	int_sector_reduce_add<<<dimGrid, dimBlock>>>(d_output, d_input, num_sectors, sector_size, warpsize);

}

void host_sum_sector_reduce(float *d_output, float *d_input, int num_sectors, int sector_size, int warpsize){
	//int warpsblock = 4;
	dim3 dimGrid= dim3(num_sectors,1,1);
	dim3 dimBlock= dim3(warpsize,1,1);
	
	float_sector_reduce_add<<<dimGrid, dimBlock>>>(d_output, d_input, num_sectors, sector_size, warpsize);
}

int main(int argc, char **argv)
{
printf("Starting debbugging of functions in this .cu file\n");

//float *h_input_A, *h_input_B, *h_output;
int *h_input_A, *h_input_B, *h_output;

// cudaEvent_t gstart, gstop;
// cudaEventCreate(&gstart);
// cudaEventCreate(&gstop);

int sector_size = 29;
int num_sectors = 99;
int size = sector_size* num_sectors;
//int vec_size_bytes = size * sizeof(float);
int vec_size_bytes = size * sizeof(int);

//float *d_input_A, *d_input_B, *d_output;
int *d_input_A, *d_input_B, *d_output;

/*h_input_A = (float*)malloc(vec_size_bytes);
h_input_B = (float*)malloc(vec_size_bytes);
h_output = (float*)malloc(num_sectors*sizeof(float));*/

h_input_A = (int*)malloc(vec_size_bytes);
h_input_B = (int*)malloc(vec_size_bytes);
h_output = (int*)malloc(num_sectors*sizeof(int));


for(int i =0; i < size; i++){
	//h_input_A[i] = 1.0f;
	//h_input_B[i] = 2.0f;
	
	h_input_A[i] = 1;
	h_input_B[i] = 2;
}
//step 1
cudaMalloc((void**)&d_input_A, vec_size_bytes);
cudaMalloc((void**)&d_input_B, vec_size_bytes); 
//cudaMalloc((void**)&d_output, num_sectors*sizeof(float));
cudaMalloc((void**)&d_output, num_sectors*sizeof(int));

cudaMemcpy(d_input_A, h_input_A, vec_size_bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_input_B, h_input_B, vec_size_bytes, cudaMemcpyHostToDevice);
//cudaThreadSynchronize();
host_sum_sector_reduce(d_output, d_input_A, num_sectors, sector_size,32);
//cudaMemcpy(h_output, d_output, num_sectors*sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(h_output, d_output, num_sectors*sizeof(int), cudaMemcpyDeviceToHost);

//float result_gpu = 0.0f;
int result_gpu = 0;
for(int i = 0; i< num_sectors; i++) result_gpu += h_output[i];
/*printf("gpu sector result = %f\n", h_output[0]);
printf("gpu total result = %f\n", result_gpu);
printf("cpu total result = %f\n", size * 1.0);*/

printf("gpu sector result = %d\n", h_output[0]);
printf("gpu total result = %d\n", result_gpu);
printf("cpu total result = %d\n", size * 1);

//step 6
cudaFree(d_input_A);
cudaFree(d_input_B);
cudaFree(d_output);

free(h_input_A);
free(h_input_B);

//system("PAUSE");

return 0;

}