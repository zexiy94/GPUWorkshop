// Reduce SUM ... improved version, 27th May 2013
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include "float.h"
#include "device_launch_parameters.h"

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void init_float_vec(float *vec, int size){
	//printf("Hi there");
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if(gid < size)vec[gid] = 0.0f;
}

__global__ void float_reduce_add(float *output, float *input, int size){
	__shared__ float s_data[512]; //shared memory
	float dstat;
	int chunk_size = ceil( (1.0f * size)/gridDim.x);
	int start_elem = blockIdx.x * chunk_size;
	int tid = threadIdx.x;
	int gid;
	s_data[tid] = 0.0f; //initialize properly depending on the reduce operation.
	for(int k = 0; (k * blockDim.x + tid ) < chunk_size; k++){ //we do a lot of “serial work”
		gid = start_elem + k * blockDim.x + tid;
		if(gid < size) s_data[tid] += input[gid];//load & compute.
	}
	__syncthreads(); //always
	for(int s = blockDim.x/2; s > 0 ;s/=2){ //using s>>=2 is faster
		if(tid<s) s_data[tid] += s_data[tid + s]; //no thread divergence until last 5 iter.
		__syncthreads(); //again.
	}
	if(tid == 0)  dstat= atomicAdd(&output[0],s_data[0]);
}

__global__ void float_reduce_min(float *output, float *input, int size){
	__shared__ float s_data[512]; //shared memory
	float dstat;
	int chunk_size = ceil( (1.0f * size)/gridDim.x);
	int start_elem = blockIdx.x * chunk_size;
	int tid = threadIdx.x;
	int gid;
	s_data[tid] = FLT_MAX; //initialize properly depending on the reduce operation.
	for(int k = 0; (k * blockDim.x + tid ) < chunk_size; k++){ //we do a lot of “serial work”
		gid = start_elem + k * blockDim.x + tid;
		if(gid < size) 
			if(input[gid] < s_data[tid]) s_data[tid] = input[gid];//load & compute.
	}
	__syncthreads(); //always
	for(int s = blockDim.x/2; s > 0 ;s/=2){ //using s>>=2 is faster
		if((tid<s)&&(s_data[tid + s] < s_data[tid])) s_data[tid] = s_data[tid + s];
		__syncthreads(); //again.
	}
	if(tid == 0)  dstat= atomicMin(&output[0],s_data[0]);
}

float host_final_reduce(float *d_input, int size){
	float h_output_gpu[32];
	float *d_output;
	float result;
	
	//h_output_gpu = (float*)malloc(32*sizeof(float));
	cudaMalloc((void**)&d_output, 32*sizeof(float));
	
	dim3 dimGrid= dim3((size+511)/512,1,1);
	dim3 dimBlock= dim3(512,1,1);
	
	init_float_vec<<<32, 1>>>(d_output, 32);
	
	float_reduce_add<<<dimGrid, dimBlock>>>(d_output, d_input, size);
	
	cudaMemcpy(h_output_gpu,d_output,32*sizeof(float),cudaMemcpyDeviceToHost);
	
	result = h_output_gpu[0];
	//printf("result: %f\n", result);
	//free(h_output_gpu);
	cudaFree(d_output);
	return result;
}


int main(int argc, char **argv)
{
printf("Starting debbugging of functions in this .cu file\n");

float *h_input_A, *h_input_B;

// cudaEvent_t gstart, gstop;
// cudaEventCreate(&gstart);
// cudaEventCreate(&gstop);

int size = 2048*2048;
int vec_size_bytes = size * sizeof(float);

float result_gpu;


float *d_input_A, *d_input_B, *d_result_gpu;//, *d_output;


h_input_A = (float*)malloc(vec_size_bytes);
h_input_B = (float*)malloc(vec_size_bytes);
float* h_result_gpu = (float*)malloc(vec_size_bytes);


for(int i =0; i < size; i++){
	h_input_A[i] = 1.0f;
	h_input_B[i] = pow(-1.0f, (int)rand()*4)*rand()*1000;
	h_result_gpu[i] = FLT_MAX;
}
//step 1
cudaMalloc((void**)&d_input_A, vec_size_bytes);
cudaMalloc((void**)&d_input_B, vec_size_bytes); 
cudaMalloc((void**)&d_result_gpu, vec_size_bytes);
 
cudaMemcpy(d_input_A, h_input_A, vec_size_bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_input_B, h_input_B, vec_size_bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_result_gpu, h_result_gpu, vec_size_bytes, cudaMemcpyHostToDevice);
//cudaThreadSynchronize();
//result_gpu = host_final_reduce(d_input_A,size);
float_reduce_min<<<512, 512>>>(d_result_gpu,d_input_B, size);
cudaMemcpy(h_result_gpu, d_result_gpu, vec_size_bytes, cudaMemcpyDeviceToHost);
printf("gpu result = %f\n", h_result_gpu[0]);
//printf("cpu result = %f\n", size * 1.0);

//step 6
cudaFree(d_input_A);
cudaFree(d_input_B);
cudaFree(d_result_gpu);

free(h_input_A);
free(h_input_B);
free(h_result_gpu);



return 0;

}