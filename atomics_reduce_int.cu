// Reduce SUM ... improved version, 21st June 2013
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"

__global__ void init_int_vec(int *vec, int size){
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if(gid < size)vec[gid] = 0;
}

__global__ void init_int_vec_factor(int *vec, int factor,int size){
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if(gid < size)vec[gid] = factor;
}

__global__ void int_reduce_add(int *output, int *input, int size){
	__shared__ int s_data[512]; //shared memory
	int dstat;
	int chunk_size = ceil( (1.0f * size)/gridDim.x);
	int start_elem = blockIdx.x * chunk_size;
	int tid = threadIdx.x;
	int gid;
	s_data[tid] = 0; //initialize properly depending on the reduce operation.
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

__global__ void int_reduce_min(int *output, int *input, int size){
	__shared__ int s_data[512]; //shared memory
	int dstat;
	int chunk_size = ceil( (1.0f * size)/gridDim.x);
	int start_elem = blockIdx.x * chunk_size;
	int tid = threadIdx.x;
	int gid;
	s_data[tid] = 2147483647; //initialize properly depending on the reduce operation.
	for(int k = 0; (k * blockDim.x + tid ) < chunk_size; k++){ //we do a lot of “serial work”
		gid = start_elem + k * blockDim.x + tid;
		if(gid < size) s_data[tid] = min(s_data[tid],input[gid]);//load & compute.
	}
	__syncthreads(); //always
	for(int s = blockDim.x/2; s > 0 ;s/=2){ //using s>>=2 is faster
		if(tid<s) s_data[tid] = min(s_data[tid],s_data[tid + s]); //no thread divergence until last 5 iter.
		__syncthreads(); //again.
	}
	if(tid == 0)  dstat= atomicMin(&output[0],s_data[0]);
}

__global__ void int_reduce_min_w_index(int *output, int* outindex, int *input, int size){
	__shared__ int s_data[512]; //shared memory
	__shared__ int s_index[512]; //shared memory
	int dstat;
	int chunk_size = ceil( (1.0f * size)/gridDim.x);
	int start_elem = blockIdx.x * chunk_size;
	int tid = threadIdx.x;
	int gid;
	s_data[tid] = 2147483647; //initialize properly depending on the reduce operation.
	for(int k = 0; (k * blockDim.x + tid ) < chunk_size; k++){ //we do a lot of “serial work”
		gid = start_elem + k * blockDim.x + tid;
		if(gid < size){
			 if(s_data[tid] > input[gid]){
				s_data[tid] = input[gid];
				s_index[tid] = gid;
			 }
		
		}//load & compute.
	}
	__syncthreads(); //always
	for(int s = blockDim.x/2; s > 0 ;s/=2){ //using s>>=2 is faster
		if(tid<s) {
			if(s_data[tid] > s_data[tid + s] ){
				s_data[tid] = s_data[tid + s];
				s_index[tid] = s_index[tid + s];
			}
		}
		__syncthreads(); //again.
	}
	if(tid == 0){  
		output[blockIdx.x] = s_data[0];
		outindex[blockIdx.x] = s_index[0];
	}
}

int host_int_min_reduce_w_index(int *d_input, int size){
	int h_output_gpu[32], h_outindex_gpu[32];
	int *d_output, d_outindex;
	int result, resindex;
	result = 2147483647;
	//h_output_gpu = (float*)malloc(32*sizeof(int));
	cudaMalloc((void**)&d_output, 32*sizeof(int));
	
	dim3 dimGrid= dim3(32,1,1);
	dim3 dimBlock= dim3(512,1,1);
	
	init_int_vec_factor<<<32, 1>>>(d_output,2147483647,32);
	
	int_reduce_min_w_index<<<dimGrid, dimBlock>>>(d_output, d_outindex d_input, size);
	
	cudaMemcpy(h_output_gpu,d_output,32*sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_outindex_gpu,d_outindex,32*sizeof(int),cudaMemcpyDeviceToHost);
	
	for(int i = 0; i< 32; i++){
		if(result > h_output_gpu[i]){
			result = h_output_gpu[i];
			resindex = h_outindex_gpu[i];
		}
	}
	printf("min: %d, with index: %d\n", result, resindex);
	//free(h_output_gpu);
	cudaFree(d_output);
	cudaFree(d_outindex);
	return resindex;
}

int host_int_min_reduce(int *d_input, int size){
	int h_output_gpu[32];
	int *d_output;
	int result;
	
	//h_output_gpu = (float*)malloc(32*sizeof(int));
	cudaMalloc((void**)&d_output, 32*sizeof(int));
	
	dim3 dimGrid= dim3((size+511)/512,1,1);
	dim3 dimBlock= dim3(512,1,1);
	
	init_int_vec_factor<<<32, 1>>>(d_output,2147483647,32);
	
	int_reduce_min<<<dimGrid, dimBlock>>>(d_output, d_input, size);
	
	cudaMemcpy(h_output_gpu,d_output,32*sizeof(int),cudaMemcpyDeviceToHost);
	
	result = h_output_gpu[0];
	//printf("result: %f\n", result);
	//free(h_output_gpu);
	cudaFree(d_output);
	return result;
}

int host_int_sum_reduce(int *d_input, int size){
	int h_output_gpu[32];
	int *d_output;
	int result;
	
	//h_output_gpu = (float*)malloc(32*sizeof(int));
	cudaMalloc((void**)&d_output, 32*sizeof(int));
	
	dim3 dimGrid= dim3((size+511)/512,1,1);
	dim3 dimBlock= dim3(512,1,1);
	
	init_int_vec<<<32, 1>>>(d_output, 32);
	
	int_reduce_add<<<dimGrid, dimBlock>>>(d_output, d_input, size);
	
	cudaMemcpy(h_output_gpu,d_output,32*sizeof(int),cudaMemcpyDeviceToHost);
	
	result = h_output_gpu[0];
	//printf("result: %f\n", result);
	//free(h_output_gpu);
	cudaFree(d_output);
	return result;
}


int main(int argc, char **argv)
{
printf("Starting debbugging of functions in this .cu file\n");

int *h_input_A, *h_input_B;

// cudaEvent_t gstart, gstop;
// cudaEventCreate(&gstart);
// cudaEventCreate(&gstop);

int size = 37*2049+1+1024*1024;
int vec_size_bytes = size * sizeof(int);

int result_gpu;


int *d_input_A, *d_input_B;//, *d_output;


h_input_A = (int*)malloc(vec_size_bytes);
h_input_B = (int*)malloc(vec_size_bytes);


for(int i =0; i < size; i++){
	h_input_A[i] = rand()%2048-127;
	h_input_B[i] = 2;
}
//step 1
cudaMalloc((void**)&d_input_A, vec_size_bytes);
cudaMalloc((void**)&d_input_B, vec_size_bytes); 
//cudaMalloc((void**)&d_output, vec_size_bytes);
 
cudaMemcpy(d_input_A, h_input_A, vec_size_bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_input_B, h_input_B, vec_size_bytes, cudaMemcpyHostToDevice);
//cudaThreadSynchronize();
result_gpu = host_int_min_reduce(d_input_A,size);
printf("gpu result = %d\n", result_gpu);
printf("cpu result = %d\n", size);

//step 6
cudaFree(d_input_A);
cudaFree(d_input_B);
//cudaFree(d_output);

free(h_input_A);
free(h_input_B);

return 0;

}