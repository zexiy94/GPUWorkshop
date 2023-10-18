// nvcc -arch=sm_60 -run warp_cosinesim.cu cuda_helper.cu
#include "cuda_runtime.h"
#include "chTimer.h"
#include "cuda_helper.h"
#include <stdio.h>
#include "device_launch_parameters.h"
#define WARP_SIZE 32
#define VECTOR_SIZE 512
#define DICTIONARY_SIZE 2048*1024

/*The objective of this exercise is to create a kernel, using shuffle instructions
to compute the dot product of many vectors. we will compare to a shared memory approach*/

/* One warp per dot product*/
__global__ void k_warp_dotproduct(float* all_scores, 
								float* all_vectors, 
								int* index_vec_a, 
								int* index_vec_b, 
								int vector_size, 
								int number_of_vectors)
{
	int nloops = vector_size / WARP_SIZE; // we can unroll if we use VECTOR_SIZE
	int wid = (blockIdx.x * blockDim.x + threadIdx.x)/WARP_SIZE;
	if(wid >= number_of_vectors) return;
	int lid = threadIdx.x % WARP_SIZE;
	int my_a_index = index_vec_a[wid];
	int my_b_index = index_vec_b[wid];
	float acc_sum = 0.0;
	//float acc_sum = 0.0;
	for (int i = 0; i < nloops; i++){
		int vid = lid + i * WARP_SIZE;
		acc_sum += all_vectors[my_a_index * vector_size + vid] * all_vectors[my_b_index * vector_size + vid]; 
	}
	__syncwarp(); //future proof
	//wap reduction
	for(unsigned mask = WARP_SIZE / 2 ; mask > 0 ; mask >>= 1) {
		acc_sum += __shfl_xor(acc_sum, mask);
		//__shfl_reduce_add_sync(acc_sum, mask); //c.c 8.0
		__syncwarp(); //future proof	
	}
	if(lid == 0)all_scores[wid] = acc_sum;

}

__device__ void reduce_zero(volatile float* sdata, int tid){
	for(int s = 1; s < blockDim.x ; s *= 2) { 
		if(tid % (2*s) == 0){ 
			sdata[tid] += sdata[tid + s]; 
		} 
		__syncthreads(); 
	} 
}

__device__ void reduce_one(volatile float* sdata, int tid){
	for(unsigned int s = 1; s < blockDim.x; s *= 2) { 
		int index = 2 * s * tid; 
	  if (index < blockDim.x) { 
		sdata[index] += sdata[index + s]; 
	  } 
	  __syncthreads(); 
	}
}

__device__ void reduce_two(volatile float* sdata, int tid){
	for(int s = blockDim.x/2; s > 0 ; s >>= 1) { 
		if(tid < s){ 
			sdata[tid] += sdata[tid + s]; 
		} 
		__syncthreads(); 
	} 
}



/* One block per dot product*/
__global__ void k_shmem_dotproduct(float* all_scores,  
								 float* all_vectors, 
								 int* index_vec_a, 
								 int* index_vec_b, 
								 int vector_size, 
								 int number_of_vectors) //https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
{
	extern __shared__ float shared_buffer[];
	if(blockIdx.x > number_of_vectors) return;
    int my_a_index = index_vec_a[blockIdx.x];
	int my_b_index = index_vec_b[blockIdx.x];
	// write to shared memory
	shared_buffer[threadIdx.x] = all_vectors[my_a_index * vector_size + threadIdx.x] * all_vectors[my_b_index * vector_size + threadIdx.x];
	//syncthreads
	__syncthreads();
	//reduction time!
	reduce_two(shared_buffer, threadIdx.x);
	// write to global mem 	
	if(threadIdx.x == 0) all_scores[blockIdx.x] = shared_buffer[0];
}

/* One thread per dot product*/
__global__ void k_dotproduct(float* all_scores,  
							 float* all_vectors, 
							 int* index_vec_a, 
							 int* index_vec_b, 
							 int vector_size, 
							 int number_of_vectors)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if(gid >= number_of_vectors) return;
    int my_a_index = index_vec_a[gid];
	int my_b_index = index_vec_b[gid];
	float acc_sum = 0.0f;
	for(int i = 0; i < vector_size; i++){
	 //read from global memory, transposing would help in this case, still this is just a baseline
		 acc_sum += all_vectors[my_a_index * vector_size + i] * all_vectors[my_b_index * vector_size + i];
	}
	//write to global memory
	all_scores[gid] = acc_sum;
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
	cudaClock cknaive, ckshmem, ckwarp;
	float *d_dictionary, *d_scores;
	int *d_indexes_a, *d_indexes_b;
	float *h_dictionary, *h_scores;
	int *h_indexes_a, *h_indexes_b;
	printf("\nStarting program execution..\n\n");

	printf("\nAllocating and creating problem data..\n");

	h_dictionary = (float*)malloc(DICTIONARY_SIZE * VECTOR_SIZE * sizeof(float));
	h_scores 	 = (float*)malloc(DICTIONARY_SIZE * sizeof(float));
	h_indexes_a  = (int*)malloc(DICTIONARY_SIZE * sizeof(int));
	h_indexes_b  = (int*)malloc(DICTIONARY_SIZE * sizeof(int));


	for(int i = 0; i < DICTIONARY_SIZE; i++){
		h_indexes_a[i] = (rand()%DICTIONARY_SIZE);
		h_indexes_b[i] = (rand()%DICTIONARY_SIZE);
	}
	for(int i = 0; i < DICTIONARY_SIZE * VECTOR_SIZE; i++){
		h_dictionary[i] = rand() * 1.0f;
	}
	 ////
	 //------ Step 1: Allocate the memory-------
	 printf("\nAllocating Device Memory\n");
	CudaSafeCall(cudaMalloc((void**)&d_dictionary, DICTIONARY_SIZE * VECTOR_SIZE * sizeof(float)));
	CudaSafeCall(cudaMalloc((void**)&d_scores,     DICTIONARY_SIZE * sizeof(float)));
	CudaSafeCall(cudaMalloc((void**)&d_indexes_a,  DICTIONARY_SIZE * sizeof(int)));
	CudaSafeCall(cudaMalloc((void**)&d_indexes_b,  DICTIONARY_SIZE * sizeof(int)));

	checkGPUMemory();

	//------ Step 2: Copy Memory to the device-------
	printf("\nTransfering data to the Device..\n");
	CudaSafeCall(cudaMemcpy(d_dictionary, h_dictionary, DICTIONARY_SIZE * VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_indexes_a, h_indexes_a,   DICTIONARY_SIZE * sizeof(int), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_indexes_b, h_indexes_b,   DICTIONARY_SIZE * sizeof(int), cudaMemcpyHostToDevice));
	CudaCheckError();
	//------ Step 3: Prepare launch parameters-------
	printf("\npreparing launch parameters..\n");
	//warp launch params
	dim3 warpGrid = dim3((DICTIONARY_SIZE + WARP_SIZE - 1)/WARP_SIZE, 1, 1);
	dim3 warpBlock = dim3(128,1,1);
	//shmem launch params
	dim3 shmemGrid = dim3(DICTIONARY_SIZE, 1, 1);//
	dim3 shmemBlock = dim3(VECTOR_SIZE,1,1);
	int shmem_size = VECTOR_SIZE * sizeof(float);
	//naive GPU launch params
	dim3 naiveGrid = dim3((DICTIONARY_SIZE + 127)/128, 1, 1);//
	dim3 naiveBlock = dim3(128,1,1);
	//------ Step 4: Launch device kernel-------
	printf("\nLaunch Device Kernels.\n");

	CudaCheckError();
	// KERNEL LAUNCHS GOES HERE------------------------>>>>>>>>>
	printf("\nLaunch Naive Kernel.\n");
	cudaTick(&cknaive);
	k_dotproduct<<<naiveGrid, naiveBlock>>>(d_scores, 
											d_dictionary, 
											d_indexes_a, 
											d_indexes_b, 
											VECTOR_SIZE, 
											DICTIONARY_SIZE);
	cudaTock(&cknaive, "naive dotproduct");
	CudaCheckError();
	cudaDeviceSynchronize();
	printf("\nLaunch Shared Memory Kernel.\n");
	cudaTick(&ckshmem);
	k_shmem_dotproduct<<<shmemGrid, shmemBlock, shmem_size>>>(d_scores, 
															  d_dictionary, 
															  d_indexes_a, 
															  d_indexes_b, 
															  VECTOR_SIZE, 
															  DICTIONARY_SIZE);
	cudaTock(&ckshmem, "shared memory dotproduct");
	CudaCheckError();
	cudaDeviceSynchronize();
	printf("\nLaunch Warp Kernel.\n");
	cudaTick(&ckwarp);
	k_warp_dotproduct<<<warpGrid, warpBlock>>>(d_scores, 
												d_dictionary, 
												d_indexes_a, 
												d_indexes_b, 
												VECTOR_SIZE, 
												DICTIONARY_SIZE);
	cudaTock(&ckwarp, "warp dotproduct");

	std::cout << std::endl;
	CudaCheckError();
	//------ Step 5: Copy Memory back to the host-------
	printf("\nTransfering result data to the Host..\n");
	CudaSafeCall(cudaMemcpy(h_scores, d_scores, DICTIONARY_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
	//
	printf("\nComparing  times..\n");

	std::cout << "the shmem gpu is " << cknaive.elapsedMicroseconds/ckshmem.elapsedMicroseconds << " times faster than naive gpu" << std::endl;
	std::cout << "the warp gpu is " << cknaive.elapsedMicroseconds/ckwarp.elapsedMicroseconds << " times faster than naive gpu" << std::endl;
	std::cout << "the warp gpu is " << ckshmem.elapsedMicroseconds/ckwarp.elapsedMicroseconds << " times faster than shmem gpu" << std::endl;


	// -----------Step 6: Free the memory --------------
	printf("Deallocating device memory..\n");

	cudaFree(d_dictionary);
	cudaFree(d_scores);
	cudaFree(d_indexes_a);
	cudaFree(d_indexes_b);

	printf("Deallocating host memory..\n");
	free(h_dictionary);
	free(h_scores);
	free(h_indexes_a);
	free(h_indexes_b);

	return 42;
}

