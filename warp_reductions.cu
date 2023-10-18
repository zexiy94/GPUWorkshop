// nvcc -arch=sm_61 -run warp_reductions.cu cuda_helper.cu
#include "cuda_runtime.h"
#include "chTimer.h"
#include "cuda_helper.h"
#include <stdio.h>
#define WARP_SIZE 32

/*The objective of this exercise is to create a kernel, using shuffle instructions
to compute the average of many vectors. Can you think of another version using atomics?*/

__global__ void scan_by_warp(int* scanned, int* input, int vector_size, int number_of_vectors)
{
  //step 1.. define lane and vector
  int myLane = threadIdx.x;
  int myVector = threadIdx.y + blockIdx.y * blockDim.y;
  if(myVector >= number_of_vectors) return;
  int nLoops = (vector_size + 31)/32;

  int init = 0;
  if (myVector == 1) init = 4;

  int in_index = init + myLane;
  int out_index  = init + myLane;
  int only_first = 4;
  if (myVector == 1) only_first = 5;

  int temp1 = input[in_index];
  int t3 = temp1;
  int temp2 = 0;
  for (int d=1; d<32; d<<=1) {
    temp2 = __shfl_up(temp1,d);
    if (myLane >= d && myLane < only_first) temp1 += temp2;
  }
  if( myLane < only_first)
    input[out_index] = temp1 - input[in_index];
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
cudaClock ck1, ck2;
int *d_vec_avg, *d_vec_avg2, *d_vec_in;
int *h_vec_avg_cpu, *h_vec_avg_gpu, *h_vec_avg_gpu2, *h_vec_in;
printf("\nStarting program execution..\n\n");
int vec_size = 32;
int number_of_vectors = 2;

printf("\nAllocating and creating problem data..\n");

//allocation of host memory
//h_vec_avg_cpu = (int*)malloc(number_of_vectors * sizeof(int));
//h_vec_avg_gpu = (int*)malloc(number_of_vectors * sizeof(int));
h_vec_avg_gpu2 = (int*)malloc(number_of_vectors * vec_size * sizeof(int));
h_vec_in = (int*)malloc(number_of_vectors * vec_size * sizeof(int));

for(int i =0; i < number_of_vectors * vec_size; i++){
	h_vec_in[i] = (rand()%2);
}
 ////
 //------ Step 1: Allocate the memory-------
 printf("\nAllocating Device Memory\n");
//CudaSafeCall(cudaMalloc((void**)&d_vec_avg,  vec_size * number_of_vectors * sizeof(int)));
CudaSafeCall(cudaMalloc((void**)&d_vec_avg2, vec_size * number_of_vectors * sizeof(int)));
CudaSafeCall(cudaMalloc((void**)&d_vec_in,   number_of_vectors * vec_size* sizeof(int)));

//------ Step 2: Copy Memory to the device-------
printf("\nTransfering data to the Device..\n");
CudaSafeCall(cudaMemcpy(d_vec_in, h_vec_in, number_of_vectors * vec_size * sizeof(int), cudaMemcpyHostToDevice));
CudaCheckError();
//------ Step 3: Prepare launch parameters-------
printf("\npreparing launch parameters..\n");
//shuffle launch params
int tpby = 4;
dim3 shuffleGrid = dim3(1, (number_of_vectors + tpby - 1) / tpby, 1);//.... CONFIGURE THE GRID IN BLOCKS OF 256 THREADS BLOCKS
dim3 shuffleBlock = dim3(32,tpby,1);
//------ Step 4: Launch device kernel-------
printf("\nLaunch Device Kernels.\n");
;
CudaCheckError();
// YOUR KERNEL LAUNCH GOES HERE------------------------>>>>>>>>>
cudaTick(&ck2);
scan_by_warp<<<shuffleGrid, shuffleBlock>>>(d_vec_avg2,
                                                        d_vec_in,
                                                        vec_size,
                                                        number_of_vectors);
cudaTock(&ck2, "scan_by_warp");
CudaCheckError();
std::cout << std::endl;
//------ Step 5: Copy Memory back to the host-------
printf("\nTransfering result data to the Host..\n");
CudaSafeCall(cudaMemcpy(h_vec_avg_gpu2, d_vec_in, vec_size * number_of_vectors * sizeof(int), cudaMemcpyDeviceToHost));//to compute in place
//CudaSafeCall(cudaMemcpy(h_vec_avg_gpu2, d_vec_avg2, vec_size * number_of_vectors * sizeof(int), cudaMemcpyDeviceToHost));
//
printf("\nCPU version...\n");
/*cpuTick(&cpuck);
host_vector_multi_avg(h_vec_avg_cpu, h_vec_in, vec_size, number_of_vectors);//serial version to compare
cpuTock(&cpuck, "host_vector_multi_avg");
std::cout << "the best gpu is " << cpuck.elapsedMicroseconds/ck2.elapsedMicroseconds << " times faster" << std::endl;*/
printf("\nChecking solutions..\n");
std::cout << "Original vector: " << std::endl;
for(int i = 0; i < WARP_SIZE; i++) {
  std::cout << h_vec_in[i] << " ";
}
std::cout << std::endl;

std::cout << "Scanned: " << std::endl;
for(int i = 0; i < WARP_SIZE; i++) {
  std::cout << h_vec_avg_gpu2[i] << " ";
}
std::cout << std::endl;
/*check_equal_float_vec(h_vec_avg_gpu, h_vec_avg_cpu, number_of_vectors);
check_equal_float_vec(h_vec_avg_gpu2, h_vec_avg_cpu, number_of_vectors);
check_equal_float_vec(h_vec_avg_gpu2, h_vec_avg_gpu, number_of_vectors);*/
// -----------Step 6: Free the memory --------------
printf("Deallocating device memory..\n");

cudaFree(d_vec_avg2);

free(h_vec_in);
free(h_vec_avg_gpu2);

return 42;
}

// cudaEvent_t gstart, gstop;
// cudaEventCreate(&gstart);
// cudaEventCreate(&gstop);
