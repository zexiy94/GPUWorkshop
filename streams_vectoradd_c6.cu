#include <stdio.h>
#define SIZE 2048*2048*4
#define BLOCK_SIZE 256
#define NSTREAMS 4 //32
// our kernel... identified by the keyword __global__
__global__ void add(int size, float* inputA, float* inputB, float* output){
	int gid = blockIdx.x * blockDim.x + threadIdx.x; //we calculate the thread global index
	if(gid<size) output[gid] = inputA[gid] + inputB[gid];
}
 void serial_add (int size, float* inputA, float* inputB, float* output){
// serial implementation of the vector add.
	for (int i =0; i< size; i++){
		output[i] = inputA[i] + inputB[i];
	}
}
 void checkresults (int size, float* result_cpu, float* result_gpu){
// error checking function.
  float tolerance = 0.001f;
  float dist;
  int total_errors =0;
	for (int i =0; i< size; i++){
	   dist = result_cpu[i] - result_gpu[i];
		if(tolerance <(dist * dist)) total_errors++;
	}
	if(total_errors){
		printf("%d Errors in the code!\n", total_errors);
	}else{
	    printf("Correct!\n");
	}
}
int main(int argc,char **argv)
{

 printf("Starting the execution!\n");
 if(cudaSuccess != cudaGetLastError()) printf("Previous error in the system");
 
 cudaStream_t *streams = (cudaStream_t *) malloc(NSTREAMS * sizeof(cudaStream_t));
 for(int is =0; is < NSTREAMS; is++){
	cudaStreamCreate(&(streams[is]));
	if(cudaSuccess != cudaGetLastError()) printf("Error creating the %d stream", is);
 }
 
 //float *h_input_A, *h_input_B, *h_output_gpu, *h_output_cpu;
 float *d_input_A[NSTREAMS];
 float *d_input_B[NSTREAMS]; 
 float *d_output[NSTREAMS];
  
 int size_bytes = SIZE * sizeof(float);
 
 //printf("Allocating host memory!\n");
 //h_input_A = (float*)malloc(size_bytes);
 //h_input_B = (float*)malloc(size_bytes);
 //h_output_gpu = (float*)malloc(size_bytes);
 //h_output_cpu = (float*)malloc(size_bytes);
 
 printf("Allocating device memory!\n");
 for(int is =0; is < NSTREAMS; is++){
	cudaMallocManaged((void**) &d_input_A[is], size_bytes);
	cudaMallocManaged((void**) &d_input_B[is], size_bytes);
	cudaMallocManaged((void**) &d_output[is], size_bytes);
	if(cudaSuccess != cudaGetLastError()) printf("Error allocating device memory component %d", is);
 }
 
  
 
 //printf("Copying memory from host to device!\n");
 // for(int is =0; is < NSTREAMS; is++){
//	cudaMemcpyAsync(d_input_A[is], h_input_A, size_bytes, cudaMemcpyHostToDevice, streams[is]);
//	cudaMemcpyAsync(d_input_B[is], h_input_B, size_bytes, cudaMemcpyHostToDevice, streams[is]);
//	if(cudaSuccess != cudaGetLastError()) printf("Error memcpy stream %d", is);
// }
 
 dim3 dimGrid = dim3((SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE, 1, 1);
 dim3 dimBlock = dim3(BLOCK_SIZE,1,1);
 
 for(int is =0; is < NSTREAMS; is++){ 
	add<<<dimGrid, dimBlock, 0, streams[is]>>>(SIZE, d_input_A[is], d_input_B[is], d_output[is]);
	//add<<<dimGrid, dimBlock, 0, streams[is]>>>(SIZE, d_input_A[1], d_input_B[0], d_output[is]);
 }
 if(cudaSuccess != cudaGetLastError()) printf("Error in Stream calculation");
 
 //printf("Copying memory back from the device to the host!\n");
 //cudaMemcpyAsync(h_output_gpu, d_output[0], size_bytes, cudaMemcpyDeviceToHost, streams[0]);
 //if(cudaSuccess != cudaGetLastError()) printf("Error coying memory back to the host ");
 
 printf("Deallocating GPU memory \n");
  for(int is =0; is < NSTREAMS; is++){
	cudaFree(d_input_A[is]);
	cudaFree(d_input_B[is]);
	cudaFree(d_output[is]); 
	if(cudaSuccess != cudaGetLastError()) printf("Error deallocating device memory component %d", is);
 }
 //printf("Deallocating CPU memory \n");
 //free(h_input_A);
 //free(h_input_B);
 //free(h_output_gpu); 
 //free(h_output_cpu); 
 //printf("That's all!\n");

    return 0;
}

