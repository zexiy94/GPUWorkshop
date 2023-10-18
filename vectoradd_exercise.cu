// nvcc -arch=sm_37 -run vectoradd_exercise.cu
#include "cuda_runtime.h"
#include <stdio.h>


__global__ void kernel_vector_add(float *result_vec, float *vec_a, float *vec_b, int vector_size){
  ////// YOUR CODE GOES HERE!////////////////////////////////////////////
  //step 1.. define the thread ID

  //step 2.. make sure the thread works whithin the array bounds

	//step 3.. compute the vector sum

}
/////////////////////////////////Serial version/////////////////////////////////////////////
void host_vector_add(float *result_vec, float *vec_a, float *vec_b, int vector_size){

	for(int elem = 0; elem < vector_size; elem++){
		result_vec[elem] = vec_a[elem] + vec_b[elem];
	}

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
float *d_vec_out, *d_vec_a, *d_vec_b;
float *h_vec_out_cpu, *h_vec_out_gpu, *h_vec_a, *h_vec_b;
printf("\nStarting program execution..\n\n");
int vec_size = 4096;

printf("Allocating and creating problem data..\n");
int vec_size_bytes = vec_size * sizeof(float);
//allocation of host memory
h_vec_out_cpu = (float*)malloc(vec_size_bytes);
h_vec_out_gpu = (float*)malloc(vec_size_bytes);
h_vec_a = (float*)malloc(vec_size_bytes);
h_vec_b = (float*)malloc(vec_size_bytes);

for(int i =0; i < vec_size; i++){
	h_vec_a[i] = (rand()%10)/10.0f;
	h_vec_b[i] = (rand()%15)/10.0f;
}
 ////
 //------ Step 1: Allocate the memory-------
 printf("Allocating Device Memory..\n");
cudaMalloc((void**)&d_vec_out, vec_size_bytes);
//.... ALLOCATE THE REST OF THE VECTORS

//------ Step 2: Copy Memory to the device-------
printf("Transfering data to the Device..\n");
cudaMemcpy(d_vec_a, h_vec_a, vec_size_bytes, cudaMemcpyHostToDevice);
//.... COPY THE REST OF THE VECTORS

//------ Step 3: Prepare launch parameters-------
printf("preparing launch parameters..\n");

dim3 dimGrid = //.... CONFIGURE THE GRID IN BLOCKS OF 256 THREADS BLOCKS
dim3 dimBlock = dim3(256,1,1);
//------ Step 4: Launch device kernel-------
printf("Launch Device Kernel.\n");
// YOUR KERNEL LAUNCH GOES HERE------------------------>>>>>>>>>

//------ Step 5: Copy Memory back to the host-------
printf("Transfering result data to the Host..\n");
cudaMemcpy(h_vec_out_gpu, d_vec_out, vec_size_bytes, cudaMemcpyDeviceToHost);

 //
 printf("CPU version...\n");
host_vector_add(h_vec_out_cpu, h_vec_a, h_vec_b, vec_size);//serial version to compare
printf("Checking solutions..\n");
check_equal_float_vec(h_vec_out_gpu, h_vec_out_cpu, vec_size);
//

// -----------Step 6: Free the memory --------------
printf("Deallocating device memory..\n");
cudaFree(d_vec_out);
//.... FREE THE REST OF THE VECTORS

free(h_vec_a);
free(h_vec_b);
free(h_vec_out_cpu);
free(h_vec_out_gpu);

return 0;
}

// cudaEvent_t gstart, gstop;
// cudaEventCreate(&gstart);
// cudaEventCreate(&gstop);
