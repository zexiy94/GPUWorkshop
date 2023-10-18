#include "cuda_runtime.h"
#include <stdio.h>


__global__ void kernel_weigthed_sum(float *result_vec, float *in0, float *in1,float *in2,float *in3,
									float *in4,float *weight, int size){
  ////// YOUR CODE GOES HERE!////////////////////////////////////////////
  //step 1.. define the thread ID 
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < size){
	result_vec[gid] = weight[0]*in0[gid] + weight[1]*in1[gid] + weight[2]*in2[gid] +
					weight[3]*in3[gid] + weight[4]*in4[gid];
    }
  
  
}
/////////////////////////////////Serial version/////////////////////////////////////////////
void host_serial(float *result_vec, float *in0, float *in1,float *in2,float *in3,
									float *in4,float *weight, int size){
	
	for(int elem = 0; elem < size; elem++){
		result_vec[elem] = weight[0]*in0[elem] + weight[1]*in1[elem] + weight[2]*in2[elem] +
					weight[3]*in3[elem] + weight[4]*in4[elem];
	}
	
}
/////////////////////////////////Diagnostic routines/////////////////////////////////////////////
int check_equal_float_vec(float *vec1,float *vec2,int size){
	int numerrors=0;
	float dist;
	float tolerance = 0.01f;
	for(int i =0; i< size; i++){
	    dist = (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
		if(dist > tolerance) numerrors++;
		//printf("vec1:%f; vec2:%f; dist:%f\n",vec1[i],vec2[i],dist);
	}
	if(numerrors ==0)printf("Congratulations you have 0 errors!\n");
	if(numerrors >0)printf("Wrong results, you have %d errors!\n", numerrors);
	
	return numerrors;
}
//////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
float *d_vec_out, *d_in0, *d_in1, *d_in2, *d_in3, *d_in4, *d_weight;
float *h_vec_out_cpu, *h_vec_out_gpu, *h_in0, *h_in1, *h_in2, *h_in3, *h_in4, *h_weight;
printf("\nStarting program execution..\n\n");
int vec_size = 4096*4096;

printf("Allocating and creating problem data..\n");
int vec_size_bytes = vec_size * sizeof(float);
//allocation of host memory
h_vec_out_cpu = (float*)malloc(vec_size_bytes);
h_vec_out_gpu = (float*)malloc(vec_size_bytes);
h_in0 = (float*)malloc(vec_size_bytes);
h_in1 = (float*)malloc(vec_size_bytes);
h_in2 = (float*)malloc(vec_size_bytes);
h_in3 = (float*)malloc(vec_size_bytes);
h_in4 = (float*)malloc(vec_size_bytes);
h_weight = (float*)malloc(5*sizeof(float));

for(int i =0; i < vec_size; i++){
	h_in0[i] = (rand()%10)/10.0f;
	h_in1[i] = (rand()%15)/10.0f;
	h_in2[i] = (rand()%5)/10.0f;
	h_in3[i] = (rand()%2)/10.0f;
	h_in4[i] = (rand()%18)/10.0f;
}
for(int i =0; i<5; i++)h_weight[i] = (rand()%10)/10.0f;//initializing the weights
 ////
 //------ Step 1: Allocate the memory-------
 printf("Allocating Device Memory..\n");
cudaMalloc((void**)&d_vec_out, vec_size_bytes);
cudaMalloc((void**)&d_in0, vec_size_bytes);
cudaMalloc((void**)&d_in1, vec_size_bytes);
cudaMalloc((void**)&d_in2, vec_size_bytes);
cudaMalloc((void**)&d_in3, vec_size_bytes);
cudaMalloc((void**)&d_in4, vec_size_bytes);
cudaMalloc((void**)&d_weight, 5 * sizeof(float));

//------ Step 2: Copy Memory to the device-------
printf("Transfering data to the Device..\n");
cudaMemcpy(d_in0, h_in0, vec_size_bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_in1, h_in1, vec_size_bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_in2, h_in2, vec_size_bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_in3, h_in3, vec_size_bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_in4, h_in4, vec_size_bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_weight, h_weight, 5 * sizeof(float), cudaMemcpyHostToDevice);
//------ Step 3: Prepare launch parameters-------
printf("preparing launch parameters..\n");
dim3 dimGrid = dim3(( vec_size + 255)/256,1,1); //we will launch one thread per professor
dim3 dimBlock = dim3(256,1,1);
//------ Step 4: Launch device kernel-------
printf("Launch Device Kernel.\n");

// YOUR KERNEL LAUNCH GOES HERE------------------------>>>>>>>>>
kernel_weigthed_sum<<<dimGrid, dimBlock>>>(d_vec_out, d_in0, d_in1, d_in2, d_in3, d_in4, d_weight, vec_size);


//------ Step 5: Copy Memory back to the host-------
printf("Transfering result data to the Host..\n");
cudaMemcpy(h_vec_out_gpu, d_vec_out, vec_size_bytes, cudaMemcpyDeviceToHost);
 
 //
 printf("CPU version...\n");
host_serial(h_vec_out_cpu, h_in0, h_in1,h_in2,h_in3, h_in4, h_weight, vec_size);//serial version to compare 
printf("Checking solutions..\n");
check_equal_float_vec(h_vec_out_gpu, h_vec_out_cpu, vec_size);
//

// -----------Step 6: Free the memory -------------- 
printf("Deallocating device memory..\n");
cudaFree(d_in0);
cudaFree(d_in1);
cudaFree(d_in2);
cudaFree(d_in3);
cudaFree(d_in4);
cudaFree(d_weight);
cudaFree(d_vec_out);

free(h_in0);
free(h_in1);
free(h_in2);
free(h_in3);
free(h_in4);
free(h_weight);
free(h_vec_out_cpu);
free(h_vec_out_gpu);

return 0;
}

// cudaEvent_t gstart, gstop;
// cudaEventCreate(&gstart);
// cudaEventCreate(&gstop);