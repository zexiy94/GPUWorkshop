// nvcc -arch=sm_37 -run matrixadd_solution.cu cuda_helper.cu
#include "cuda_runtime.h"
#include "chTimer.h"
#include "cuda_helper.h"
#include <stdio.h>


__global__ void gpu_matrix_add(float *result_mat,
                               float *mat_a,
                               float *mat_b,
                               int nrows, int ncols)
{

}
/////////////////////////////////Serial version/////////////////////////////////////////////
void host_matrix_add(float *result_mat, float *mat_a, float *mat_b, int nrows, int ncols){

	for(int row = 0; row < nrows; row++){
      for(int col = 0; col < ncols; col++){
		      result_mat[row * ncols + col] = mat_a[row * ncols + col] + mat_b[row * ncols + col];
      }
	}

}
/////////////////////////////////Diagnostic routines/////////////////////////////////////////////
int check_equal_float_vec(float *vec1,float *vec2,int size){
	int numerrors = 0;
	float dist;
	float tolerance = 0.01f;
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
  cudaDeviceReset();
//performance measure
cpuClock cpuck;
cudaClock ck;

float *d_mat_out, *d_mat_a, *d_mat_b;
float *h_mat_out_cpu, *h_mat_out_gpu, *h_mat_a, *h_mat_b;
printf("\nStarting program execution..\n\n");
int nrows = 4096;
int ncols = 4096;
std::cout << "rows " << nrows << " x cols " << ncols << std::endl;

printf("Allocating and creating problem data..\n");
int mat_size_bytes = nrows * ncols * sizeof(float);
std::cout << "mat_size in bytes: " << mat_size_bytes <<std::endl;
//allocation of host memory
h_mat_out_cpu = (float*)malloc(mat_size_bytes);
h_mat_out_gpu = (float*)malloc(mat_size_bytes);
h_mat_a = (float*)malloc(mat_size_bytes);
h_mat_b = (float*)malloc(mat_size_bytes);

for(int i =0; i < nrows * ncols; i++){
	h_mat_a[i] = (rand()%10)/10.0f;
	h_mat_b[i] = (rand()%15)/10.0f;
}
 ////
 //------ Step 1: Allocate the memory-------
 printf("Allocating Device Memory..\n");
CudaSafeCall(cudaMalloc((void**)&d_mat_out, mat_size_bytes));
//rest of the vectors

checkGPUMemory();
//------ Step 2: Copy Memory to the device-------
printf("Transfering data to the Device..\n");
CudaSafeCall(   );
CudaSafeCall(  );
//------ Step 3: Prepare launch parameters-------
printf("preparing launch parameters..\n");
int block_size_x = 32;
int block_size_y = 32;
dim3 dimGrid = dim3( , , 1);//.... CONFIGURE THE GRID IN 2D NOW!
std::cout << "dimGrid = " << dimGrid.x << " x " << dimGrid.y << std::endl;
dim3 dimBlock = dim3( , , 1);//.... we have  alimit of 1024 threads per block!!!!!
std::cout << "dimBlock = " << dimBlock.x << " x " << dimBlock.y << std::endl;
//------ Step 4: Launch device kernel-------
printf("Launch Device Kernel.\n");
cudaTick(&ck);
//---------------KERNEL LAUNCH HERE -----------------------
cudaTock(&ck, "kernel_matrix_add");
CudaCheckError();

//------ Step 5: Copy Memory back to the host-------
printf("Transfering result data to the Host..\n");
CudaSafeCall();
 //
printf("CPU version...\n");
cpuTick(&cpuck);
host_matrix_add(h_mat_out_cpu, h_mat_a, h_mat_b, nrows, ncols);//serial version to compare
cpuTock(&cpuck, "host_matrix_add");
std::cout << "gpu is " << cpuck.elapsedMicroseconds/ck.elapsedMicroseconds << " times faster" << std::endl;

printf("Checking solutions..\n");
check_equal_float_vec(h_mat_out_gpu, h_mat_out_cpu, nrows * ncols);
//

// -----------Step 6: Free the memory --------------
printf("Deallocating device memory..\n");
CudaSafeCall();
CudaSafeCall();
CudaSafeCall();

free(h_mat_a);
free(h_mat_b);
free(h_mat_out_cpu);
free(h_mat_out_gpu);

return 0;
}
