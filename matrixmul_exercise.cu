// // nvcc -run -arch=sm_37 matrixmul_exercise.cu cuda_helper.cu
#include "cuda_runtime.h"
#include "chTimer.h"
#include "cuda_helper.h"
#include <stdio.h>


__global__ void gpu_matrix_mul(float *result_mat,
                               float *mat_a,
                               float *mat_b,
                               int nrowsA, int nrowsB, int ncolsA, int ncolsB)
{

  
}
/////////////////////////////////Serial version/////////////////////////////////////////////
void host_matrix_mul(float *result_mat, float *mat_a, float *mat_b, int nrowsA, int nrowsB, int ncolsA, int ncolsB){

	for(int row = 0; row < nrowsA; row++){
      for(int col = 0; col < ncolsB; col++){
        int tmp = 0;
        for(int k = 0; k < nrowsB; k++){
		       tmp += mat_a[row * ncolsA + k] * mat_b[k * ncolsB + col];
        }
        result_mat[row * ncolsA + col] = tmp;
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
int noofmatrix = 32;
int nrows = 256;
int ncols = 256;
std::cout << "rows " << nrows << " x cols " << ncols<<std::endl;

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
    h_mat_out_cpu[i] = 0;
    h_mat_out_gpu[i] = 0;
}

int nrowsA = nrows;
int ncolsA = ncols;
int nrowsB = nrows;
int ncolsB = ncols;
 ////
 //------ Step 1: Allocate the memory-------
 printf("Allocating Device Memory..\n");
 // The two input matrix arrays for gpu are d_mat_a and d_mat_b.
 // The output matrix array for gpu is d_mat_out
 // Also remember to allocate your streams!



checkGPUMemory();



//------ Step 3: Prepare launch parameters-------
printf("preparing launch parameters..\n");



cudaTick(&ck);
//------ Step 2: Copy Memory to the device -------
// TODO: Add stream compatibility
// The two input matrix arrays on the cpu are h_mat_a and h_mat_b
printf("Transfering data to the Device..\n");

int block_size_x = ; //TODO: FILL IN
int block_size_y = ; //TODO: FILL IN
dim3 dimGrid = ; // TODO: FILL IN
std::cout << "dimGrid = " << dimGrid.x << " x " << dimGrid.y << std::endl;
dim3 dimBlock = ;// TODO: FILL IN
std::cout << "dimBlock = " << dimBlock.x << " x " << dimBlock.y << std::endl;


//------ Step 4: Launch device kernel-------
printf("Launch Device Kernel.\n");
// YOUR KERNEL LAUNCH GOES HERE------------------------>>>>>>>>>

CudaCheckError();

//------ Step 5: Copy Memory back to the host-------
printf("Transfering result data to the Host..\n");
// The output matrix for cpu is h_mat_out_gpu


 //

cudaTock(&ck, "gpu_matrix_mul");
printf("CPU version...\n");
cpuTick(&cpuck);
host_matrix_mul(h_mat_out_cpu, h_mat_a, h_mat_b, nrowsA, ncolsA, nrowsB, ncolsB);//serial version to compare
cpuTock(&cpuck, "host_matrix_mul");
std::cout << "gpu is " << cpuck.elapsedMicroseconds/ck.elapsedMicroseconds << " times faster" << std::endl;

printf("Checking solutions..\n");
check_equal_float_vec(h_mat_out_gpu, h_mat_out_cpu, noofmatrix * nrows * ncols);
//

// -----------Step 6: Free the memory --------------
printf("Deallocating device memory..\n");



// CPU side freeing
free(h_mat_a);
free(h_mat_b);
free(h_mat_out_cpu);
free(h_mat_out_gpu);

return 0;
}