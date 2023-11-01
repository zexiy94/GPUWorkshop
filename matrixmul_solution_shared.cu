// // nvcc -run matrixmul_solution_naive.cu cuda_helper.cu
#include "cuda_runtime.h"
#include "chTimer.h"
#include "cuda_helper.h"
#include <stdio.h>


__global__ void gpu_matrix_mul(float *result_mat,
                               float *mat_a,
                               float *mat_b,
                               int nrowsA, int nrowsB, int ncolsA, int ncolsB)
{
      // Compute each thread's global row and column index
    int row_id = blockIdx.y * blockDim.y + threadIdx.y;
    int col_id = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float shared_local[];
    // Statically allocated shared memory
    float *s_a = &shared_local[0];
    float *s_b = &shared_local[blockDim.x * blockDim.y];

    // Accumulate in temporary variable
    int tmp = 0;

    // Sweep tile across matrix
    for (int i = 0; i < ncolsA; i += blockDim.x) {
      // Load in elements for this tile
      s_a[threadIdx.y * blockDim.x + threadIdx.x] = mat_a[row_id * ncolsA + i + threadIdx.x];
      s_b[threadIdx.y * blockDim.x + threadIdx.x] =
          mat_b[i * ncolsB + threadIdx.y * ncolsB + col_id];

      // Wait for both tiles to be loaded in before doing computation
      __syncthreads();

      // Do matrix multiplication on the small matrix
      for (int j = 0; j < blockDim.x; j++) {
        tmp +=
            s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
      }

      // Wait for all threads to finish using current tiles before loading in new
      // ones
      __syncthreads();
    }
    // Write back results
    result_mat[row_id * nrowsA + col_id] = tmp;
  
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
int nrows = 1024;
int ncols = 1024;
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
CudaSafeCall(cudaMalloc((void**)&d_mat_out, mat_size_bytes));
CudaSafeCall(cudaMalloc((void**)&d_mat_a,   mat_size_bytes));
CudaSafeCall(cudaMalloc((void**)&d_mat_b,   mat_size_bytes));
checkGPUMemory();
//------ Step 2: Copy Memory to the device-------
printf("Transfering data to the Device..\n");
CudaSafeCall(cudaMemcpy(d_mat_a, h_mat_a, mat_size_bytes, cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpy(d_mat_b, h_mat_b, mat_size_bytes, cudaMemcpyHostToDevice));
//------ Step 3: Prepare launch parameters-------
printf("preparing launch parameters..\n");
int block_size_x = 32;
int block_size_y = 32;
int SHMEM_SIZE = block_size_x * block_size_y * 2 * sizeof(float);
dim3 dimGrid = dim3((ncols + block_size_x - 1)/ block_size_x, (nrows + block_size_y - 1)/block_size_y, 1);//.... CONFIGURE THE GRID IN 2D NOW!
std::cout << "dimGrid = " << dimGrid.x << " x " << dimGrid.y << std::endl;
dim3 dimBlock = dim3(block_size_x, block_size_y, 1);//.... we have  alimit of 1024 threads per block!!!!!
std::cout << "dimBlock = " << dimBlock.x << " x " << dimBlock.y << std::endl;
//------ Step 4: Launch device kernel-------
printf("Launch Device Kernel.\n");
// YOUR KERNEL LAUNCH GOES HERE------------------------>>>>>>>>>
cudaTick(&ck);
gpu_matrix_mul<<<dimGrid, dimBlock, SHMEM_SIZE>>>(d_mat_out, d_mat_a, d_mat_b, nrowsA, ncolsA, nrowsB, ncolsB);
cudaTock(&ck, "kernel_matrix_mul");
CudaCheckError();

//------ Step 5: Copy Memory back to the host-------
printf("Transfering result data to the Host..\n");
CudaSafeCall(cudaMemcpy(h_mat_out_gpu, d_mat_out, mat_size_bytes, cudaMemcpyDeviceToHost));
 //
printf("CPU version...\n");
cpuTick(&cpuck);
host_matrix_mul(h_mat_out_cpu, h_mat_a, h_mat_b, nrowsA, ncolsA, nrowsB, ncolsB);//serial version to compare
cpuTock(&cpuck, "host_matrix_mul");
std::cout << "gpu is " << cpuck.elapsedMicroseconds/ck.elapsedMicroseconds << " times faster" << std::endl;

printf("Checking solutions..\n");
check_equal_float_vec(h_mat_out_gpu, h_mat_out_cpu, nrows * ncols);
//

// -----------Step 6: Free the memory --------------
printf("Deallocating device memory..\n");
CudaSafeCall(cudaFree(d_mat_out));
//.... FREE THE REST OF THE VECTORS
CudaSafeCall(cudaFree(d_mat_a));
CudaSafeCall(cudaFree(d_mat_b));

free(h_mat_a);
free(h_mat_b);
free(h_mat_out_cpu);
free(h_mat_out_gpu);

return 0;
}