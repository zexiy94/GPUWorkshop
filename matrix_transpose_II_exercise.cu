// nvcc -arch=sm_37 -run matrix_transpose_II_exercise.cu cuda_helper.cu
#include "cuda_runtime.h"
#include "chTimer.h"
#include "cuda_helper.h"
#include <stdio.h>
#define TILE_WIDTH 32


__global__ void gpu_matrix_transpose_shared(float *mat_out,
                                           float *mat_in,
                                           int nrows,
                                           int ncols)
{
    int col_id = blockIdx.x * blockDim.x + threadIdx.x;
    int row_id = blockIdx.y * blockDim.y + threadIdx.y;
    int col_out = blockIdx.y * blockDim.y + threadIdx.x;
    int row_out = blockIdx.x * blockDim.x + threadIdx.y;
    extern float sharedmem[];
    //printf(" %d %d, ", gidx, 1);
    if(col_id < ncols &&  row_id < nrows){
          = mat_in[col_id + ncols * row_id]; //load the shared memory
        //ADD SYNCHRONIZATION BARRIER
        mat_out[col_out + ncols * row_out] = ;//write from shared memory
    }
}

__global__ void gpu_matrix_transpose_naive(float *mat_out,
                                           float *mat_in,
                                           int nrows,
                                           int ncols)
{
  int col_id = blockIdx.x * blockDim.x + threadIdx.x;
  int row_id = blockIdx.y * blockDim.y + threadIdx.y;
  int col_out = row_id;
  int row_out = col_id;
  //printf(" %d %d, ", gidx, 1);
  if(col_id < ncols &&  row_id < nrows)
        mat_out[col_out + ncols * row_out] = mat_in[col_id + ncols * row_id];

}
/////////////////////////////////Serial version/////////////////////////////////////////////
void host_matrix_transpose(float *mat_out,
                           float *mat_in,
                           int nrows,
                           int ncols)
{
	for(int row = 0; row < nrows; row++){
      for(int col = 0; col < ncols; col++){
		      mat_out[col * ncols + row] = mat_in[row * ncols + col] ;
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
cudaClock ck, ck2;

float *d_mat_out, *d_mat_in;
float *h_mat_out_cpu, *h_mat_out_gpu, *h_mat_in;
printf("\nStarting program execution..\n\n");
int nrows = 4096;
int ncols = 4096;
std::cout << "rows " << nrows << " x cols " << ncols<<std::endl;

printf("Allocating and creating problem data..\n");
int mat_size_bytes = nrows * ncols * sizeof(float);
std::cout << "mat_size in bytes: " << mat_size_bytes << std::endl;
//allocation of host memory
h_mat_out_cpu = (float*)malloc(mat_size_bytes);
h_mat_out_gpu = (float*)malloc(mat_size_bytes);
h_mat_in = (float*)malloc(mat_size_bytes);

for(int i =0; i < nrows * ncols; i++){
	h_mat_in[i] = (rand()%10)/10.0f;
}
////
 //------ Step 1: Allocate the memory-------
 printf("Allocating Device Memory..\n");
CudaSafeCall(cudaMalloc((void**)&d_mat_out, mat_size_bytes));
CudaSafeCall(cudaMalloc((void**)&d_mat_in,   mat_size_bytes));
checkGPUMemory();
//------ Step 2: Copy Memory to the device-------
printf("Transfering data to the Device..\n");
CudaSafeCall(cudaMemcpy(d_mat_in, h_mat_in, mat_size_bytes, cudaMemcpyHostToDevice));
//------ Step 3: Prepare launch parameters-------
printf("preparing launch parameters..\n");
int block_size_x = 32;
int block_size_y = 32;
int MEMUSAGE = block_size_x*block_size_y*sizeof(float);
dim3 dimGrid = dim3((ncols + block_size_x - 1)/ block_size_x, (nrows + block_size_y - 1)/block_size_y, 1);//.... CONFIGURE THE GRID IN 2D NOW!
std::cout << "dimGrid = " << dimGrid.x << " x " << dimGrid.y << std::endl;
dim3 dimBlock = dim3(block_size_x, block_size_y, 1);//.... we have  alimit of 1024 threads per block!!!!!
std::cout << "dimBlock = " << dimBlock.x << " x " << dimBlock.y << std::endl;
//------ Step 4: Launch device kernel-------
printf("Launch Device Kernel.\n");
// YOUR KERNEL LAUNCH GOES HERE------------------------>>>>>>>>>
cudaTick(&ck);
gpu_matrix_transpose_naive<<<dimGrid, dimBlock>>>(d_mat_out, d_mat_in, nrows, ncols);
cudaTock(&ck, "gpu_matrix_transpose_naive");
CudaCheckError();

cudaTick(&ck2);
gpu_matrix_transpose_shared<<<dimGrid, dimBlock,MEMUSAGE>>>(d_mat_out, d_mat_in, nrows, ncols);
cudaTock(&ck2, "gpu_matrix_transpose_shared");
CudaCheckError();
std::cout << "new version is " << ck.elapsedMicroseconds/ck2.elapsedMicroseconds << " times faster that naive version" << std::endl;

//------ Step 5: Copy Memory back to the host-------
printf("Transfering result data to the Host..\n");
CudaSafeCall(cudaMemcpy(h_mat_out_gpu, d_mat_out, mat_size_bytes, cudaMemcpyDeviceToHost));
 //
printf("CPU version...\n");
cpuTick(&cpuck);
host_matrix_transpose(h_mat_out_cpu, h_mat_in, nrows, ncols);//serial version to compare
cpuTock(&cpuck, "host_matrix_transpose");
std::cout << "naive version gpu is " << cpuck.elapsedMicroseconds/ck.elapsedMicroseconds << " times faster" << std::endl;
std::cout << "shared version gpu is " << cpuck.elapsedMicroseconds/ck2.elapsedMicroseconds << " times faster" << std::endl;

printf("Checking solutions..\n");
check_equal_float_vec(h_mat_out_gpu, h_mat_out_cpu, nrows * ncols);
//

// -----------Step 6: Free the memory --------------
printf("Deallocating device memory..\n");
CudaSafeCall(cudaFree(d_mat_out));
CudaSafeCall(cudaFree(d_mat_in));

free(h_mat_in);
free(h_mat_out_cpu);
free(h_mat_out_gpu);
return 0;
}
