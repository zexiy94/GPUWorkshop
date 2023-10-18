#include <stdio.h>
#define SIZE 2048*2048*4
#define BLOCK_SIZE 256
#define NSTREAMS 4 //32
// our kernel... identified by the keyword __global__
__global__ void calc(int size, float* input, float* prices, float* output){
	int gid = blockIdx.x * blockDim.x + threadIdx.x; //we calculate the thread global index
	if(gid<size) output[gid] = input[gid] * prices[gid];
}
 void serial_calc (int size, float* input, float* prices, float* output){
// serial implementation of the vector add.
	for (int i =0; i< size; i++){
		output[i] = input[i] + prices[i];
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

 cudaDeviceReset();
//configuramos la medida de tiempo de la CPU.
  LARGE_INTEGER ticksPerSecond;
  LARGE_INTEGER tick;  
  LARGE_INTEGER tock;

 // get the high resolution counter's accuracy
 QueryPerformanceFrequency(&ticksPerSecond);
 
 
 printf("Starting the execution!\n");
 if(cudaSuccess != cudaGetLastError()) printf("Previous error in the system");
 
 cudaStream_t *streams = (cudaStream_t *) malloc(NSTREAMS * sizeof(cudaStream_t));
 for(int is =0; is < NSTREAMS; is++){
	cudaStreamCreate(&(streams[is]));
	if(cudaSuccess != cudaGetLastError()) printf("Error creating the %d stream", is);
 }
 
 float *h_input, *h_prices, *h_output_gpu, *h_output_cpu;
 float *d_input[NSTREAMS];
 float *d_prices[NSTREAMS]; 
 float *d_output[NSTREAMS];
  
 int size_bytes = SIZE * sizeof(float);
 
 printf("Allocating host memory!\n");
 h_input = (float*)malloc(size_bytes);
 h_prices = (float*)malloc(size_bytes);
 h_output_gpu = (float*)malloc(size_bytes);
 h_output_cpu = (float*)malloc(size_bytes);
 
 printf("Allocating device memory!\n");
 for(int is =0; is < NSTREAMS; is++){
	cudaMalloc((void**) &d_input[is], size_bytes);
	cudaMalloc((void**) &d_prices[is], size_bytes);
	cudaMalloc((void**) &d_output[is], size_bytes);
	if(cudaSuccess != cudaGetLastError()) printf("Error allocating device memory component %d", is);
 }
 
  
 
 printf("Copying memory from host to device!\n");
  for(int is =0; is < NSTREAMS; is++){
	cudaMemcpyAsync(d_input[is], h_input, size_bytes, cudaMemcpyHostToDevice, streams[is]);
	cudaMemcpyAsync(d_prices[is], h_prices, size_bytes, cudaMemcpyHostToDevice, streams[is]);
	if(cudaSuccess != cudaGetLastError()) printf("Error memcpy stream %d", is);
 }
 
 dim3 dimGrid = dim3((SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE, 1, 1);
 dim3 dimBlock = dim3(BLOCK_SIZE,1,1);
 
  cudaEvent_t gstart,gstop;//no os fijéis en estas líneas
 cudaEventCreate(&gstart);//nada importante
 cudaEventCreate(&gstop);//nada que ver...continuen..
 
 //PASO 4: Lanzamos el kernel
 printf("PASO 4: Lanzamos el kernel...");
 cudaEventRecord(gstart, 0);//nada importante...
 
 
 for(int is =0; is < NSTREAMS; is++){ 
	add<<<dimGrid, dimBlock, 0, streams[is]>>>(SIZE, d_input[is], d_prices[is], d_output[is]);
	//add<<<dimGrid, dimBlock, 0, streams[is]>>>(SIZE, d_input_A[1], d_input_B[0], d_output[is]);
 }
 if(cudaSuccess != cudaGetLastError()) printf("Error in Stream calculation");
 
 cudaEventRecord(gstop, 0);//lalalalalala
 cudaEventSynchronize(gstop);//sigue sin importar
 printf("ok!\n\n");
 
 float gpu_time; 
 cudaEventElapsedTime(&gpu_time, gstart, gstop);
 printf("La version GPU ha necesitado %.3f milisegundos\n\n",gpu_time );
 
 cudaEventDestroy(gstart); //limpiando un pelín
 cudaEventDestroy(gstop); //insisto..nada que ver..circulen
 
 printf("Copying memory back from the device to the host!\n");
 cudaMemcpyAsync(h_output_gpu, d_output[0], size_bytes, cudaMemcpyDeviceToHost, streams[0]);
 if(cudaSuccess != cudaGetLastError()) printf("Error coying memory back to the host ");
 
 printf("Deallocating GPU memory \n");
  for(int is =0; is < NSTREAMS; is++){
	cudaFree(d_input[is]);
	cudaFree(d_prices[is]);
	cudaFree(d_output[is]); 
	if(cudaSuccess != cudaGetLastError()) printf("Error deallocating device memory component %d", is);
 }
   for(int is =0; is < NSTREAMS; is++){
	cudaStreamDestroy(streams[is]);

	if(cudaSuccess != cudaGetLastError()) printf("Error destroying stream %d", is);
 }
 
 printf("Deallocating CPU memory \n");
 free(h_input);
 free(h_prices);
 free(h_output_gpu); 
 free(h_output_cpu); 
 printf("That's all!\n");

    return 0;
}

