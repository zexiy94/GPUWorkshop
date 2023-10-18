#include <stdio.h>
#include <stdlib.h>
__global__ void init_float_vector_w_factor(float* vector, float factor, int size){
	int gid = threadIdx.x + blockIdx.x * blockDim.x;

	if(gid < size){
		vector[gid] = factor;
	}	
}


__global__ void shuffle_tests(float* professor_cost, float* professor_factor, int* turmas_course, int* professor_turmas, int professor_size, int maxturma_size){
	//int laneid = threadIdx%32;
	int laneid = threadIdx.x;
	//int warpid = threadIdx/32;
	int professorid = blockIdx.x; // for the time being...
	int calculateFlag = 1;  
	int otherCourse;
	
	if(professorid < professor_size){
		float myfactor = professor_factor[professorid];
		int myturma = professor_turmas[professorid * maxturma_size +laneid];
		int myCourse = turmas_course[myturma];
		for(int i=0; i< 32; i++){
		  otherCourse = __shfl(myCourse,i);
		  if((i < laneid)&&(otherCourse == myCourse)) calculateFlag--;
		}
		if(calculateFlag == 1) int istat = atomicAdd(&professor_cost[professorid],myfactor); 
	}
}

int main(int argc, char** argv){

  cudaDeviceReset();
  printf("Test start...\n");
  int *h_turmas_course, *h_professor_turmas;
  int *d_turmas_course, *d_professor_turmas;
  float *h_professor_cost, *d_professor_cost;
  float *h_professor_factor, *d_professor_factor;
  
  int professor_size = 320;
  int turmas_size = 1280;
  int course_size = 23;
  int maxturma_size = 32;
  
  //long time_start, time_end;
  cudaEvent_t gstart, gstop;
  cudaEventCreate(&gstart);
  cudaEventCreate(&gstop);
  
 printf("Allocating host memory\n");
 h_turmas_course = (int*)malloc(turmas_size*sizeof(int));
 h_professor_turmas = (int*)malloc(professor_size*maxturma_size*sizeof(int)); 
 h_professor_cost = (float*)malloc(professor_size*sizeof(float)); 
 h_professor_factor = (float*)malloc(professor_size*sizeof(float)); 
  
  for(int i =0 ; i < turmas_size; i++){
	h_turmas_course[i] = rand()%course_size; //vector of the tummas size indicating to which disciplina they belong
  }
  for(int i =0 ; i < professor_size*maxturma_size; i++){
	h_professor_turmas[i] = rand()%turmas_size; //vector of the tummas size indicating to which disciplina they belong
  }
   for(int i =0 ; i < professor_size; i++){
	h_professor_factor[i] = (float)(1+rand()%3); //vector of the professor size indicating the cost of that type of professor
  }
  
  printf("Allocating device memory\n");
  cudaMalloc((void**) &d_turmas_course, turmas_size*sizeof(int));
  cudaMalloc((void**) &d_professor_turmas,professor_size*maxturma_size*sizeof(int));
  cudaMalloc((void**) &d_professor_cost,professor_size*sizeof(float));
  cudaMalloc((void**) &d_professor_factor,professor_size*sizeof(float));
  
  printf("transfer memory host to device\n");
  cudaMemcpy(d_turmas_course, h_turmas_course, turmas_size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_professor_turmas, h_professor_turmas, professor_size*maxturma_size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_professor_factor, h_professor_factor, professor_size*sizeof(int), cudaMemcpyHostToDevice);
  
  //initializing cost vector
  
  init_float_vector_w_factor<<<(professor_size + 255)/256,256>>>(d_professor_cost, 0.0f, professor_size);
  
  //setting up the launch parameters
  dim3 shuffleGrid = dim3(professor_size, 1, 1);//we launch one warp per professor
  dim3 shuffleBlock = dim3(32, 1, 1);
  printf("dimGrid: %d \n", shuffleGrid.x);
  printf("dimBlock: %d \n", shuffleBlock.x);
  
   printf("launching kernel..");
   cudaEventRecord(gstart, 0);
   shuffle_tests<<<shuffleGrid, shuffleBlock>>>(d_professor_cost, d_professor_factor, d_turmas_course, d_professor_turmas, professor_size, maxturma_size);
   cudaEventRecord(gstop,0);
   cudaEventSynchronize(gstop);
   float gputime;
   cudaEventElapsedTime(&gputime, gstart, gstop); 
   printf("Calculations took: %f miliseconds \n", gputime);   
   printf("transfer memory device to host\n");
   
   cudaMemcpy(h_professor_cost, d_professor_cost, professor_size*sizeof(float), cudaMemcpyDeviceToHost);
   
   //checking wether it works...
   float totresult=0.0f;
   for(int j =0; j< professor_size; j++){
	totresult += h_professor_cost[j];
   }
   printf("criteria total cost: %f, against naive cost estimate: %f \n", totresult, professor_size*maxturma_size*2.0f);
  
printf("Cleaning up device memory\n");
   cudaFree(d_professor_cost);
   cudaFree(d_professor_turmas);
   cudaFree(d_professor_factor);
   cudaFree(d_turmas_course);
   printf("Cleaning up host memory\n");
   free(h_professor_cost);
   free(h_professor_turmas);
   free(h_professor_factor);
   free(h_turmas_course);
   
   printf("Goodbye!\n");
   
   return 0;

}