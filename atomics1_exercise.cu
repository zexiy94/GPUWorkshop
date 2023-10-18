#include "cuda_runtime.h"
#include <stdio.h>


__global__ void kernel_atomics(int *count_vec, int *input_vec, int num_rows, int num_cols){
  ////// YOUR CODE GOES HERE!////////////////////////////////////////////
  //step 1.. define the thread ID 
  
  //check that you are not accessing out of array bounds
  
		// loop throu prfessors subjects IDs
		
		//Use that IDs to write to the proper memory location with atomics..
  
  
}
/////////////////////////////////Serial version/////////////////////////////////////////////
void host_serial(int *count_vec, int *input_vec, int num_rows, int num_cols){
	
	for(int row = 0; row < num_rows; row++){
		for (int col = 0; col < num_cols; col++){
			int pos = input_vec[col + row * num_cols] ;
			count_vec[pos] +=1; 
		}
	}
	
}
/////////////////////////////////Diagnostic routines/////////////////////////////////////////////
int check_equal_int_vec(int *vec1,int *vec2,int size){
	int numerrors=0;
	for(int i =0; i< size; i++){
		if(vec1[i] != vec2[i]) numerrors++;
	}
	if(numerrors ==0)printf("Congratulations you have 0 errors!\n");
	if(numerrors >0)printf("Wrong results, you have %d errors!\n", numerrors);
	
	return numerrors;
}
//////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
int *d_prof_mat, *d_subject_prof_count, *h_prof_mat, *h_subject_prof_count, *h_subject_prof_count_cpu;
printf("\nStarting program execution..\n\n");
int subject_size = 390;
int professor_size = 20000;
int subject_per_professor = 5;
int prof_mat_size = professor_size * subject_per_professor;
printf("Allocating and creating problem data..\n");
int prof_mat_size_bytes = prof_mat_size * sizeof(int);
int subject_size_bytes = subject_size*sizeof(int);


h_prof_mat = (int*)malloc(prof_mat_size_bytes);
h_subject_prof_count = (int*)malloc(subject_size_bytes);
h_subject_prof_count_cpu = (int*)malloc(subject_size_bytes);

for(int i =0; i < prof_mat_size; i++){
h_prof_mat[i] = rand()%subject_size; //generate random values...
}

for(int i =0; i < subject_size; i++){
	h_subject_prof_count[i] = 0; //initialize values...
	h_subject_prof_count_cpu[i] = 0; //initialize values...
}
 ////
 //------ Step 1: Allocate the memory-------
 printf("Allocating Device Memory..\n");
cudaMalloc((void**)&d_prof_mat, prof_mat_size_bytes);
cudaMalloc((void**)&d_subject_prof_count, subject_size_bytes);
//------ Step 2: Copy Memory to the device-------
printf("Transfering data to the Device..\n");
cudaMemcpy(d_prof_mat, h_prof_mat, prof_mat_size_bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_subject_prof_count, h_subject_prof_count, subject_size_bytes, cudaMemcpyHostToDevice);
//------ Step 3: Prepare launch parameters-------
printf("preparing launch parameters..\n");
dim3 dimGrid = dim3(( professor_size + 255)/256,1,1); //we will launch one thread per professor
dim3 dimBlock = dim3(256,1,1);
//------ Step 4: Launch device kernel-------
printf("Launch Device Kernel.\n");

// YOUR KERNEL LAUNCH GOES HERE------------------------>>>>>>>>>


//------ Step 5: Copy Memory back to the host-------
printf("Transfering result data to the Host..\n");
cudaMemcpy(h_subject_prof_count, d_subject_prof_count, subject_size_bytes, cudaMemcpyDeviceToHost);
 
 //
 printf("CPU version...\n");
host_serial(h_subject_prof_count_cpu, h_prof_mat, professor_size, subject_per_professor);//serial version to compare 
printf("Checking solutions..\n");
check_equal_int_vec(h_subject_prof_count, h_subject_prof_count_cpu, subject_size);
//

// -----------Step 6: Free the memory -------------- 
printf("Deallocating device memory..\n");
cudaFree(d_prof_mat);
cudaFree(d_subject_prof_count);

free(h_prof_mat);
free(h_subject_prof_count);

return 0;
}

// cudaEvent_t gstart, gstop;
// cudaEventCreate(&gstart);
// cudaEventCreate(&gstop);