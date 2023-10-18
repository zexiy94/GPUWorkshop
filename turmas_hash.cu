#include "cuda_runtime.h"
#include <stdio.h>

#define TURMAS_SIZE 512
#define PROFESSOR_SIZE 128
#define DAY_SIZE 24
struct sparseIntRep {
   int* array;
   int* count;
   int* init;
   int dim;  //number of rows, size of the count and init arrays
   int cSize; //current size in the structure
   int maxSize; //in case is necesary
};


int hostGetHash(int day, int slot, int numSlots){
    return day*numSlots + slot;
}
void intExcPrefixSum(int* scanned, int* input, int size ){
	scanned[0] = 0; 
	for(int i = 1; i < size; i++) scanned[i] = scanned[i-1]+input[i-1];
}
void intIncPrefixSum(int* scanned, int* input, int size ){
	scanned[0] = input[0]; 
	for(int i = 1; i < size; i++) scanned[i] = scanned[i-1]+input[i];
}
int intSum( int* input, int size ){
	int tempSum = 0; 
	for(int i = 0; i < size; i++) tempSum += input[i];
	return tempSum;
}
  
//////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
//int* professor_allowed_turmas;
//int* professor_allowed_init;
//int* professor_allowed_count; 
sparseIntRep pa; //pa = professor_available_turmas
sparseIntRep ts; // turmas slots
int* turmas_day; //integer 0 to 4
//int* turmas_slot_count; //number of slots each turmas uses
int* turmas_hour_slot; //integer 0 to 23 (DAY_SIZE)
//int* turmas_slot_init;
int* turmas_hash;
int* hash_counter;
//allocation of host memory
////////////////////////////////////////////////////////////////////////////////
pa.dim = PROFESSOR_SIZE;
pa.count = (int*)malloc(pa.dim*sizeof(int)); 
pa.init = (int*)malloc(pa.dim*sizeof(int)); 
for(int i=0; i< pa.dim; i++)pa.count[i] = rand()%6;
intExcPrefixSum(pa.init, pa.count, pa.dim);

pa.cSize = intSum(pa.count, pa.dim);

pa.array = (int*)malloc(pa.cSize*sizeof(int));

 
//generate random example data
for(int i=0; i< pa.dim; i++){
  int count = pa.count[i];
  int init = pa.init[i];
  for(int j =0; j< count; j++){
	pa.array[init +j] = rand()%TURMAS_SIZE;
  }
}

ts.dim = TURMAS_SIZE;
ts.count = (int*)malloc(ts.dim*sizeof(int));
ts.init = (int*)malloc(ts.dim*sizeof(int));
for(int i=0; i< ts.dim; i++)ts.count[i] = rand()%4;
intExcPrefixSum(ts.init, ts.count, ts.dim);
ts.cSize = intSum(ts.count, ts.dim);


turmas_hour_slot = (int*)malloc(ts.cSize*sizeof(int));
turmas_day = (int*)malloc(ts.cSize*sizeof(int));
//turmas_hash = (int*)malloc(ts.cSize*sizeof(int));
ts.array = (int*)malloc(ts.cSize*sizeof(int));


//generate random example data
for(int i=0; i< ts.dim; i++){
  int count = ts.count[i];
  int init = ts.init[i];
  for(int j =0; j< count; j++){
	turmas_day[init +j] = rand()%5;
	turmas_hour_slot[init + j] = rand()%24;
	
	//we take the chance to calculate the hash code for this turma slot
	ts.array[init + j] = hostGetHash(turmas_day[init +j],turmas_hour_slot[init + j], DAY_SIZE);
  }
}

 ////
 //------ Step 1: Allocate the memory-------
 printf("Allocating Device Memory..\n");

//.... ALLOCATE THE REST OF THE VECTORS

//------ Step 2: Copy Memory to the device-------
printf("Transfering data to the Device..\n");

//.... COPY THE REST OF THE VECTORS

//------ Step 3: Prepare launch parameters-------
printf("preparing launch parameters..\n");


//------ Step 4: Launch device kernel-------
printf("Launch Device Kernel.\n");

// YOUR KERNEL LAUNCH GOES HERE------------------------>>>>>>>>>


//------ Step 5: Copy Memory back to the host-------
printf("Transfering result data to the Host..\n");

 
 //
 printf("CPU version...\n");

printf("Checking solutions..\n");

//

// -----------Step 6: Free the memory -------------- 
printf("Deallocating device memory..\n");

//.... FREE THE REST OF THE VECTORS



return 0;
}

// cudaEvent_t gstart, gstop;
// cudaEventCreate(&gstart);
// cudaEventCreate(&gstop);