/*
This function gives the winner (min value) in a torunament fasion.... the parameter depth stops the tournament in the proper round
*/
#include "stdio.h"
#define SIZE 2048
#define NTHREADS 512
__global__ void tournament_index( int* winners, float *fitness, int depth){
	__shared__ float s_fitness[NTHREADS]; //shared memory
	__shared__ int s_winner[NTHREADS]; //shared memory
	
	int tid = threadIdx.x;
	int gid = threadIdx.x + blockIdx.x*blockDim.x;
	
	s_fitness[tid] = fitness[gid];
	s_winner[tid] = gid;

	__syncthreads(); //always
	for(int step = 0; step < depth ;step++){ //using s<<=1 is faster
		int challenger = 1 << step;
		
		if(tid%challenger==0 && (tid+challenger) < NTHREADS) {
			if(s_fitness[tid] > s_fitness[tid + challenger] ){
				s_fitness[tid] = s_fitness[tid + challenger];
				s_winner[tid] = s_winner[tid + challenger];
			}
		}
		__syncthreads(); //again.
		
	}
	
	//winners[gid] = s_winner[tid];
	int breadth = 1 << depth;//pow(2, depth);
	winners[gid] = s_winner[tid - tid%breadth];
	//if (gid==0) printf("MY winner in host memory %d\n",winners[gid]);
}

int main(){
  float* h_scores = (float*)malloc(SIZE*sizeof(float));
  int* h_winners = (int*)malloc(SIZE*sizeof(int));
  float* d_scores;
  int* d_winners;
  cudaMalloc((void**)&d_scores, SIZE*sizeof(float));
  cudaMalloc((void**)&d_winners, SIZE*sizeof(int));
  
  printf("starting the execution\n");
  
  printf("%d", 1<< 9);
  
  for(int i =0; i< SIZE; i++){
	h_scores[i] = 28; //rand()%10;
	//if(i%123 == 7)h_scores[i] = -11000.0f;
	h_winners[i] = 77;
  }
   h_scores[28] = -111.0f;
  h_scores[193] = -11111.0f;
   h_scores[1564] = -111.0f;
  h_scores[1729] = -11111.0f;
  printf("Done initializing variables\n");
 
  cudaMemcpy(d_scores, h_scores, SIZE*sizeof(float), cudaMemcpyHostToDevice);
  
  tournament_index<<<SIZE/NTHREADS,NTHREADS>>>(d_winners, d_scores, 9);
  
  
 cudaMemcpy(h_winners, d_winners, SIZE*sizeof(int), cudaMemcpyDeviceToHost);

  printf("Done transfering back results\n");
  for(int i=0; i< SIZE; i++){
	printf("%d ", h_winners[i]);
	if((i+1)%NTHREADS == 0) printf("\n\n");
	
  }
  
  printf("\n \n -> %f\n",h_scores[h_winners[0]]);
 printf("\n \n -> %f\n",h_scores[24]);
 free(h_scores);
 free(h_winners);
 cudaFree(d_scores);
 cudaFree(d_winners);
  return 0;
}
