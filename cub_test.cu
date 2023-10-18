//nvcc -run -arch=sm_61 cub_test.cu
#include <stdio.h>
#include<vector>
#include<iostream>
#include<cuda_runtime.h>
#include "cub/cub.cuh"

#define SIZE 4865*1216*4*8

int main()
{

  cudaDeviceReset();
 cudaSetDevice(0);
  int *v;
  int *ids;
  v = (int*)malloc(SIZE * sizeof(int));
  ids = (int*)malloc(SIZE * sizeof(int));

  for(int i = 0; i < SIZE; i++){
    v[i] = rand()%(SIZE*4) + 32;
    ids[i] = i;
  }

  std::cout << "keys before: ";
  for(int i=0;i<16;i++)
    std::cout << " " << v[i];
  std::cout << std::endl;
  //
  std::cout << "indexes before: ";
  for(int i=0;i<16;i++)
    std::cout << " " << ids[i];
  std::cout << std::endl;

  int *dids, *dv;
  void *dtemp;

  cudaMalloc((void**) &dids, SIZE * sizeof(int));
  cudaMalloc((void**) &dv, SIZE * sizeof(int));
  cudaMalloc((void**) &dtemp, SIZE * 2 * sizeof(int));

  cudaMemcpy(dids, ids, SIZE * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dv, v, SIZE * sizeof(int), cudaMemcpyHostToDevice);

  void *dsize = NULL;
  size_t expected_size = 0;
  cub::DeviceRadixSort::SortPairs(
                        dsize,
                        expected_size,
                        (const int*)dv,
                        dv,
                        (const int*)dids,
                        dids,
                        SIZE,
                        0,
                        32,
                        0,
                        true);

  std::cout << "SIZE " << expected_size << std::endl;
cudaGetLastError();
  cub::DeviceRadixSort::SortPairs(
                        dtemp,
                        expected_size, //size,
                        (const int*)dv,
                        dv,
                        (const int*)dids,
                        dids,
                        SIZE,
                        0,
                        32,
                        0,
                        true);
cudaDeviceSynchronize();
cudaGetLastError();

  std::cout << "----------------------------------\n";
  int *v2; // = {1,2,3,4,5,6,7,8,9,10,11,12};
  int *ids2;// = {5,1,3,1,2,4,9,8,7,2,3,3};
  v2 = (int*)malloc(SIZE * sizeof(int));
  ids2 = (int*)malloc(SIZE * sizeof(int));

  cudaMemcpy(v2, dv, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(ids2, dids, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

std::cout << "keys after: ";
  for(int i=0;i<16;i++)
    std::cout << " " << v2[i];
  std::cout << std::endl;
  //
std::cout << "indexes after: ";
  for(int i=0;i<16;i++)
    std::cout << " " << ids2[i];
  std::cout << std::endl;

free(v); free(ids); free(v2); free(ids2);

  cudaFree(dids);
  cudaFree(dv);
  cudaFree(dtemp);
	return 0;
}
