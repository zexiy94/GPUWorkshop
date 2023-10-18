
//nvcc -run -arch=sm_61 cub_test_2.cu
#include <stdio.h>
#include<vector>
#include<iostream>
#include<cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include "cub/cub.cuh"
#include <thrust/device_vector.h>


int main()
{

 cudaDeviceReset();


  int n_elems = 1024*131 +256 +0;//1024*4;


  std::vector<int> keys(n_elems);
  std::vector<int> values (n_elems);

  for(int i=0;i<n_elems;i++)
  {
    keys[i] = rand() % 100000;
    values[i] = i;
  }

  thrust::device_vector<int> t_keys = keys;
  thrust::device_vector<int> t_values = values;

  thrust::sort_by_key(thrust::device,
                      t_keys.begin(),
                      t_keys.end(),
                      t_values.begin(),
                      thrust::less<int>());



  int *d_keys;
  int *d_values;

 cudaMalloc((void**) &d_keys, n_elems * sizeof(int));
 cudaMalloc((void**) &d_values, n_elems * sizeof(int));

 cudaMemcpy(d_keys, keys.data(), n_elems * sizeof(int), cudaMemcpyHostToDevice);
 cudaMemcpy(d_values, values.data(), n_elems * sizeof(int), cudaMemcpyHostToDevice);

 void *dtemp = NULL;
 size_t expected_size = 0;
 cub::DeviceRadixSort::SortPairs(
                       dtemp,
                       expected_size,
                       d_keys,
                       d_keys,
                       d_values,
                       d_values,
                       n_elems,
                       0,
                       32,
                       0,
                       true);

 std::cout << "Temp size " << expected_size << std::endl;
 cudaMalloc((void**) &dtemp, expected_size * sizeof(int));

cudaGetLastError();
 cub::DeviceRadixSort::SortPairs(
                       dtemp,
                       expected_size, //size,
                       d_keys,
                       d_keys,
                       d_values,
                       d_values,
                       n_elems,
                       0,
                       32,
                       0,
                       true);
cudaDeviceSynchronize();
cudaGetLastError();

  std::vector<int> cub_keys_output(n_elems);
  std::vector<int> cub_values_output(n_elems);

 cudaMemcpy(cub_keys_output.data(), d_keys, n_elems * sizeof(int), cudaMemcpyDeviceToHost);
 cudaMemcpy(cub_values_output.data(), d_values, n_elems * sizeof(int), cudaMemcpyDeviceToHost);

/*std::cout << "keys after: ";
 for(int i=0;i<16;i++)
   std::cout << " " << v2[i];
 std::cout << std::endl;*/
 //
std::cout << "indexes after: ";
 for(int i=0;i<16;i++)
   std::cout << " " << cub_values_output[i];
 std::cout << std::endl;


int diff_count = 0;
 for(int i=0;i<n_elems;i++)
 {
   if(cub_keys_output[i] != t_keys[i]) diff_count++;
    //std::cout << "Sorted differently!!! cub: " << cub_keys_output[i] << " thrust:" << t_keys[i] << "\n";
 }
if(0 == diff_count)std::cout << "Sorted the same!!!  " << diff_count << " diferences"<< std::endl;
else std::cout << "Sorted differently!!!  " << diff_count << " difefrences"<< std::endl;

 cudaFree(d_keys);
 cudaFree(d_values);
 cudaFree(dtemp);
    return 0;
}
