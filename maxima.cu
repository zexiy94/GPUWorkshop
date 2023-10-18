#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <vector>
#include <thrust/extrema.h>

int main()
{
  std::vector<float> v(40,1);
  thrust::device_vector<float> devicio = v;

  thrust::device_vector<float>::iterator iter =
  thrust::max_element(devicio.begin(), devicio.begin());

  unsigned int position = iter - devicio.begin();
  float max_val = *iter;

  std::cout << "The maximum value is " << max_val << " at position " << position << std::endl;



  return 261;
}
