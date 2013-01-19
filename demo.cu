#include "collective_ptr.hpp"
#include <thrust/device_vector.h>
#include <cstdio>
#include <iostream>

__global__ void single_reader_hazard(thrust::device_ptr<int> result)
{
  __shared__ int storage;
  collective_ptr<int> s_obj(&storage);

  if(threadIdx.x == 0)
  {
    *s_obj = 1;
  }

  // XXX missing barrier here

  if(threadIdx.x == 1)
  {
    // read after write
    int val = *s_obj;

    *result = val;
  }
}

__global__ void multiple_writers_hazard(thrust::device_ptr<int> result)
{
  __shared__ int storage;
  collective_ptr<int> s_obj(&storage);

  // XXX all threads write to the variable at once
  *s_obj = 1;

  s_obj.barrier();

  if(threadIdx.x == 0)
  {
    *result = *s_obj;
  }
}

__global__ void race_free(thrust::device_ptr<int> result)
{
  __shared__ int storage;
  collective_ptr<int> s_obj(&storage);

  if(threadIdx.x == 0)
  {
    *s_obj = 1;
  }

  s_obj.barrier();

  if(threadIdx.x == 1)
  {
    // read after write
    int val = *s_obj;

    *result = val;
  }
}

int main()
{
  thrust::device_vector<int> vec(1);

  race_free<<<1,128>>>(vec.data());

  std::cout << "race_free result is " << vec[0] << std::endl;

  single_reader_hazard<<<1,128>>>(vec.data());

  std::cout << "single_reader_hazard result is " << vec[0] << std::endl;

  multiple_writers_hazard<<<1,128>>>(vec.data());

  std::cout << "multiple_writers_hazard result is " << vec[0] << std::endl;

  return 0;
}


