/*
Copyright 2017 the anode authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef __ANODE_KERNELS_GPU_COMMON_REDUCE_CUH__
#define __ANODE_KERNELS_GPU_COMMON_REDUCE_CUH__

#include "common.cuh"

template <typename T, typename Reduce>
__forceinline__ __device__ void threadblock_reduce_sync(T *cache)
{
  for (uint32_t s = (blockDim.x >> 1); s > 0; s >>= 1) {
    if (s > threadIdx.x && s + threadIdx.x < blockDim.x) {
      Reduce::Reduce(&cache[threadIdx.x], cache[s + threadIdx.x]);
    }
    __syncthreads();
  }
}

template <typename T, typename Reduce>
__forceinline__ __device__ void threadblock_reduce1024_sync(T *cache)
{
  for (uint32_t s = 512; s > 0; s >>= 1) {
    if (s > threadIdx.x && s + threadIdx.x < blockDim.x) {
      Reduce::Reduce(&cache[threadIdx.x], cache[s + threadIdx.x]);
    }
    __syncthreads();
  }
}

#endif
