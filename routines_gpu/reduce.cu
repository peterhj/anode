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

#include "lib.h"
#include "common.cuh"
#include "common_reduce.cuh"
#include <cassert>
#include <cuda_runtime.h>

template <typename T>
class CopyMap {
public:
  __forceinline__ __device__ static T Map(T x) {
    return x;
  }
};

template <typename T>
class SquareMap {
public:
  __forceinline__ __device__ static T Map(T x) {
    return x * x;
  }
};

template <typename T, typename Map, typename Reduce>
__global__ void anode_gpu_map_reduce_Iab_Ob_packed_deterministic_kernel(
    uint32_t reduce_dim,
    uint32_t outer_dim,
    const T *x,
    T *y)
{
  extern __shared__ T cache[];
  for (uint32_t blk = gblock(); blk < outer_dim; blk += gblockcount()) {
    T accumulator = Reduce::InitVal();
    uint32_t rdup_reduce_dim = (reduce_dim + blockDim.x - 1) / blockDim.x * blockDim.x;
    for (uint32_t i = threadIdx.x; i < rdup_reduce_dim; i += blockDim.x) {
      if (i < reduce_dim) {
        cache[threadIdx.x] = Map::Map(x[Index2::Pack(i, reduce_dim, blk)]);
      } else {
        cache[threadIdx.x] = Reduce::InitVal();
      }
      __syncthreads();
      threadblock_reduce_sync<T, Reduce>(cache);
      if (0 == threadIdx.x) {
        Reduce::Reduce(&accumulator, cache[0]);
      }
      __syncthreads();
    }
    y[blk] = accumulator;
  }
}

extern "C" void anode_gpu_sum_Iab_Ob_packed_deterministic_f32(
    uint32_t reduce_dim,
    uint32_t outer_dim,
    const float *x,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  assert(check_power_of_2(cfg->flat_block_dim().x));
  anode_gpu_map_reduce_Iab_Ob_packed_deterministic_kernel<float, CopyMap<float>, AddReduce<float>><<<cfg->flat_block_count(outer_dim), cfg->flat_block_dim(), cfg->flat_block_len() * sizeof(float), stream>>>(
      reduce_dim, outer_dim, x, y);
}

template <typename T, typename Map, typename Reduce>
__global__ void anode_gpu_map_reduce_Iabc_Ob_packed_deterministic_kernel(
    uint32_t reduce_inner_dim,
    uint32_t mid_dim,
    uint32_t reduce_outer_dim,
    const T *x,
    T *y)
{
  extern __shared__ T cache[];
  for (uint32_t blk = gblock(); blk < mid_dim; blk += gblockcount()) {
    T accumulator = Reduce::InitVal();
    for (uint32_t j = 0; j < reduce_outer_dim; ++j) {
      uint32_t rdup_reduce_inner_dim = (reduce_inner_dim + blockDim.x - 1) / blockDim.x * blockDim.x;
      for (uint32_t i = threadIdx.x; i < rdup_reduce_inner_dim; i += blockDim.x) {
        if (i < reduce_inner_dim) {
          cache[threadIdx.x] = Map::Map(x[Index3::Pack(i, reduce_inner_dim, blk, mid_dim, j)]);
        } else {
          cache[threadIdx.x] = Reduce::InitVal();
        }
        __syncthreads();
        threadblock_reduce_sync<T, Reduce>(cache);
        if (0 == threadIdx.x) {
          Reduce::Reduce(&accumulator, cache[0]);
        }
        __syncthreads();
      }
    }
    y[blk] = accumulator;
  }
}

extern "C" void anode_gpu_sum_Iabc_Ob_packed_deterministic_f32(
    uint32_t reduce_inner_dim,
    uint32_t mid_dim,
    uint32_t reduce_outer_dim,
    const float *x,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  assert(check_power_of_2(cfg->flat_block_dim().x));
  anode_gpu_map_reduce_Iabc_Ob_packed_deterministic_kernel<float, CopyMap<float>, AddReduce<float>><<<cfg->flat_block_count(mid_dim), cfg->flat_block_dim(), cfg->flat_block_len() * sizeof(float), stream>>>(
      reduce_inner_dim, mid_dim, reduce_outer_dim, x, y);
}

extern "C" void anode_gpu_square_map_sum_Iabc_Ob_packed_deterministic_f32(
    uint32_t reduce_inner_dim,
    uint32_t mid_dim,
    uint32_t reduce_outer_dim,
    const float *x,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  assert(check_power_of_2(cfg->flat_block_dim().x));
  anode_gpu_map_reduce_Iabc_Ob_packed_deterministic_kernel<float, SquareMap<float>, AddReduce<float>><<<cfg->flat_block_count(mid_dim), cfg->flat_block_dim(), cfg->flat_block_len() * sizeof(float), stream>>>(
      reduce_inner_dim, mid_dim, reduce_outer_dim, x, y);
}
