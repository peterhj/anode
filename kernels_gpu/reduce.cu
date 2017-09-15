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
#include <cuda_runtime.h>

template <typename T, typename Reduce>
__global__ void anode_gpu_reduce_I1ab_Ob_packed_deterministic_kernel(
    uint32_t inner_dim,
    uint32_t keep_dim,
    const T *x,
    T *y)
{
  __shared__ T cache[512];
  for (uint32_t blk = gblock(); blk < keep_dim; blk += gblockcount()) {
    T accumulator = Reduce::InitVal();
    uint32_t rdup_inner_dim = (inner_dim + blockDim.x - 1) / blockDim.x * blockDim.x;
    uint32_t tid = threadIdx.x;
    uint32_t base_tid = 0;
    for ( ; tid < rdup_inner_dim; tid += blockDim.x, base_tid += blockDim.x) {
      uint32_t i = tid - base_tid;
      if (tid < inner_dim) {
        cache[i] = x[Index2::Pack(tid, inner_dim, blk, keep_dim)];
      } else {
        cache[i] = Reduce::InitVal();
      }
      __syncthreads();
      threadblock_reduce1024<T, Reduce>(cache);
      if (0 == i) {
        Reduce::Reduce(&accumulator, cache[0]);
      }
      __syncthreads();
    }
    y[blk] = accumulator;
  }
}

extern "C" void anode_gpu_sum_reduce_I1ab_Ob_packed_deterministic_f32(
    uint32_t inner_dim,
    uint32_t keep_dim,
    const float *x,
    float *y,
    KernelConfig cfg,
    cudaStream_t stream)
{
  (void)cfg;
  anode_gpu_reduce_I1ab_Ob_packed_deterministic_kernel<float, AddReduce<float>><<<keep_dim, 512, 0, stream>>>(
      inner_dim, keep_dim, x, y);
}

template <typename T, typename AtomicReduce>
__global__ void anode_gpu_reduce_I1ab_Ob_packed_atomic_accumulate_kernel(
    uint32_t inner_dim,
    uint32_t keep_dim,
    const T *x,
    T *y)
{
  __shared__ T cache[512];
  for (uint32_t blk = gblock(); blk < keep_dim; blk += gblockcount()) {
    T accumulator = AtomicReduce::InitVal();
    uint32_t rdup_inner_dim = (inner_dim + blockDim.x - 1) / blockDim.x * blockDim.x;
    uint32_t tid = threadIdx.x;
    uint32_t base_tid = 0;
    for ( ; tid < rdup_inner_dim; tid += blockDim.x, base_tid += blockDim.x) {
      uint32_t i = tid - base_tid;
      if (tid < inner_dim) {
        cache[i] = x[Index2::Pack(tid, inner_dim, blk, keep_dim)];
      } else {
        cache[i] = AtomicReduce::InitVal();
      }
      __syncthreads();
      threadblock_reduce1024<T, AtomicReduce>(cache);
      if (0 == i) {
        AtomicReduce::AtomicReduce(&accumulator, cache[0]);
      }
      __syncthreads();
    }
    AtomicReduce::AtomicReduce(&y[blk], accumulator);
  }
}
