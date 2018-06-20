/*
Copyright 2017-2018 Peter Jin

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
#include <math_constants.h>

__global__ void anode_gpu_batch_norm_3d1_packed_kernel_f32(
    uint32_t len,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float epsilon,
    const float *x,
    const float *mean,
    const float *var,
    float *y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    uint32_t i0, i1, i2;
    Index3::Unpack(
        idx,
        &i0, dim0,
        &i1, dim1,
        &i2
    );
    if (i0 < dim0 && i1 < dim1 && i2 < dim2) {
      y[idx] = (x[idx] - mean[i1]) * rsqrtf(var[i1] + epsilon);
    }
  }
}

__global__ void anode_gpu_batch_norm_bwd_3d1_packed_kernel_f32(
    uint32_t len,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float epsilon,
    const float *dy,
    const float *var,
    float *dx)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    uint32_t i0, i1, i2;
    Index3::Unpack(
        idx,
        &i0, dim0,
        &i1, dim1,
        &i2
    );
    if (i0 < dim0 && i1 < dim1 && i2 < dim2) {
      dx[idx] = dy[idx] * rsqrtf(var[i1] + epsilon);
    }
  }
}

__global__ void anode_gpu_batch_norm_bwd_mean_3d1_packed_deterministic_kernel_f32(
    uint32_t len,
    uint32_t reduce_inner_dim,
    uint32_t mid_dim,
    uint32_t reduce_outer_dim,
    float epsilon,
    const float *dy,
    const float *var,
    float *dmean)
{
  extern __shared__ float cache[];
  uint32_t fused_inner_outer_dim = reduce_inner_dim * reduce_outer_dim;
  uint32_t rdup_fused_inner_outer_dim = (fused_inner_outer_dim + blockDim.x - 1) / blockDim.x * blockDim.x;
  for (uint32_t blk1 = gblock(); blk1 < mid_dim; blk1 += gblockcount()) {
    float accumulator = 0.0f;
    for (uint32_t i = threadIdx.x; i < rdup_fused_inner_outer_dim; i += blockDim.x) {
      if (i < fused_inner_outer_dim) {
        uint32_t i0, i2;
        Index2::Unpack(i, &i0, reduce_inner_dim, &i2);
        uint32_t idx = Index3::Pack(i0, reduce_inner_dim, blk1, mid_dim, i2);
        cache[threadIdx.x] = -dy[idx] * rsqrtf(var[blk1] + epsilon);
      } else {
        cache[threadIdx.x] = 0.0f;
      }
      __syncthreads();
      threadblock_reduce_sync<float, AddReduce<float>>(cache);
      if (0 == threadIdx.x) {
        accumulator += cache[0];
      }
      __syncthreads();
    }
    dmean[blk1] = accumulator;
  }
}

__global__ void anode_gpu_batch_norm_bwd_var_3d1_packed_deterministic_kernel_f32(
    uint32_t len,
    uint32_t reduce_inner_dim,
    uint32_t mid_dim,
    uint32_t reduce_outer_dim,
    float epsilon,
    const float *y,
    const float *dy,
    const float *var,
    float *dvar)
{
  extern __shared__ float cache[];
  uint32_t fused_inner_outer_dim = reduce_inner_dim * reduce_outer_dim;
  uint32_t rdup_fused_inner_outer_dim = (fused_inner_outer_dim + blockDim.x - 1) / blockDim.x * blockDim.x;
  for (uint32_t blk1 = gblock(); blk1 < mid_dim; blk1 += gblockcount()) {
    float accumulator = 0.0f;
    for (uint32_t i = threadIdx.x; i < rdup_fused_inner_outer_dim; i += blockDim.x) {
      if (i < fused_inner_outer_dim) {
        uint32_t i0, i2;
        Index2::Unpack(i, &i0, reduce_inner_dim, &i2);
        uint32_t idx = Index3::Pack(i0, reduce_inner_dim, blk1, mid_dim, i2);
        cache[threadIdx.x] = -0.5f * y[idx] * dy[idx] / (var[blk1] + epsilon);
      } else {
        cache[threadIdx.x] = 0.0f;
      }
      __syncthreads();
      threadblock_reduce_sync<float, AddReduce<float>>(cache);
      if (0 == threadIdx.x) {
        accumulator += cache[0];
      }
      __syncthreads();
    }
    dvar[blk1] = accumulator;
  }
}
