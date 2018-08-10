/*
Copyright 2018 Peter Jin

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

__global__ void anode_gpu_smooth_negative_f1_loss_3d1_packed_f32_kernel(
    uint32_t dim0,
    uint32_t cat_dim1,
    uint32_t dim2,
    float epsilon,
    const float *x,
    const float *target,
    float *y)
{
  extern __shared__ float cache[];
  float *numer_cache = cache;
  float *denom_cache = cache + blockDim.x;
  uint32_t fused_dim = cat_dim1 * dim2;
  uint32_t rdup_dim0 = (dim0 + blockDim.x - 1) / blockDim.x * blockDim.x;
  for (uint32_t blk = gblock(); blk < fused_dim; blk += gblockcount()) {
    uint32_t blk_i1, blk_i2;
    Index2::Unpack(blk,
        &blk_i1, cat_dim1,
        &blk_i2);
    float numer_accumulator = 0.0f;
    float denom_accumulator = 0.0f;
    for (uint32_t i0 = threadIdx.x; i0 < rdup_dim0; i0 += blockDim.x) {
      if (i0 < dim0) {
        uint32_t idx = Index3::Pack(
            i0, dim0,
            blk_i1, cat_dim1,
            blk_i2);
        float x_i = x[idx];
        float tg_i = target[idx];
        numer_cache[threadIdx.x] = x_i * tg_i;
        denom_cache[threadIdx.x] = x_i + tg_i;
      } else {
        numer_cache[threadIdx.x] = 0.0f;
        denom_cache[threadIdx.x] = 0.0f;
      }
      __syncthreads();
      threadblock_reduce_sync<float, AddReduce<float>>(numer_cache);
      threadblock_reduce_sync<float, AddReduce<float>>(denom_cache);
      if (0 == threadIdx.x) {
        numer_accumulator += numer_cache[0];
        denom_accumulator += denom_cache[0];
      }
      __syncthreads();
    }
    if (0 == threadIdx.x) {
      y[blk] = -(2.0f * numer_accumulator + epsilon) / (denom_accumulator + epsilon);
    }
  }
}

extern "C" void anode_gpu_smooth_negative_f1_loss_3d1_packed_f32(
    uint32_t dim0,
    uint32_t cat_dim1,
    uint32_t dim2,
    float epsilon,
    const float *x,
    const float *target,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  assert(check_power_of_2(cfg->flat_block_dim().x));
  anode_gpu_smooth_negative_f1_loss_3d1_packed_f32_kernel<<<cfg->flat_block_count(cat_dim1 * dim2), cfg->flat_block_dim(), cfg->flat_block_len() * sizeof(float) * 2, stream>>>(
      dim0, cat_dim1, dim2, epsilon, x, target, y);
}

__global__ void anode_gpu_smooth_negative_f1_loss_3d1_bwd_packed_f32_kernel(
    uint32_t flat_len,
    uint32_t dim0,
    uint32_t cat_dim1,
    uint32_t dim2,
    float epsilon,
    const float *dy,
    const float *x,
    const float *target,
    float *dx)
{
  for (uint32_t idx = gtindex(); idx < flat_len; idx += gtcount()) {
    uint32_t i0, i1, i2;
    Index3::Unpack(idx,
        &i0, dim0,
        &i1, cat_dim1,
        &i2);
    uint32_t dy_idx = Index2::Pack(
        i1, cat_dim1,
        i2);
    float x_i = x[idx];
    float tg_i = target[idx];
    float numer_i = 2.0f * tg_i * (tg_i + epsilon) - epsilon;
    float denom_i = x_i + tg_i + epsilon;
    dx[idx] = dy[dy_idx] * -numer_i / (denom_i * denom_i);
  }
}

extern "C" void anode_gpu_smooth_negative_f1_loss_3d1_bwd_packed_f32(
    uint32_t dim0,
    uint32_t cat_dim1,
    uint32_t dim2,
    float epsilon,
    const float *dy,
    const float *x,
    const float *target,
    float *dx,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t len = dim0 * cat_dim1 * dim2;
  anode_gpu_smooth_negative_f1_loss_3d1_bwd_packed_f32_kernel<<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, dim0, cat_dim1, dim2, epsilon, dy, x, target, dx);
}
