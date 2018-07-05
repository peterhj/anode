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
#include <math_constants.h>

__global__ void anode_gpu_batch_norm_external_3d1_packed_kernel_f32(
    uint32_t flat_len,
    uint32_t rdup_len_per_blk,
    uint32_t num_blks_per_group,
    uint32_t num_groups,
    uint32_t reduce_inner_dim,
    uint32_t mid_dim,
    uint32_t reduce_outer_dim,
    float epsilon,
    const float *x,
    float *y,
    float *mean,
    float *var,
    float *workspace)
{
  extern __shared__ float cache[];
  float *x_cache = cache;
  float *reduce_cache = cache + rdup_len_per_blk;

  uint32_t fused_inner_outer_dim = reduce_inner_dim * reduce_outer_dim;
  //uint32_t rdup_fused_inner_outer_dim = (fused_inner_outer_dim + blockDim.x - 1) / blockDim.x * blockDim.x;

  //uint32_t iter_idx = gtindex();
  uint32_t group_nr = blockIdx.x / num_blks_per_group;
  uint32_t group_blk_nr = blockIdx.x % num_blks_per_group;
  uint32_t group_offset = rdup_len_per_blk * group_blk_nr;

  float mean_accumulator = 0.0f;

  for (uint32_t blk_idx = threadIdx.x; blk_idx < rdup_len_per_blk; blk_idx += blockDim.x) {
    uint32_t group_idx = group_offset + blk_idx;

    uint32_t i0, i2;
    Index2::Unpack(group_idx,
        &i0, reduce_inner_dim,
        &i2);
    uint32_t flat_idx = Index3::Pack(
        i0, reduce_inner_dim,
        group_nr, mid_dim,
        i2);

    if (flat_idx < flat_len) {
      float x_i = x[flat_idx];
      x_cache[blk_idx] = x_i;
      reduce_cache[threadIdx.x] = x_i;
    } else {
      reduce_cache[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    threadblock_reduce_sync<float, AddReduce<float>>(reduce_cache);
    if (0 == threadIdx.x) {
      mean_accumulator += reduce_cache[0];
    }
    __syncthreads();

    // TODO: reduce b/w blocks.
  }

  // TODO: reduce b/w blocks.
  // FIXME: this is the slow reduction.
  if (0 == threadIdx.x) {
    workspace[blockIdx.x] = mean_accumulator;
  }
  // FIXME: memory fence, or sync b/w all blocks in group.
  if (0 == threadIdx.x && 0 == group_blk_nr) {
    mean_accumulator = 0.0f;
    for (uint32_t b = 0; b < num_blks_per_group; ++b) {
      mean_accumulator += workspace[blockIdx.x + b];
    }
    workspace[blockIdx.x] = mean_accumulator;
  }
  // FIXME: memory fence, or sync b/w all blocks in group.
  mean_accumulator = workspace[group_blk_nr];

  float m = mean_accumulator / (((float)reduce_inner_dim) * ((float)reduce_outer_dim));
  float var_accumulator = 0.0f;

  for (uint32_t blk_idx = threadIdx.x; blk_idx < rdup_len_per_blk; blk_idx += blockDim.x) {
    uint32_t group_idx = group_offset + blk_idx;

    uint32_t i0, i2;
    Index2::Unpack(group_idx,
        &i0, reduce_inner_dim,
        &i2);
    uint32_t flat_idx = Index3::Pack(
        i0, reduce_inner_dim,
        group_nr, mid_dim,
        i2);

    if (flat_idx < flat_len) {
      float x_i = x_cache[blk_idx];
      reduce_cache[threadIdx.x] = (x_i - m) * (x_i - m);
    } else {
      reduce_cache[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    threadblock_reduce_sync<float, AddReduce<float>>(reduce_cache);
    if (0 == threadIdx.x) {
      var_accumulator += reduce_cache[0];
    }
    __syncthreads();

    // TODO: reduce b/w blocks.
  }

  float v = var_accumulator / (((float)reduce_inner_dim - 1.0f) * ((float)reduce_outer_dim - 1.0f)) + epsilon;
  float v_term = rsqrtf(v);

  for (uint32_t blk_idx = threadIdx.x; blk_idx < rdup_len_per_blk; blk_idx += blockDim.x) {
    uint32_t group_idx = group_offset + blk_idx;

    uint32_t i0, i2;
    Index2::Unpack(group_idx,
        &i0, reduce_inner_dim,
        &i2);
    uint32_t flat_idx = Index3::Pack(
        i0, reduce_inner_dim,
        group_nr, mid_dim,
        i2);

    if (flat_idx < flat_len) {
      y[flat_idx] = (x_cache[blk_idx] - m) * v_term;
    }
  }

  if (0 == threadIdx.x && 0 == group_blk_nr) {
    mean[group_nr] = m;
    var[group_nr] = v;
  }
}
