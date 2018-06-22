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
#include <cassert>
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void anode_gpu_batch_mean_3d1_packed_deterministic_kernel_f32(
    uint32_t reduce_inner_dim,
    uint32_t mid_dim,
    uint32_t reduce_outer_dim,
    const float *x,
    float *mean)
{
  extern __shared__ float cache[];
  uint32_t fused_inner_outer_dim = reduce_inner_dim * reduce_outer_dim;
  uint32_t rdup_fused_inner_outer_dim = (fused_inner_outer_dim + blockDim.x - 1) / blockDim.x * blockDim.x;
  float norm1 = ((float)reduce_inner_dim) * ((float)reduce_outer_dim);
  for (uint32_t blk1 = gblock(); blk1 < mid_dim; blk1 += gblockcount()) {
    float accumulator = 0.0f;
    for (uint32_t i = threadIdx.x; i < rdup_fused_inner_outer_dim; i += blockDim.x) {
      if (i < fused_inner_outer_dim) {
        uint32_t i0, i2;
        Index2::Unpack(i, &i0, reduce_inner_dim, &i2);
        uint32_t idx = Index3::Pack(i0, reduce_inner_dim, blk1, mid_dim, i2);
        cache[threadIdx.x] = x[idx];
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
    mean[blk1] = accumulator / norm1;
  }
}

extern "C" void anode_gpu_batch_mean_3d1_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *x,
    float *mean,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  assert(check_power_of_2(cfg->flat_block_dim().x));
  anode_gpu_batch_mean_3d1_packed_deterministic_kernel_f32<<<cfg->flat_block_count(dim1), cfg->flat_block_dim(), cfg->flat_block_len() * sizeof(float), stream>>>(
      dim0, dim1, dim2, x, mean);
}

template <typename Write>
__global__ void anode_gpu_batch_mean_bwd_3d1_packed_kernel_f32(
    uint32_t len,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dmean,
    float *dx)
{
  float norm1 = ((float)dim0) * ((float)dim2);
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    uint32_t i0, i1, i2;
    Index3::Unpack(
        idx,
        &i0, dim0,
        &i1, dim1,
        &i2
    );
    if (i0 < dim0 && i1 < dim1 && i2 < dim2) {
      Write::Write(&dx[idx], dmean[i1] / norm1);
    }
  }
}

extern "C" void anode_gpu_batch_mean_bwd_3d1_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dmean,
    float *dx,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t len = dim0 * dim1 * dim2;
  anode_gpu_batch_mean_bwd_3d1_packed_kernel_f32<AssignWrite<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, dim0, dim1, dim2, dmean, dx);
}

extern "C" void anode_gpu_batch_mean_bwd_3d1_packed_accumulate_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dmean,
    float *dx,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t len = dim0 * dim1 * dim2;
  anode_gpu_batch_mean_bwd_3d1_packed_kernel_f32<AccumulateWrite<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, dim0, dim1, dim2, dmean, dx);
}

__global__ void anode_gpu_batch_var_3d1_packed_deterministic_kernel_f32(
    uint32_t reduce_inner_dim,
    uint32_t mid_dim,
    uint32_t reduce_outer_dim,
    float epsilon,
    const float *x,
    const float *mean,
    float *var)
{
  extern __shared__ float cache[];
  uint32_t fused_inner_outer_dim = reduce_inner_dim * reduce_outer_dim;
  uint32_t rdup_fused_inner_outer_dim = (fused_inner_outer_dim + blockDim.x - 1) / blockDim.x * blockDim.x;
  float norm2 = ((float)reduce_inner_dim - 1.0f) * ((float)reduce_outer_dim - 1.0f);
  for (uint32_t blk1 = gblock(); blk1 < mid_dim; blk1 += gblockcount()) {
    float accumulator = 0.0f;
    float m = mean[blk1];
    for (uint32_t i = threadIdx.x; i < rdup_fused_inner_outer_dim; i += blockDim.x) {
      if (i < fused_inner_outer_dim) {
        uint32_t i0, i2;
        Index2::Unpack(i, &i0, reduce_inner_dim, &i2);
        uint32_t idx = Index3::Pack(i0, reduce_inner_dim, blk1, mid_dim, i2);
        float x_i = x[idx];
        cache[threadIdx.x] = (x_i - m) * (x_i - m);
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
    var[blk1] = accumulator / norm2 + epsilon;
  }
}

extern "C" void anode_gpu_batch_var_3d1_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float epsilon,
    const float *x,
    const float *mean,
    float *var,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  assert(check_power_of_2(cfg->flat_block_dim().x));
  anode_gpu_batch_var_3d1_packed_deterministic_kernel_f32<<<cfg->flat_block_count(dim1), cfg->flat_block_dim(), cfg->flat_block_len() * sizeof(float), stream>>>(
      dim0, dim1, dim2, epsilon, x, mean, var);
}

template <typename Write>
__global__ void anode_gpu_batch_var_bwd_3d1_packed_kernel_f32(
    uint32_t len,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dvar,
    const float *x,
    const float *mean,
    float *dx)
{
  float norm2 = ((float)dim0 - 1.0f) * ((float)dim2 - 1.0f);
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    uint32_t i0, i1, i2;
    Index3::Unpack(
        idx,
        &i0, dim0,
        &i1, dim1,
        &i2
    );
    if (i0 < dim0 && i1 < dim1 && i2 < dim2) {
      Write::Write(&dx[idx], dvar[i1] * (x[idx] - mean[i1]) * 2.0f / norm2);
    }
  }
}

extern "C" void anode_gpu_batch_var_bwd_3d1_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dvar,
    const float *x,
    const float *mean,
    float *dx,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t len = dim0 * dim1 * dim2;
  anode_gpu_batch_var_bwd_3d1_packed_kernel_f32<AssignWrite<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, dim0, dim1, dim2, dvar, x, mean, dx);
}

extern "C" void anode_gpu_batch_var_bwd_3d1_packed_accumulate_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dvar,
    const float *x,
    const float *mean,
    float *dx,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t len = dim0 * dim1 * dim2;
  anode_gpu_batch_var_bwd_3d1_packed_kernel_f32<AccumulateWrite<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, dim0, dim1, dim2, dvar, x, mean, dx);
}

template <typename Write>
__global__ void anode_gpu_batch_var_bwd_mean_3d1_packed_deterministic_kernel_f32(
    uint32_t reduce_inner_dim,
    uint32_t mid_dim,
    uint32_t reduce_outer_dim,
    const float *dvar,
    const float *x,
    const float *mean,
    float *dmean)
{
  extern __shared__ float cache[];
  uint32_t fused_inner_outer_dim = reduce_inner_dim * reduce_outer_dim;
  uint32_t rdup_fused_inner_outer_dim = (fused_inner_outer_dim + blockDim.x - 1) / blockDim.x * blockDim.x;
  float norm2 = ((float)reduce_inner_dim - 1.0f) * ((float)reduce_outer_dim - 1.0f);
  for (uint32_t blk1 = gblock(); blk1 < mid_dim; blk1 += gblockcount()) {
    float accumulator = 0.0f;
    float m = mean[blk1];
    float dv = dvar[blk1];
    for (uint32_t i = threadIdx.x; i < rdup_fused_inner_outer_dim; i += blockDim.x) {
      if (i < fused_inner_outer_dim) {
        uint32_t i0, i2;
        Index2::Unpack(i, &i0, reduce_inner_dim, &i2);
        uint32_t idx = Index3::Pack(i0, reduce_inner_dim, blk1, mid_dim, i2);
        cache[threadIdx.x] = -dv * (x[idx] - m) * 2.0f;
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
    Write::Write(&dmean[blk1], accumulator / norm2);
  }
}

extern "C" void anode_gpu_batch_var_bwd_mean_3d1_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dvar,
    const float *x,
    const float *mean,
    float *dmean,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  assert(check_power_of_2(cfg->flat_block_dim().x));
  anode_gpu_batch_var_bwd_mean_3d1_packed_deterministic_kernel_f32<AssignWrite<float>><<<cfg->flat_block_count(dim1), cfg->flat_block_dim(), cfg->flat_block_len() * sizeof(float), stream>>>(
      dim0, dim1, dim2, dvar, x, mean, dmean);
}

extern "C" void anode_gpu_batch_var_bwd_mean_3d1_packed_accumulate_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dvar,
    const float *x,
    const float *mean,
    float *dmean,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  assert(check_power_of_2(cfg->flat_block_dim().x));
  anode_gpu_batch_var_bwd_mean_3d1_packed_deterministic_kernel_f32<AccumulateWrite<float>><<<cfg->flat_block_count(dim1), cfg->flat_block_dim(), cfg->flat_block_len() * sizeof(float), stream>>>(
      dim0, dim1, dim2, dvar, x, mean, dmean);
}

__global__ void anode_gpu_batch_norm_3d1_packed_kernel_f32(
    uint32_t len,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
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
      y[idx] = (x[idx] - mean[i1]) * rsqrtf(var[i1]);
    }
  }
}

extern "C" void anode_gpu_batch_norm_3d1_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *x,
    const float *mean,
    const float *var,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t len = dim0 * dim1 * dim2;
  anode_gpu_batch_norm_3d1_packed_kernel_f32<<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, dim0, dim1, dim2, x, mean, var, y);
}

template <typename Write>
__global__ void anode_gpu_batch_norm_bwd_3d1_packed_kernel_f32(
    uint32_t len,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
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
      Write::Write(&dx[idx], dy[idx] * rsqrtf(var[i1]));
    }
  }
}

extern "C" void anode_gpu_batch_norm_bwd_3d1_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dy,
    const float *var,
    float *dx,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t len = dim0 * dim1 * dim2;
  anode_gpu_batch_norm_bwd_3d1_packed_kernel_f32<AssignWrite<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, dim0, dim1, dim2, dy, var, dx);
}

extern "C" void anode_gpu_batch_norm_bwd_3d1_packed_accumulate_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dy,
    const float *var,
    float *dx,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t len = dim0 * dim1 * dim2;
  anode_gpu_batch_norm_bwd_3d1_packed_kernel_f32<AccumulateWrite<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, dim0, dim1, dim2, dy, var, dx);
}

template <typename Write>
__global__ void anode_gpu_batch_norm_bwd_mean_3d1_packed_deterministic_kernel_f32(
    uint32_t reduce_inner_dim,
    uint32_t mid_dim,
    uint32_t reduce_outer_dim,
    const float *dy,
    const float *var,
    float *dmean)
{
  extern __shared__ float cache[];
  uint32_t fused_inner_outer_dim = reduce_inner_dim * reduce_outer_dim;
  uint32_t rdup_fused_inner_outer_dim = (fused_inner_outer_dim + blockDim.x - 1) / blockDim.x * blockDim.x;
  for (uint32_t blk1 = gblock(); blk1 < mid_dim; blk1 += gblockcount()) {
    float accumulator = 0.0f;
    float v = var[blk1];
    for (uint32_t i = threadIdx.x; i < rdup_fused_inner_outer_dim; i += blockDim.x) {
      if (i < fused_inner_outer_dim) {
        uint32_t i0, i2;
        Index2::Unpack(i, &i0, reduce_inner_dim, &i2);
        uint32_t idx = Index3::Pack(i0, reduce_inner_dim, blk1, mid_dim, i2);
        cache[threadIdx.x] = -dy[idx] * rsqrtf(v);
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
    Write::Write(&dmean[blk1], accumulator);
  }
}

extern "C" void anode_gpu_batch_norm_bwd_mean_3d1_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dy,
    const float *var,
    float *dmean,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  assert(check_power_of_2(cfg->flat_block_dim().x));
  anode_gpu_batch_norm_bwd_mean_3d1_packed_deterministic_kernel_f32<AssignWrite<float>><<<cfg->flat_block_count(dim1), cfg->flat_block_dim(), cfg->flat_block_len() * sizeof(float), stream>>>(
      dim0, dim1, dim2, dy, var, dmean);
}

extern "C" void anode_gpu_batch_norm_bwd_mean_3d1_packed_accumulate_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dy,
    const float *var,
    float *dmean,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  assert(check_power_of_2(cfg->flat_block_dim().x));
  anode_gpu_batch_norm_bwd_mean_3d1_packed_deterministic_kernel_f32<AccumulateWrite<float>><<<cfg->flat_block_count(dim1), cfg->flat_block_dim(), cfg->flat_block_len() * sizeof(float), stream>>>(
      dim0, dim1, dim2, dy, var, dmean);
}

template <typename Write>
__global__ void anode_gpu_batch_norm_bwd_var_3d1_packed_deterministic_kernel_f32(
    uint32_t reduce_inner_dim,
    uint32_t mid_dim,
    uint32_t reduce_outer_dim,
    const float *dy,
    const float *x,
    const float *mean,
    const float *var,
    float *dvar)
{
  extern __shared__ float cache[];
  uint32_t fused_inner_outer_dim = reduce_inner_dim * reduce_outer_dim;
  uint32_t rdup_fused_inner_outer_dim = (fused_inner_outer_dim + blockDim.x - 1) / blockDim.x * blockDim.x;
  for (uint32_t blk1 = gblock(); blk1 < mid_dim; blk1 += gblockcount()) {
    float accumulator = 0.0f;
    float m = mean[blk1];
    float v = var[blk1];
    for (uint32_t i = threadIdx.x; i < rdup_fused_inner_outer_dim; i += blockDim.x) {
      if (i < fused_inner_outer_dim) {
        uint32_t i0, i2;
        Index2::Unpack(i, &i0, reduce_inner_dim, &i2);
        uint32_t idx = Index3::Pack(i0, reduce_inner_dim, blk1, mid_dim, i2);
        cache[threadIdx.x] = -dy[idx] * (x[idx] - m) * 0.5f * rsqrtf(v) / v;
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
    Write::Write(&dvar[blk1], accumulator);
  }
}

extern "C" void anode_gpu_batch_norm_bwd_var_3d1_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dy,
    const float *x,
    const float *mean,
    const float *var,
    float *dvar,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  assert(check_power_of_2(cfg->flat_block_dim().x));
  anode_gpu_batch_norm_bwd_var_3d1_packed_deterministic_kernel_f32<AssignWrite<float>><<<cfg->flat_block_count(dim1), cfg->flat_block_dim(), cfg->flat_block_len() * sizeof(float), stream>>>(
      dim0, dim1, dim2, dy, x, mean, var, dvar);
}

extern "C" void anode_gpu_batch_norm_bwd_var_3d1_packed_accumulate_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dy,
    const float *x,
    const float *mean,
    const float *var,
    float *dvar,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  assert(check_power_of_2(cfg->flat_block_dim().x));
  anode_gpu_batch_norm_bwd_var_3d1_packed_deterministic_kernel_f32<AccumulateWrite<float>><<<cfg->flat_block_count(dim1), cfg->flat_block_dim(), cfg->flat_block_len() * sizeof(float), stream>>>(
      dim0, dim1, dim2, dy, x, mean, var, dvar);
}

template <typename Write>
__global__ void anode_gpu_batch_norm_bwd_var_v2_3d1_packed_deterministic_kernel_f32(
    uint32_t reduce_inner_dim,
    uint32_t mid_dim,
    uint32_t reduce_outer_dim,
    const float *dy,
    const float *y,
    const float *var,
    float *dvar)
{
  extern __shared__ float cache[];
  uint32_t fused_inner_outer_dim = reduce_inner_dim * reduce_outer_dim;
  uint32_t rdup_fused_inner_outer_dim = (fused_inner_outer_dim + blockDim.x - 1) / blockDim.x * blockDim.x;
  for (uint32_t blk1 = gblock(); blk1 < mid_dim; blk1 += gblockcount()) {
    float accumulator = 0.0f;
    float v = var[blk1];
    for (uint32_t i = threadIdx.x; i < rdup_fused_inner_outer_dim; i += blockDim.x) {
      if (i < fused_inner_outer_dim) {
        uint32_t i0, i2;
        Index2::Unpack(i, &i0, reduce_inner_dim, &i2);
        uint32_t idx = Index3::Pack(i0, reduce_inner_dim, blk1, mid_dim, i2);
        cache[threadIdx.x] = -dy[idx] * y[idx] * 0.5f / v;
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
    Write::Write(&dvar[blk1], accumulator);
  }
}

extern "C" void anode_gpu_batch_norm_bwd_var_v2_3d1_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dy,
    const float *y,
    const float *var,
    float *dvar,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  assert(check_power_of_2(cfg->flat_block_dim().x));
  anode_gpu_batch_norm_bwd_var_v2_3d1_packed_deterministic_kernel_f32<AssignWrite<float>><<<cfg->flat_block_count(dim1), cfg->flat_block_dim(), cfg->flat_block_len() * sizeof(float), stream>>>(
      dim0, dim1, dim2, dy, y, var, dvar);
}

extern "C" void anode_gpu_batch_norm_bwd_var_v2_3d1_packed_accumulate_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dy,
    const float *y,
    const float *var,
    float *dvar,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  assert(check_power_of_2(cfg->flat_block_dim().x));
  anode_gpu_batch_norm_bwd_var_v2_3d1_packed_deterministic_kernel_f32<AccumulateWrite<float>><<<cfg->flat_block_count(dim1), cfg->flat_block_dim(), cfg->flat_block_len() * sizeof(float), stream>>>(
      dim0, dim1, dim2, dy, y, var, dvar);
}
