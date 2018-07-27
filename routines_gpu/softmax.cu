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
//#include <cassert>
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void anode_gpu_softmax_packed_block_f32_kernel(
    uint32_t reduce_inner_dim,
    uint32_t outer_dim,
    const float *x,
    float *y)
{
  extern __shared__ float cache[];
  for (uint32_t blk1 = gblock(); blk1 < outer_dim; blk1 += gblockcount()) {
    uint32_t idx = Index2::Pack(threadIdx.x, reduce_inner_dim, blk1);

    float x_i;
    if (threadIdx.x < reduce_inner_dim) {
      x_i = x[idx];
      cache[threadIdx.x] = x_i;
    } else {
      cache[threadIdx.x] = MaxReduce<float>::InitVal();
    }
    __syncthreads();
    threadblock_reduce_sync<float, MaxReduce<float>>(cache);

    float x_max_accumulator = cache[0];
    __syncthreads();

    float z_i;
    if (threadIdx.x < reduce_inner_dim) {
      z_i = expf(x_i - x_max_accumulator);
      cache[threadIdx.x] = z_i;
    } else {
      cache[threadIdx.x] = AddReduce<float>::InitVal();
    }
    __syncthreads();
    threadblock_reduce_sync<float, AddReduce<float>>(cache);

    float z_sum_accumulator = cache[0];
    __syncthreads();

    if (threadIdx.x < reduce_inner_dim) {
      float y_i = z_i / z_sum_accumulator;
      y[idx] = y_i;
    }
    __syncthreads();
  }
}

extern "C" void anode_gpu_softmax_packed_block_f32(
    uint32_t dim0,
    uint32_t dim1,
    const float *x,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  anode_gpu_softmax_packed_block_f32_kernel<<<cfg->flat_block_count(dim1), cfg->flat_block_dim(), cfg->flat_block_len() * sizeof(float), stream>>>(
      dim0, dim1, x, y);
}

__global__ void anode_gpu_softmax_packed_deterministic_f32_kernel(
    uint32_t reduce_inner_dim,
    uint32_t outer_dim,
    const float *x,
    float *y)
{
  extern __shared__ float cache[];
  uint32_t rdup_reduce_inner_dim = (reduce_inner_dim + blockDim.x - 1) / blockDim.x * blockDim.x;
  for (uint32_t blk1 = gblock(); blk1 < outer_dim; blk1 += gblockcount()) {
    float x_max_accumulator = MaxReduce<float>::InitVal();
    for (uint32_t i0 = threadIdx.x; i0 < rdup_reduce_inner_dim; i0 += blockDim.x) {
      if (i0 < reduce_inner_dim) {
        uint32_t idx = Index2::Pack(i0, reduce_inner_dim, blk1);
        float x_i = x[idx];
        cache[threadIdx.x] = x_i;
      } else {
        cache[threadIdx.x] = MaxReduce<float>::InitVal();
      }
      __syncthreads();
      threadblock_reduce_sync<float, MaxReduce<float>>(cache);
      MaxReduce<float>::Reduce(&x_max_accumulator, cache[0]);
      __syncthreads();
    }

    float z_sum_accumulator = AddReduce<float>::InitVal();
    for (uint32_t i0 = threadIdx.x; i0 < rdup_reduce_inner_dim; i0 += blockDim.x) {
      if (i0 < reduce_inner_dim) {
        uint32_t idx = Index2::Pack(i0, reduce_inner_dim, blk1);
        float x_i = x[idx];
        float z_i = expf(x_i - x_max_accumulator);
        cache[threadIdx.x] = z_i;
      } else {
        cache[threadIdx.x] = AddReduce<float>::InitVal();
      }
      __syncthreads();
      threadblock_reduce_sync<float, AddReduce<float>>(cache);
      AddReduce<float>::Reduce(&z_sum_accumulator, cache[0]);
      __syncthreads();
    }

    for (uint32_t i0 = threadIdx.x; i0 < rdup_reduce_inner_dim; i0 += blockDim.x) {
      if (i0 < reduce_inner_dim) {
        uint32_t idx = Index2::Pack(i0, reduce_inner_dim, blk1);
        float x_i = x[idx];
        float z_i = expf(x_i - x_max_accumulator);
        float y_i = z_i / z_sum_accumulator;
        y[idx] = y_i;
      }
    }
  }
}

extern "C" void anode_gpu_softmax_packed_deterministic_f32(
    uint32_t dim0,
    uint32_t dim1,
    const float *x,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  anode_gpu_softmax_packed_deterministic_f32_kernel<<<cfg->flat_block_count(dim1), cfg->flat_block_dim(), cfg->flat_block_len() * sizeof(float), stream>>>(
      dim0, dim1, x, y);
}

__global__ void anode_gpu_softmax_cat_nll_packed_f32_kernel(
    uint32_t len,
    uint32_t dim0,
    uint32_t dim1,
    const float *softmax,
    const uint32_t *cat_data,
    float *nll)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    uint32_t cat_k0 = cat_data[idx];
    if (cat_k0 < dim0) {
      uint32_t i = Index2::Pack(cat_k0, dim0, idx);
      nll[idx] = -logf(softmax[i]);
    } else {
      nll[idx] = 0.0f / 0.0f;
    }
  }
}

extern "C" void anode_gpu_softmax_cat_nll_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    const float *softmax,
    const uint32_t *cat_data,
    float *nll,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t len = dim1;
  anode_gpu_softmax_cat_nll_packed_f32_kernel<<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, dim0, dim1, softmax, cat_data, nll);
}

template <typename Write>
__global__ void anode_gpu_softmax_cat_nll_bwd_packed_f32_kernel(
    uint32_t len,
    uint32_t dim0,
    uint32_t dim1,
    const float *dy,
    const float *softmax,
    const uint32_t *cat_data,
    float *dx)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    uint32_t k0, i1;
    Index2::Unpack(idx, &k0, dim0, &i1);
    if (k0 < dim0 && i1 < dim1) {
      uint32_t cat_k = cat_data[i1];
      Write::Write(&dx[idx], dy[i1] * (softmax[idx] - ((float)(cat_k == k0))));
    }
  }
}

extern "C" void anode_gpu_softmax_cat_nll_bwd_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    const float *dy,
    const float *softmax,
    const uint32_t *cat_data,
    float *dx,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t len = dim0 * dim1;
  anode_gpu_softmax_cat_nll_bwd_packed_f32_kernel<AssignWrite<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, dim0, dim1, dy, softmax, cat_data, dx);
}

extern "C" void anode_gpu_softmax_cat_nll_bwd_packed_accumulate_f32(
    uint32_t dim0,
    uint32_t dim1,
    const float *dy,
    const float *softmax,
    const uint32_t *cat_data,
    float *dx,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t len = dim0 * dim1;
  anode_gpu_softmax_cat_nll_bwd_packed_f32_kernel<AccumulateWrite<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, dim0, dim1, dy, softmax, cat_data, dx);
}

__global__ void anode_gpu_softmax_nd_packed_f32_kernel(
    uint32_t len,
    uint32_t prefix_dim,
    uint32_t reduce_dim,
    uint32_t outer_dim,
    const float *x,
    float *y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    uint32_t p, i1;
    Index2::Unpack(idx, &p, prefix_dim, &i1);

    float x_max_accumulator = MaxReduce<float>::InitVal();
    for (uint32_t k0 = 0; k0 < reduce_dim; ++k0) {
      uint32_t i = Index3::Pack(p, prefix_dim, k0, reduce_dim, i1);
      float x_i = x[i];
      MaxReduce<float>::Reduce(&x_max_accumulator, x_i);
    }

    float z_sum_accumulator = AddReduce<float>::InitVal();
    for (uint32_t k0 = 0; k0 < reduce_dim; ++k0) {
      uint32_t i = Index3::Pack(p, prefix_dim, k0, reduce_dim, i1);
      float x_i = x[i];
      float z_i = expf(x_i - x_max_accumulator);
      AddReduce<float>::Reduce(&z_sum_accumulator, z_i);
    }

    for (uint32_t k0 = 0; k0 < reduce_dim; ++k0) {
      uint32_t i = Index3::Pack(p, prefix_dim, k0, reduce_dim, i1);
      float x_i = x[i];
      float z_i = expf(x_i - x_max_accumulator);
      float y_i = z_i / z_sum_accumulator;
      y[i] = y_i;
    }
  }
}

extern "C" void anode_gpu_softmax_nd_packed_f32(
    uint32_t prefix_dim,
    uint32_t dim0,
    uint32_t dim1,
    const float *x,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t len = prefix_dim * dim1;
  anode_gpu_softmax_nd_packed_f32_kernel<<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, prefix_dim, dim0, dim1, x, y);
}

__global__ void anode_gpu_softmax_nd_cat_nll_packed_f32_kernel(
    uint32_t len,
    uint32_t prefix_dim,
    uint32_t dim0,
    uint32_t dim1,
    const float *softmax,
    const uint32_t *cat_data,
    float *nll)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    uint32_t cat_k0 = cat_data[idx];
    if (cat_k0 < dim0) {
      uint32_t p, i1;
      Index2::Unpack(idx, &p, prefix_dim, &i1);
      uint32_t i = Index3::Pack(p, prefix_dim, cat_k0, dim0, i1);
      nll[idx] = -logf(softmax[i]);
    } else {
      nll[idx] = 0.0f / 0.0f;
    }
  }
}

extern "C" void anode_gpu_softmax_nd_cat_nll_packed_f32(
    uint32_t prefix_dim,
    uint32_t dim0,
    uint32_t dim1,
    const float *softmax,
    const uint32_t *cat_data,
    float *nll,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t len = prefix_dim * dim1;
  anode_gpu_softmax_nd_cat_nll_packed_f32_kernel<<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, prefix_dim, dim0, dim1, softmax, cat_data, nll);
}

template <typename Write>
__global__ void anode_gpu_softmax_nd_cat_nll_bwd_packed_f32_kernel(
    uint32_t len,
    uint32_t prefix_dim,
    uint32_t dim0,
    uint32_t dim1,
    const float *dy,
    const float *softmax,
    const uint32_t *cat_data,
    float *dx)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    uint32_t p, k0, i1;
    Index3::Unpack(idx, &p, prefix_dim, &k0, dim0, &i1);
    if (p < prefix_dim && k0 < dim0 && i1 < dim1) {
      uint32_t i = Index2::Pack(p, prefix_dim, i1);
      Write::Write(&dx[idx], dy[i] * (softmax[idx] - ((float)(cat_data[i] == k0))));
    }
  }
}

extern "C" void anode_gpu_softmax_nd_cat_nll_bwd_packed_f32(
    uint32_t prefix_dim,
    uint32_t dim0,
    uint32_t dim1,
    const float *dy,
    const float *softmax,
    const uint32_t *cat_data,
    float *dx,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t len = prefix_dim * dim0 * dim1;
  anode_gpu_softmax_nd_cat_nll_bwd_packed_f32_kernel<AssignWrite<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, prefix_dim, dim0, dim1, dy, softmax, cat_data, dx);
}

extern "C" void anode_gpu_softmax_nd_cat_nll_bwd_packed_accumulate_f32(
    uint32_t prefix_dim,
    uint32_t dim0,
    uint32_t dim1,
    const float *dy,
    const float *softmax,
    const uint32_t *cat_data,
    float *dx,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t len = prefix_dim * dim0 * dim1;
  anode_gpu_softmax_nd_cat_nll_bwd_packed_f32_kernel<AccumulateWrite<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, prefix_dim, dim0, dim1, dy, softmax, cat_data, dx);
}
