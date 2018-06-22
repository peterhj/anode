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

__global__ void anode_gpu_softmax_cat_nll_packed_kernel_f32(
    uint32_t len,
    uint32_t dim0,
    uint32_t dim1,
    const float *softmax,
    const uint32_t *cat_data,
    float *nll)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    if (idx < dim1) {
      uint32_t cat_k = cat_data[idx];
      if (cat_k < dim0) {
        uint32_t i = Index2::Pack(cat_k, dim0, idx);
        nll[idx] = -logf(softmax[i]);
      } else {
        nll[idx] = 0.0f / 0.0f;
      }
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
  anode_gpu_softmax_cat_nll_packed_kernel_f32<<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, dim0, dim1, softmax, cat_data, nll);
}

template <typename Write>
__global__ void anode_gpu_softmax_cat_nll_bwd_packed_kernel_f32(
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
      // TODO: check sign.
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
  anode_gpu_softmax_cat_nll_bwd_packed_kernel_f32<AssignWrite<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
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
  anode_gpu_softmax_cat_nll_bwd_packed_kernel_f32<AccumulateWrite<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, dim0, dim1, dy, softmax, cat_data, dx);
}
