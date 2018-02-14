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
#include <cuda_runtime.h>

template <typename T>
__global__ void anode_gpu_bcast_flat_mult_I1b_I2ab_Oab_packed_kernel(
    uint32_t len,
    uint32_t inner_dim,
    uint32_t bcast_dim,
    const T *lx,
    const T *rx,
    T *y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    uint32_t _inner_idx, bcast_idx;
    Index2::Unpack(
        idx,
        &_inner_idx, inner_dim,
        &bcast_idx);
    T lx_i = lx[bcast_idx];
    T rx_i = rx[idx];
    y[idx] = lx_i * rx_i;
  }
}

extern "C" void anode_gpu_bcast_flat_mult_I1b_I2ab_Oab_packed_f32(
    uint32_t inner_dim,
    uint32_t bcast_dim,
    const float *lx,
    const float *rx,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t len = inner_dim * bcast_dim;
  anode_gpu_bcast_flat_mult_I1b_I2ab_Oab_packed_kernel<float><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, inner_dim, bcast_dim, lx, rx, y);
}

template <typename T>
__global__ void anode_gpu_bcast_flat_mult_add_I1b_I2ab_I3b_Oab_packed_kernel(
    uint32_t len,
    uint32_t inner_dim,
    uint32_t bcast_dim,
    const T *lx,
    const T *rx,
    const T *shift,
    T *y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    uint32_t _inner_idx, bcast_idx;
    Index2::Unpack(
        idx,
        &_inner_idx, inner_dim,
        &bcast_idx);
    T lx_i = lx[bcast_idx];
    T rx_i = rx[idx];
    T shift_i = shift[bcast_idx];
    y[idx] = lx_i * rx_i + shift_i;
  }
}

extern "C" void anode_gpu_bcast_flat_mult_add_I1b_I2ab_I3b_Oab_packed_f32(
    uint32_t inner_dim,
    uint32_t bcast_dim,
    const float *lx,
    const float *rx,
    const float *shift,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t len = inner_dim * bcast_dim;
  anode_gpu_bcast_flat_mult_add_I1b_I2ab_I3b_Oab_packed_kernel<float><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, inner_dim, bcast_dim, lx, rx, shift, y);
}

template <typename T>
__global__ void anode_gpu_bcast_flat_mult_I1b_I2abc_Oabc_packed_kernel(
    uint32_t len,
    uint32_t inner_dim,
    uint32_t bcast_dim,
    uint32_t outer_dim,
    const T *lx,
    const T *rx,
    T *y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    uint32_t _inner_idx, bcast_idx, _outer_idx;
    Index3::Unpack(
        idx,
        &_inner_idx, inner_dim,
        &bcast_idx, bcast_dim,
        &_outer_idx);
    T lx_i = lx[bcast_idx];
    T rx_i = rx[idx];
    y[idx] = lx_i * rx_i;
  }
}

extern "C" void anode_gpu_bcast_flat_mult_I1b_I2abc_Oabc_packed_f32(
    uint32_t inner_dim,
    uint32_t bcast_dim,
    uint32_t outer_dim,
    const float *lx,
    const float *rx,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t len = inner_dim * bcast_dim * outer_dim;
  anode_gpu_bcast_flat_mult_I1b_I2abc_Oabc_packed_kernel<float><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, inner_dim, bcast_dim, outer_dim, lx, rx, y);
}

template <typename T>
__global__ void anode_gpu_bcast_flat_mult_add_I1b_I2abc_I3b_Oabc_packed_kernel(
    uint32_t len,
    uint32_t inner_dim,
    uint32_t bcast_dim,
    uint32_t outer_dim,
    const T *lx,
    const T *rx,
    const T *shift,
    T *y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    uint32_t _inner_idx, bcast_idx, _outer_idx;
    Index3::Unpack(
        idx,
        &_inner_idx, inner_dim,
        &bcast_idx, bcast_dim,
        &_outer_idx);
    T lx_i = lx[bcast_idx];
    T rx_i = rx[idx];
    T shift_i = shift[bcast_idx];
    y[idx] = lx_i * rx_i + shift_i;
  }
}

extern "C" void anode_gpu_bcast_flat_mult_add_I1b_I2abc_I3b_Oabc_packed_f32(
    uint32_t inner_dim,
    uint32_t bcast_dim,
    uint32_t outer_dim,
    const float *lx,
    const float *rx,
    const float *shift,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t len = inner_dim * bcast_dim * outer_dim;
  anode_gpu_bcast_flat_mult_add_I1b_I2abc_I3b_Oabc_packed_kernel<float><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, inner_dim, bcast_dim, outer_dim, lx, rx, shift, y);
}
