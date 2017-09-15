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
__global__ void anode_gpu_flat_mult_kernel(
    uint32_t len,
    const T* lx,
    const T* rx,
    T* y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    T lx_i = lx[idx];
    T rx_i = rx[idx];
    y[idx] = lx_i * rx_i;
  }
}

extern "C" void anode_gpu_flat_mult_f32(
    uint32_t len,
    const float *lx,
    const float *rx,
    float *y,
    KernelConfig cfg,
    cudaStream_t stream)
{
  anode_gpu_flat_mult_kernel<float><<<cfg.flat_grid_dim(len), cfg.flat_block_dim(), 0, stream>>>(
      len, lx, rx, y);
}

template <typename T>
__global__ void anode_gpu_flat_mult_add_kernel(
    uint32_t len,
    const T* lx,
    const T* rx,
    const T* shift,
    T* y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    T lx_i = lx[idx];
    T rx_i = rx[idx];
    T shift_i = shift[idx];
    y[idx] = lx_i * rx_i + shift_i;
  }
}

extern "C" void anode_gpu_flat_mult_add_f32(
    uint32_t len,
    const float *lx,
    const float *rx,
    const float *shift,
    float *y,
    KernelConfig cfg,
    cudaStream_t stream)
{
  anode_gpu_flat_mult_add_kernel<float><<<cfg.flat_grid_dim(len), cfg.flat_block_dim(), 0, stream>>>(
      len, lx, rx, shift, y);
}
