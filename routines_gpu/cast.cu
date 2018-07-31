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
#include <cassert>
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void anode_gpu_upcast_u8_packed_u32_kernel(
    uint32_t len,
    const uint8_t *x,
    uint32_t *y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    y[idx] = ((uint32_t)(x[idx]));
  }
}

extern "C" void anode_gpu_upcast_u8_packed_u32(
    uint32_t len,
    const uint8_t *x,
    uint32_t *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  anode_gpu_upcast_u8_packed_u32_kernel<<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, x, y);
}
