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
#include <cuda_runtime.h>

__global__ void anode_gpu_discrete_one_hot_3d1_packed_f32_kernel(
    uint32_t dst_flat_len,
    uint32_t dim0,
    uint32_t cat_dim1,
    uint32_t dim2,
    const uint32_t *cat_data,
    float *y)
{
  for (uint32_t idx = gtindex(); idx < dst_flat_len; idx += gtcount()) {
    uint32_t i0, dst_i1, i2;
    Index3::Unpack(idx,
        &i0, dim0,
        &dst_i1, cat_dim1,
        &i2);
    uint32_t src_idx = Index2::Pack(
        i0, dim0,
        i2);
    float y_i;
    if (cat_data[src_idx] == dst_i1) {
      y_i = 1.0f;
    } else {
      y_i = 0.0f;
    }
    y[idx] = y_i;
  }
}

extern "C" void anode_gpu_discrete_one_hot_3d1_packed_f32(
    uint32_t dim0,
    uint32_t cat_dim1,
    uint32_t dim2,
    const uint32_t *cat_data,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t len = dim0 * cat_dim1 * dim2;
  anode_gpu_discrete_one_hot_3d1_packed_f32_kernel<<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, dim0, cat_dim1, dim2, cat_data, y);
}
