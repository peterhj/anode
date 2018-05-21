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
#include <cuda_runtime.h>
#include <math_constants.h>

template <typename T>
class Copy_UnitStep_ProductReduce_FlatJoin {
public:
  __forceinline__ __device__ static void FlatJoinIndex(uint32_t idx, const T *x1, const T *x2, T *y);
};

template <>
class Copy_UnitStep_ProductReduce_FlatJoin<float> {
public:
  __forceinline__ __device__ static void FlatJoinIndex(uint32_t idx, const float *x1, const float *x2, float *y) {
    float x1_i = x1[idx];
    float x2_i = x2[idx];
    y[idx] = x1_i * static_cast<float>(x2_i > 0.0f);
  }
};

template <typename T, typename Map2FlatJoin>
__global__ void anode_gpu_generic_map2_flat_join_kernel(
    uint32_t len,
    const T *x1,
    const T *x2,
    T *y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    Map2FlatJoin::FlatJoinIndex(idx, x1, x2, y);
  }
}

extern "C" void anode_gpu_M1_copy_map_M2_unit_step_map_R_product_reduce_flat_join_f32(
    uint32_t len,
    const float *x1,
    const float *x2,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  anode_gpu_generic_map2_flat_join_kernel<float, Copy_UnitStep_ProductReduce_FlatJoin<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, x1, x2, y);
}
