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
#include <math_constants.h>

template <typename T>
class SetConstantFlatMap {
public:
  __forceinline__ __device__ static void set_constant_flat_map_idx(uint32_t idx, T c, T *y) {
    y[idx] = c;
  }
};

template <typename T>
class MultConstantFlatMap {
public:
  __forceinline__ __device__ static void constant_flat_map_idx(uint32_t idx, T c, const T *x, T *y) {
    y[idx] = c * x[idx];
  }
};

template <typename T>
class CopyFlatMap {
public:
  __forceinline__ __device__ static void FlatMapIndex(uint32_t idx, const T *x, T *y) {
    y[idx] = x[idx];
  }
};

template <typename T>
class ModulusFlatMap {
public:
  __forceinline__ __device__ static void FlatMapIndex(uint32_t idx, const T *x, T *y);
};

template <>
class ModulusFlatMap<float> {
public:
  __forceinline__ __device__ static void FlatMapIndex(uint32_t idx, const float *x, float *y) {
    y[idx] = fabsf(x[idx]);
  }
};

template <typename T>
class SquareFlatMap {
public:
  __forceinline__ __device__ static void FlatMapIndex(uint32_t idx, const T *x, T *y) {
    T x_i = x[idx];
    y[idx] = x_i * x_i;
  }
};

template <typename T>
class PositiveClipFlatMap {
public:
  __forceinline__ __device__ static void FlatMapIndex(uint32_t idx, const T *x, T *y);
};

template <>
class PositiveClipFlatMap<float> {
public:
  __forceinline__ __device__ static void FlatMapIndex(uint32_t idx, const float *x, float *y) {
    float x_i = x[idx];
    y[idx] = x_i * static_cast<float>(x_i > 0.0f);
  }
};

template <typename T>
class UnitStepFlatMap {
public:
  __forceinline__ __device__ static void FlatMapIndex(uint32_t idx, const T *x, T *y);
};

template <>
class UnitStepFlatMap<float> {
public:
  __forceinline__ __device__ static void FlatMapIndex(uint32_t idx, const float *x, float *y) {
    float x_i = x[idx];
    y[idx] = static_cast<float>(x_i > 0.0f);
  }
};

template <typename T>
class LogPositiveClipFlatMap {
public:
  __forceinline__ __device__ static void FlatMapIndex(uint32_t idx, const T *x, T *y);
};

template <>
class LogPositiveClipFlatMap<float> {
public:
  __forceinline__ __device__ static void FlatMapIndex(uint32_t idx, const float *x, float *y) {
    float x_i = x[idx];
    if (x_i > 0.0f) {
      y[idx] = logf(x_i);
    } else {
      y[idx] = -CUDART_INF_F;
    }
  }
};

template <typename T>
class PositiveReciprocalFlatMap {
public:
  __forceinline__ __device__ static void FlatMapIndex(uint32_t idx, const T *x, T *y);
};

template <>
class PositiveReciprocalFlatMap<float> {
public:
  __forceinline__ __device__ static void FlatMapIndex(uint32_t idx, const float *x, float *y) {
    float x_i = x[idx];
    if (x_i > 0.0f) {
      y[idx] = 1.0f / x_i;
    } else {
      y[idx] = 0.0f;
    }
  }
};

template <typename T>
class NormalCDFFlatMap {
public:
  __forceinline__ __device__ static void FlatMapIndex(uint32_t idx, const T *x, T *y);
};

template <>
class NormalCDFFlatMap<float> {
public:
  __forceinline__ __device__ static void FlatMapIndex(uint32_t idx, const float *x, float *y) {
    float x_i = x[idx];
    y[idx] = 0.5f * (1.0f + erff(x_i * const_rsqrt_2_f()));
  }
};

template <typename T>
class TanhFlatMap {
public:
  __forceinline__ __device__ static void FlatMapIndex(uint32_t idx, const T *x, T *y);
};

template <>
class TanhFlatMap<float> {
public:
  __forceinline__ __device__ static void FlatMapIndex(uint32_t idx, const float *x, float *y) {
    y[idx] = tanhf(x[idx]);
  }
};

template <typename T>
class Rcosh2FlatMap {
public:
  __forceinline__ __device__ static void FlatMapIndex(uint32_t idx, const T *x, T *y);
};

template <>
class Rcosh2FlatMap<float> {
public:
  __forceinline__ __device__ static void FlatMapIndex(uint32_t idx, const float *x, float *y) {
    float x_i = x[idx];
    float chx_i = coshf(x_i);
    y[idx] = 1.0f / (chx_i * chx_i);
  }
};

template <typename T, typename Map>
__global__ void anode_gpu_generic_flat_map_kernel(
    uint32_t len,
    const T *x,
    T *y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    Map::FlatMapIndex(idx, x, y);
  }
}

extern "C" void anode_gpu_copy_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    KernelConfig cfg,
    cudaStream_t stream)
{
  anode_gpu_generic_flat_map_kernel<float, CopyFlatMap<float>><<<cfg.flat_grid_dim(len), cfg.flat_block_dim(), 0, stream>>>(
      len, x, y);
}

extern "C" void anode_gpu_modulus_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    KernelConfig cfg,
    cudaStream_t stream)
{
  anode_gpu_generic_flat_map_kernel<float, ModulusFlatMap<float>><<<cfg.flat_grid_dim(len), cfg.flat_block_dim(), 0, stream>>>(
      len, x, y);
}

extern "C" void anode_gpu_square_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    KernelConfig cfg,
    cudaStream_t stream)
{
  anode_gpu_generic_flat_map_kernel<float, SquareFlatMap<float>><<<cfg.flat_grid_dim(len), cfg.flat_block_dim(), 0, stream>>>(
      len, x, y);
}

extern "C" void anode_gpu_positive_clip_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    KernelConfig cfg,
    cudaStream_t stream)
{
  anode_gpu_generic_flat_map_kernel<float, PositiveClipFlatMap<float>><<<cfg.flat_grid_dim(len), cfg.flat_block_dim(), 0, stream>>>(
      len, x, y);
}

extern "C" void anode_gpu_unit_step_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    KernelConfig cfg,
    cudaStream_t stream)
{
  anode_gpu_generic_flat_map_kernel<float, UnitStepFlatMap<float>><<<cfg.flat_grid_dim(len), cfg.flat_block_dim(), 0, stream>>>(
      len, x, y);
}

extern "C" void anode_gpu_normal_cdf_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    KernelConfig cfg,
    cudaStream_t stream)
{
  anode_gpu_generic_flat_map_kernel<float, NormalCDFFlatMap<float>><<<cfg.flat_grid_dim(len), cfg.flat_block_dim(), 0, stream>>>(
      len, x, y);
}

extern "C" void anode_gpu_tanh_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    KernelConfig cfg,
    cudaStream_t stream)
{
  anode_gpu_generic_flat_map_kernel<float, TanhFlatMap<float>><<<cfg.flat_grid_dim(len), cfg.flat_block_dim(), 0, stream>>>(
      len, x, y);
}

extern "C" void anode_gpu_rcosh2_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    KernelConfig cfg,
    cudaStream_t stream)
{
  anode_gpu_generic_flat_map_kernel<float, Rcosh2FlatMap<float>><<<cfg.flat_grid_dim(len), cfg.flat_block_dim(), 0, stream>>>(
      len, x, y);
}
