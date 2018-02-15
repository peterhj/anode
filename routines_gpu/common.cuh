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

#ifndef __ANODE_KERNELS_GPU_COMMON_CUH__
#define __ANODE_KERNELS_GPU_COMMON_CUH__

#include <math_constants.h>

#include <cstddef>
#include <cstdint>

//#define FLAT_FORALL(idx, len) for (idx = threadIdx.x + blockDim.x * blockIdx.x; idx < len; idx += blockDim.x * gridDim.x)

__device__ __forceinline__ float const_rsqrt_2_f() {
  return 0.7071067811865475f;
}

__device__ __forceinline__ uint32_t gtindex() {
  return threadIdx.x + blockDim.x * blockIdx.x;
}

__device__ __forceinline__ uint32_t gtcount() {
  return blockDim.x * gridDim.x;
}

__device__ __forceinline__ uint32_t gblock() {
  return blockIdx.x;
}

__device__ __forceinline__ uint32_t gblockcount() {
  return gridDim.x;
}

static bool check_power_of_2(uint32_t x) {
  return x && !(x & (x - 1));
}

struct KernelConfig {
  dim3 flat_grid_dim(uint32_t len) const {
    return min(max_block_ct, (len + block_sz - 1) / block_sz);
  }
  dim3 flat_block_count(uint32_t block_ct) const {
    return min(max_block_ct, block_ct);
  }
  dim3 flat_block_dim() const {
    return block_sz;
  }
  size_t flat_block_len() const {
    return block_sz;
  }

  uint32_t block_sz;
  uint32_t max_block_ct;
};

struct Index2 {
  __forceinline__ __device__ static uint32_t Pack(
      uint32_t idx0, uint32_t size0,
      uint32_t idx1)
  {
    return idx0 + size0 * idx1;
  }

  __forceinline__ __device__ static void Unpack(
      uint32_t index,
      uint32_t *idx0, uint32_t size0,
      uint32_t *idx1)
  {
    *idx0 = index % size0;
    *idx1 = index / size0;
  }
};

struct Index3 {
  __forceinline__ __device__ static uint32_t Pack(
      uint32_t idx0, uint32_t size0,
      uint32_t idx1, uint32_t size1,
      uint32_t idx2)
  {
    return idx0 + size0 * (idx1 + size1 * idx2);
  }

  __forceinline__ __device__ static void Unpack(
      uint32_t index,
      uint32_t *idx0, uint32_t size0,
      uint32_t *idx1, uint32_t size1,
      uint32_t *idx2)
  {
    *idx0 = index % size0;
    *idx1 = (index / size0) % size1;
    *idx2 = index / (size0 * size1);
  }
};

struct Index4 {
  __forceinline__ __device__ static uint32_t Pack(
      uint32_t idx0, uint32_t size0,
      uint32_t idx1, uint32_t size1,
      uint32_t idx2, uint32_t size2,
      uint32_t idx3)
  {
    return idx0 + size0 * (idx1 + size1 * (idx2 + size2 * idx3));
  }

  __forceinline__ __device__ static void Unpack(
      uint32_t index,
      uint32_t *idx0, uint32_t size0,
      uint32_t *idx1, uint32_t size1,
      uint32_t *idx2, uint32_t size2,
      uint32_t *idx3)
  {
    *idx0 = index % size0;
    *idx1 = (index / size0) % size1;
    *idx2 = (index / (size0 * size1)) % size2;
    *idx3 = index / (size0 * size1 * size2);
  }
};

template <typename T>
struct AddReduce {
  __forceinline__ __device__ static T InitVal();
  __forceinline__ __device__ static void Reduce(T *dst, T val);
  __forceinline__ __device__ static void AtomicReduce(T *dst, T val);
};

template <>
struct AddReduce<float> {
  __forceinline__ __device__ static float InitVal() {
    return 0.0f;
  }

  __forceinline__ __device__ static void Reduce(float *dst, float val) {
    *dst += val;
  }

  __forceinline__ __device__ static void AtomicReduce(float *dst, float val) {
    (void)atomicAdd(dst, val);
  }
};

template <typename T>
struct MaxReduce {
  __forceinline__ __device__ static T InitVal();
  __forceinline__ __device__ static void Reduce(T *dst, T val);
};

template <>
struct MaxReduce<float> {
  __forceinline__ __device__ static float InitVal() {
    return -CUDART_INF_F;
  }

  __forceinline__ __device__ static void Reduce(float *dst, float val) {
    *dst = max(*dst, val);
  }
};

#endif
