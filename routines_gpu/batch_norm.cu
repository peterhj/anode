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
__global__ void anode_gpu_batch_norm_3d1_kernel(
    uint32_t len,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    T epsilon,
    const T *x,
    const T *mean,
    const T *var,
    T *y)
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
      y[idx] = (x[idx] - mean[i1]) / (sqrtf(var[i1]) + epsilon);
    }
  }
}

template <typename T>
__global__ void anode_gpu_batch_norm_3d1_bwd_x_kernel(
    uint32_t len,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    T epsilon,
    const T *dy,
    const T *mean,
    const T *var,
    T *dx)
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
      dx[idx] = dy[idx] / (sqrtf(var[i1]) + epsilon);
    }
  }
}

template <typename T>
__global__ void anode_gpu_batch_norm_3d1_bwd_mean_kernel(
    uint32_t len,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    T epsilon,
    const T *dy,
    const T *mean,
    const T *var,
    T *dx)
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
      // FIXME
    }
  }
}

template <typename T>
__global__ void anode_gpu_batch_norm_3d1_bwd_var_kernel(
    uint32_t len,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    T epsilon,
    const T *dy,
    const T *mean,
    const T *var,
    T *dx)
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
      // FIXME
    }
  }
}
