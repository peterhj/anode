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

#ifndef __ANODE_KERNELS_GPU_LIB_H__
#define __ANODE_KERNELS_GPU_LIB_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

struct KernelConfig;
struct CUstream_st;

// "flat_linear.cu"
void anode_gpu_flat_mult_f32(
    uint32_t len,
    const float *lx,
    const float *rx,
    float *y,
    struct KernelConfig cfg,
    struct CUstream_st *stream);
void anode_gpu_flat_mult_add_f32(
    uint32_t len,
    const float *lx,
    const float *rx,
    const float *shift,
    float *y,
    struct KernelConfig cfg,
    struct CUstream_st *stream);

// "flat_map.cu"
void anode_gpu_copy_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    struct KernelConfig cfg,
    struct CUstream_st *stream);
void anode_gpu_modulus_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    struct KernelConfig cfg,
    struct CUstream_st *stream);
void anode_gpu_square_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    struct KernelConfig cfg,
    struct CUstream_st *stream);
void anode_gpu_positive_clip_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    struct KernelConfig cfg,
    struct CUstream_st *stream);
void anode_gpu_unit_step_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    struct KernelConfig cfg,
    struct CUstream_st *stream);
void anode_gpu_normal_cdf_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    struct KernelConfig cfg,
    struct CUstream_st *stream);
void anode_gpu_tanh_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    struct KernelConfig cfg,
    struct CUstream_st *stream);
void anode_gpu_rcosh2_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    struct KernelConfig cfg,
    struct CUstream_st *stream);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
