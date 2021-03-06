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

#ifndef __ANODE_ROUTINES_GPU_LIB_H__
#define __ANODE_ROUTINES_GPU_LIB_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

struct KernelConfig;
struct CUstream_st;

// "batch_norm.cu"

void anode_gpu_batch_mean_3d1_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *x,
    float *mean,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_batch_mean_bwd_3d1_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dmean,
    float *dx,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_batch_mean_bwd_3d1_packed_accumulate_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dmean,
    float *dx,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_batch_var_3d1_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float epsilon,
    const float *x,
    const float *mean,
    float *var,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_batch_var_bwd_3d1_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dvar,
    const float *x,
    const float *mean,
    float *dx,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_batch_var_bwd_3d1_packed_accumulate_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dvar,
    const float *x,
    const float *mean,
    float *dx,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_batch_var_bwd_mean_3d1_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dvar,
    const float *x,
    const float *mean,
    float *dmean,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_batch_var_bwd_mean_3d1_packed_accumulate_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dvar,
    const float *x,
    const float *mean,
    float *dmean,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_batch_norm_3d1_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *x,
    const float *mean,
    const float *var,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_batch_norm_bwd_3d1_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dy,
    const float *var,
    float *dx,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_batch_norm_bwd_3d1_packed_accumulate_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dy,
    const float *var,
    float *dx,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_batch_norm_bwd_mean_3d1_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dy,
    const float *var,
    float *dmean,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_batch_norm_bwd_mean_3d1_packed_accumulate_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dy,
    const float *var,
    float *dmean,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_batch_norm_bwd_var_3d1_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dy,
    const float *x,
    const float *mean,
    const float *var,
    float *dvar,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_batch_norm_bwd_var_3d1_packed_accumulate_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dy,
    const float *x,
    const float *mean,
    const float *var,
    float *dvar,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_batch_norm_bwd_var_v2_3d1_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dy,
    const float *y,
    const float *var,
    float *dvar,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_batch_norm_bwd_var_v2_3d1_packed_accumulate_f32(
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    const float *dy,
    const float *y,
    const float *var,
    float *dvar,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);

// "bcast_flat_linear.cu"

/*void anode_gpu_bcast_flat_mult_I1b_I2ab_Oab_packed_f32(
    uint32_t inner_dim,
    uint32_t bcast_dim,
    const float *lx,
    const float *rx,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_bcast_flat_mult_add_I1b_I2ab_Oab_packed_f32(
    uint32_t inner_dim,
    uint32_t bcast_dim,
    const float *lx,
    const float *rx,
    const float *shift,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_bcast_flat_mult_I1b_I2abc_Oabc_packed_f32(
    uint32_t inner_dim,
    uint32_t bcast_dim,
    uint32_t outer_dim,
    const float *lx,
    const float *rx,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_bcast_flat_mult_add_I1b_I2abc_Oabc_packed_f32(
    uint32_t inner_dim,
    uint32_t bcast_dim,
    uint32_t outer_dim,
    const float *lx,
    const float *rx,
    const float *shift,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);*/

// "flat_linear.cu"
/*void anode_gpu_flat_mult_f32(
    uint32_t len,
    const float *lx,
    const float *rx,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_flat_mult_add_f32(
    uint32_t len,
    const float *lx,
    const float *rx,
    const float *shift,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);*/

// "cast.cu"

void anode_gpu_upcast_u8_packed_u32(
    uint32_t len,
    const uint8_t *x,
    uint32_t *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);

// "discrete.cu"

void anode_gpu_discrete_one_hot_3d1_packed_f32(
    uint32_t dim0,
    uint32_t cat_dim1,
    uint32_t dim2,
    const uint32_t *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);

// "f1.cu"

void anode_gpu_smooth_negative_f1_loss_3d1_packed_f32(
    uint32_t dim0,
    uint32_t cat_dim1,
    uint32_t dim2,
    float epsilon,
    const float *x,
    const float *target,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_smooth_negative_f1_loss_3d1_bwd_packed_f32(
    uint32_t dim0,
    uint32_t cat_dim1,
    uint32_t dim2,
    float epsilon,
    const float *dy,
    const float *x,
    const float *target,
    float *dx,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);

// "flat_join.cu"
void anode_gpu_M1_copy_map_M2_unit_step_map_R_product_reduce_flat_join_f32(
    uint32_t len,
    const float *x1,
    const float *x2,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);

// "flat_map.cu"
void anode_gpu_leaky_relu_flat_map_f32(
    uint32_t len,
    float c,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_leaky_relu_flat_map_bwd_f32(
    uint32_t len,
    float c,
    const float *dy,
    const float *y,
    float *dx,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_copy_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_modulus_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_square_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_sqrt_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_positive_clip_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_positive_clip_flat_map_bwd_f32(
    uint32_t len,
    const float *dy,
    const float *y,
    float *dx,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_logistic_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_logistic_flat_map_bwd_f32(
    uint32_t len,
    const float *dy,
    const float *y,
    float *dx,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_unit_step_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_normal_cdf_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_tanh_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_rcosh2_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);

// "quantize.cu"

void anode_gpu_dequantize_u8_packed_f32(
    uint32_t len,
    float lo,
    float hi,
    const uint8_t *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);

// "reduce.cu"

/*void anode_gpu_sum_Iab_Ob_packed_deterministic_f32(
    uint32_t reduce_dim,
    uint32_t outer_dim,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_sum_Iabc_Ob_packed_deterministic_f32(
    uint32_t reduce_inner_dim,
    uint32_t mid_dim,
    uint32_t reduce_outer_dim,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_square_sum_Iabc_Ob_packed_deterministic_f32(
    uint32_t reduce_inner_dim,
    uint32_t mid_dim,
    uint32_t reduce_outer_dim,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);*/

// "softmax.cu"

void anode_gpu_softmax_packed_block_f32(
    uint32_t dim0,
    uint32_t dim1,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_softmax_packed_deterministic_f32(
    uint32_t dim0,
    uint32_t dim1,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_softmax_cat_nll_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    const float *softmax,
    const uint32_t *cat_data,
    float *nll,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_softmax_cat_nll_bwd_packed_f32(
    uint32_t dim0,
    uint32_t dim1,
    const float *dy,
    const float *softmax,
    const uint32_t *cat_data,
    float *dx,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_softmax_cat_nll_bwd_packed_accumulate_f32(
    uint32_t dim0,
    uint32_t dim1,
    const float *dy,
    const float *softmax,
    const uint32_t *cat_data,
    float *dx,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_softmax_nd_packed_f32(
    uint32_t prefix_dim,
    uint32_t dim0,
    uint32_t dim1,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_softmax_nd_cat_nll_packed_f32(
    uint32_t prefix_dim,
    uint32_t dim0,
    uint32_t dim1,
    const float *softmax,
    const uint32_t *cat_data,
    float *nll,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_softmax_nd_cat_nll_bwd_packed_f32(
    uint32_t prefix_dim,
    uint32_t dim0,
    uint32_t dim1,
    const float *dy,
    const float *softmax,
    const uint32_t *cat_data,
    float *dx,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void anode_gpu_softmax_nd_cat_nll_bwd_packed_accumulate_f32(
    uint32_t prefix_dim,
    uint32_t dim0,
    uint32_t dim1,
    const float *dy,
    const float *softmax,
    const uint32_t *cat_data,
    float *dx,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
