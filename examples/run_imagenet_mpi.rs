#![feature(alloc_system)]

//#[cfg(feature = "mpi")] extern crate cray_link;
extern crate anode;
extern crate colorimage;
extern crate gpudevicemem;
extern crate memarray;
extern crate minidata;
extern crate rand;
extern crate sharedmem;

//#[cfg(feature = "mpi")] use cray_link::*;
use anode::*;
use anode::log::*;
use anode::ops::*;
use anode::proc::*;
use anode::proc_dist::*;
use anode::utils::*;
use colorimage::*;
use gpudevicemem::*;
use gpudevicemem::array::*;
use memarray::*;
use minidata::*;
use minidata::datasets::imagenet::*;
use minidata::image::*;
use minidata::utils::*;
use rand::prelude::*;
use rand::distributions::{Distribution, Uniform, Normal};
//use rand::rngs::mock::*;
use sharedmem::*;

use std::cmp::{max, min};
use std::env;
use std::path::{PathBuf};
use std::rc::{Rc};
use std::sync::{Arc};

pub trait XavierLinearInit<Shape, T, R: Rng> {
  type RValue;

  fn xavier_linear_init(shape: Shape, src: usize, dst: usize, seed_rng: &mut R) -> Self::RValue;
}

pub trait XavierConv2dInit<Shape, T, R: Rng> {
  type RValue;

  fn xavier_conv2d_init(shape: Shape, ker_sz: [usize; 2], src: usize, dst: usize, seed_rng: &mut R) -> Self::RValue;
}

pub trait KaimingConv2dInit<Shape, T, R: Rng> {
  type RValue;

  fn kaiming_conv2d_init(shape: Shape, ker_sz: [usize; 2], src: usize, dst: usize, seed_rng: &mut R) -> Self::RValue;
}

impl<R: Rng> XavierLinearInit<[usize; 2], f32, R> for GPUDeviceArray2d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn xavier_linear_init(shape: [usize; 2], src_ch: usize, dst_ch: usize, seed_rng: &mut R) -> Self::RValue {
    //let seed = seed_rng.next_u64();
    Rc::new(move |_, conn: GPUDeviceConn| {
      // TODO: seed the local rng here.
      let mut h_arr = MemArray2d::<f32>::zeros(shape);
      {
        let half_width = (6.0 / (src_ch + dst_ch) as f64).sqrt();
        let dist = Uniform::new_inclusive(-half_width, half_width);
        let mut v = h_arr.as_view_mut();
        let xs = v.flat_slice_mut().unwrap();
        for x in xs.iter_mut() {
          *x = dist.sample(&mut thread_rng()) as f32;
        }
      }
      let mut arr = GPUDeviceArray2d::<f32>::zeros(shape, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}

impl<R: Rng> XavierConv2dInit<[usize; 4], f32, R> for GPUDeviceArray4d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn xavier_conv2d_init(shape: [usize; 4], ker_sz: [usize; 2], src_ch: usize, dst_ch: usize, seed_rng: &mut R) -> Self::RValue {
    //let seed = seed_rng.next_u64();
    Rc::new(move |_, conn: GPUDeviceConn| {
      // TODO: seed the local rng here.
      let mut h_arr = MemArray4d::<f32>::zeros(shape);
      {
        // TODO: distribution.
        let half_width = (6.0 / (ker_sz[0] * ker_sz[1] * (src_ch + dst_ch)) as f64).sqrt();
        let dist = Uniform::new_inclusive(-half_width, half_width);
        let mut v = h_arr.as_view_mut();
        let xs = v.flat_slice_mut().unwrap();
        for x in xs.iter_mut() {
          *x = dist.sample(&mut thread_rng()) as f32;
        }
      }
      let mut arr = GPUDeviceArray4d::<f32>::zeros(shape, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}

impl<R: Rng> KaimingConv2dInit<[usize; 4], f32, R> for GPUDeviceArray4d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn kaiming_conv2d_init(shape: [usize; 4], ker_sz: [usize; 2], src_ch: usize, dst_ch: usize, seed_rng: &mut R) -> Self::RValue {
    //let seed = seed_rng.next_u64();
    Rc::new(move |_, conn: GPUDeviceConn| {
      // TODO: seed the local rng here.
      let mut h_arr = MemArray4d::<f32>::zeros(shape);
      {
        let mean = 0.0;
        let std = (2.0 / (ker_sz[0] * ker_sz[1] * src_ch) as f64).sqrt();
        let dist = Normal::new(mean, std);
        let mut v = h_arr.as_view_mut();
        let xs = v.flat_slice_mut().unwrap();
        for x in xs.iter_mut() {
          *x = dist.sample(&mut thread_rng()) as f32;
        }
      }
      let mut arr = GPUDeviceArray4d::<f32>::zeros(shape, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}

fn build_linear(x: Val<GPUDeviceOuterBatchArray1d<f32>>, src_ch: usize, dst_ch: usize, params: &mut NodeVec) -> Val<GPUDeviceOuterBatchArray1d<f32>> {
  // TODO
  let w = src(GPUDeviceArray2d::<f32>::xavier_linear_init([dst_ch, src_ch], src_ch, dst_ch, &mut thread_rng()));
  //let b = src(GPUDeviceArray1d::<f32>::zeros_init(dst_ch));
  params.push_val(w.clone());
  //params.push_val(b.clone());
  let x = w.mult(x);
  //let x = w.mult_add(x, b);
  x
}

fn _build_batch_norm_conv(x: Val<GPUDeviceOuterBatchArray3d<f32>>, online: Val<bool>, avg_rate: Val<f32>, conv_shape: Conv2dShape, params: &mut NodeVec, online_stats: &mut NodeVec, avg_stats: &mut NodeVec) -> Val<GPUDeviceOuterBatchArray3d<f32>> {
  let w = src(GPUDeviceArray4d::<f32>::kaiming_conv2d_init(
      [conv_shape.ker_dims[0], conv_shape.ker_dims[1], conv_shape.src_features, conv_shape.features],
      conv_shape.ker_dims,
      conv_shape.src_features,
      conv_shape.features,
      &mut thread_rng()));
  params.push_val(w.clone());
  let x = w.conv(conv_shape.clone(), x);
  x
}

fn build_batch_norm_conv(x: Val<GPUDeviceOuterBatchArray3d<f32>>, online: Val<bool>, avg_rate: Val<f32>, conv_shape: Conv2dShape, params: &mut NodeVec, online_stats: &mut NodeVec, avg_stats: &mut NodeVec) -> Val<GPUDeviceOuterBatchArray3d<f32>> {
  let w = src(GPUDeviceArray4d::<f32>::kaiming_conv2d_init(
      [conv_shape.ker_dims[0], conv_shape.ker_dims[1], conv_shape.src_features, conv_shape.features],
      conv_shape.ker_dims,
      conv_shape.src_features,
      conv_shape.features,
      &mut thread_rng()));
  params.push_val(w.clone());
  let x = w.conv(conv_shape.clone(), x);
  let epsilon = 1.0e-6_f32;
  let (x, x_mean, x_var, x_avg_mean, x_avg_var) = x.batch_normalize_2d([0, 1], online, avg_rate, epsilon);
  online_stats.push_val(x_mean);
  online_stats.push_val(x_var);
  avg_stats.push_val(x_avg_mean);
  avg_stats.push_val(x_avg_var);
  // TODO: broadcast mult add here.
  /*let scale = src(GPUDeviceArray1d::<f32>::ones_init(conv_shape.features));
  let shift = src(GPUDeviceArray1d::<f32>::zeros_init(conv_shape.features));
  let x = scale.broadcast_1d_mult_add(2, x, shift);
  params.push_val(scale.clone());
  params.push_val(shift.clone());*/
  x
}

fn build_residual2_conv(x: Val<GPUDeviceOuterBatchArray3d<f32>>, online: Val<bool>, avg_rate: Val<f32>, conv_shape: Conv2dShape, params: &mut NodeVec, online_stats: &mut NodeVec, avg_stats: &mut NodeVec) -> Val<GPUDeviceOuterBatchArray3d<f32>> {
  // TODO
  let mut conv1 = conv_shape;
  conv1.ker_dims = [3, 3];
  conv1.stride = [1, 1];
  conv1.zero_pad = [1, 1];
  let y = build_batch_norm_conv(x.clone(), online.clone(), avg_rate.clone(), conv1, params, online_stats, avg_stats);
  //let y = y.positive_clip();
  let y = y.positive_clip_inplace();
  // TODO
  let mut conv2 = conv_shape;
  conv2.ker_dims = [3, 3];
  conv2.stride = [1, 1];
  conv2.zero_pad = [1, 1];
  let y = build_batch_norm_conv(y, online.clone(), avg_rate.clone(), conv2, params, online_stats, avg_stats);
  let y = x + y;
  //let y = sum_inplace_unstable(vec![x, y]);
  let y = y.positive_clip();
  //let y = y.positive_clip_inplace();
  y
}

fn build_proj_residual2_conv(x: Val<GPUDeviceOuterBatchArray3d<f32>>, online: Val<bool>, avg_rate: Val<f32>, conv_shape: Conv2dShape, src_dims: [usize; 2], src_features: usize, params: &mut NodeVec, online_stats: &mut NodeVec, avg_stats: &mut NodeVec) -> Val<GPUDeviceOuterBatchArray3d<f32>> {
  // TODO
  let mut conv1 = conv_shape;
  conv1.src_dims[0] = src_dims[0];
  conv1.src_dims[1] = src_dims[1];
  conv1.src_features = src_features;
  conv1.ker_dims = [3, 3];
  conv1.zero_pad = [1, 1];
  let y = build_batch_norm_conv(x.clone(), online.clone(), avg_rate.clone(), conv1, params, online_stats, avg_stats);
  //let y = y.positive_clip();
  let y = y.positive_clip_inplace();
  // TODO
  let mut conv2 = conv_shape;
  conv2.ker_dims = [3, 3];
  conv2.stride = [1, 1];
  conv2.zero_pad = [1, 1];
  let y = build_batch_norm_conv(y, online.clone(), avg_rate.clone(), conv2, params, online_stats, avg_stats);
  // TODO
  let mut proj_conv = conv_shape;
  proj_conv.src_dims[0] = src_dims[0];
  proj_conv.src_dims[1] = src_dims[1];
  proj_conv.src_features = src_features;
  proj_conv.ker_dims = [1, 1];
  proj_conv.zero_pad = [0, 0];
  let proj = build_batch_norm_conv(x, online.clone(), avg_rate.clone(), proj_conv, params, online_stats, avg_stats);
  let y = proj + y;
  //let y = sum_inplace_unstable(vec![x, y]);
  let y = y.positive_clip();
  //let y = y.positive_clip_inplace();
  y
}

fn build_residual3_conv(x: Val<GPUDeviceOuterBatchArray3d<f32>>, online: Val<bool>, avg_rate: Val<f32>, conv_shape: Conv2dShape, expand: usize, params: &mut NodeVec, online_stats: &mut NodeVec, avg_stats: &mut NodeVec) -> Val<GPUDeviceOuterBatchArray3d<f32>> {
  // TODO
  let mut conv1 = conv_shape;
  conv1.src_features *= expand;
  conv1.ker_dims = [1, 1];
  conv1.stride = [1, 1];
  conv1.zero_pad = [0, 0];
  let y = build_batch_norm_conv(x.clone(), online.clone(), avg_rate.clone(), conv1, params, online_stats, avg_stats);
  //let y = y.positive_clip();
  let y = y.positive_clip_inplace();
  // TODO
  let mut conv2 = conv_shape;
  conv2.ker_dims = [3, 3];
  conv2.stride = [1, 1];
  conv2.zero_pad = [1, 1];
  let y = build_batch_norm_conv(y, online.clone(), avg_rate.clone(), conv2, params, online_stats, avg_stats);
  //let y = y.positive_clip();
  let y = y.positive_clip_inplace();
  // TODO
  let mut conv3 = conv_shape;
  conv3.ker_dims = [1, 1];
  conv3.features *= expand;
  conv3.stride = [1, 1];
  conv3.zero_pad = [0, 0];
  let y = build_batch_norm_conv(y, online.clone(), avg_rate.clone(), conv3, params, online_stats, avg_stats);
  let y = x + y;
  //let y = sum_inplace_unstable(vec![x, y]);
  let y = y.positive_clip();
  //let y = y.positive_clip_inplace();
  y
}

fn build_proj_residual3_conv(x: Val<GPUDeviceOuterBatchArray3d<f32>>, online: Val<bool>, avg_rate: Val<f32>, conv_shape: Conv2dShape, src_dims: [usize; 2], src_features: usize, expand: usize, params: &mut NodeVec, online_stats: &mut NodeVec, avg_stats: &mut NodeVec) -> Val<GPUDeviceOuterBatchArray3d<f32>> {
  // TODO
  let mut conv1 = conv_shape;
  conv1.src_dims[0] = src_dims[0];
  conv1.src_dims[1] = src_dims[1];
  conv1.src_features = src_features;
  conv1.ker_dims = [1, 1];
  conv1.zero_pad = [0, 0];
  let y = build_batch_norm_conv(x.clone(), online.clone(), avg_rate.clone(), conv1, params, online_stats, avg_stats);
  //let y = y.positive_clip();
  let y = y.positive_clip_inplace();
  // TODO
  let mut conv2 = conv_shape;
  conv2.ker_dims = [3, 3];
  conv2.stride = [1, 1];
  conv2.zero_pad = [1, 1];
  let y = build_batch_norm_conv(y, online.clone(), avg_rate.clone(), conv2, params, online_stats, avg_stats);
  //let y = y.positive_clip();
  let y = y.positive_clip_inplace();
  // TODO
  let mut conv3 = conv_shape;
  conv3.ker_dims = [1, 1];
  conv3.features *= expand;
  conv3.stride = [1, 1];
  conv3.zero_pad = [0, 0];
  let y = build_batch_norm_conv(y, online.clone(), avg_rate.clone(), conv3, params, online_stats, avg_stats);
  // TODO
  let mut proj_conv = conv_shape;
  proj_conv.src_dims[0] = src_dims[0];
  proj_conv.src_dims[1] = src_dims[1];
  proj_conv.src_features = src_features;
  proj_conv.ker_dims = [1, 1];
  proj_conv.features *= expand;
  proj_conv.zero_pad = [0, 0];
  let proj = build_batch_norm_conv(x, online.clone(), avg_rate.clone(), proj_conv, params, online_stats, avg_stats);
  let y = proj + y;
  //let y = sum_inplace_unstable(vec![x, y]);
  let y = y.positive_clip();
  //let y = y.positive_clip_inplace();
  y
}

fn build_resnet2(batch_sz: usize, n2: usize, n3: usize, n4: usize, n5: usize) -> (Val<GPUDeviceOuterBatchArray3d<u8>>, Val<GPUDeviceOuterBatchScalar<u32>>, Val<GPUDeviceOuterBatchArray1d<f32>>, Val<GPUDeviceScalar<f32>>, Val<bool>, Val<f32>, NodeVec, NodeVec, NodeVec) {
  let mut params = NodeVec::default();
  let mut online_stats = NodeVec::default();
  let mut avg_stats = NodeVec::default();

  let image_var = src(GPUDeviceOuterBatchArray3d::<u8>::zeros_init(([224, 224, 3], batch_sz)));
  let label_var = src(GPUDeviceOuterBatchScalar::<u32>::zeros_init(batch_sz));
  let online = src_init(false);
  let avg_rate = src_init(0.0);
  let epsilon: f32 = 1.0e-6;

  let x = image_var.clone().dequantize(0.0_f32, 1.0_f32);

  let mut conv1 = Conv2dShape::default_nchw();
  conv1.src_dims = [224, 224];
  conv1.src_features = 3;
  conv1.ker_dims = [3, 3];
  conv1.features = 64;
  conv1.stride = [2, 2];
  conv1.zero_pad = [1, 1];

  let x = build_batch_norm_conv(x, online.clone(), avg_rate.clone(), conv1, &mut params, &mut online_stats, &mut avg_stats);
  //let x = x.positive_clip();
  let x = x.positive_clip_inplace();

  let mut pool1 = Pool2dShape::default_nchw();
  pool1.src_size = [112, 112];
  pool1.src_features = 64;
  pool1.ker_size = [3, 3];
  pool1.stride = [2, 2];
  pool1.zero_pad = [1, 1];

  let x = x.max_pool(pool1);

  let mut conv2 = Conv2dShape::default_nchw();
  conv2.src_dims = [56, 56];
  conv2.src_features = 64;
  conv2.ker_dims = [3, 3];
  conv2.features = 64;
  conv2.stride = [1, 1];
  //conv2.stride = [2, 2];
  conv2.zero_pad = [0, 0];

  let mut x = x;
  x = build_proj_residual2_conv(x, online.clone(), avg_rate.clone(), conv2, [56, 56], 64, &mut params, &mut online_stats, &mut avg_stats);
  for _ in 1 .. n2 {
    x = build_residual2_conv(x, online.clone(), avg_rate.clone(), conv2, &mut params, &mut online_stats, &mut avg_stats);
  }

  let mut conv3 = Conv2dShape::default_nchw();
  conv3.src_dims = [28, 28];
  conv3.src_features = 128;
  conv3.ker_dims = [3, 3];
  conv3.features = 128;
  //conv3.stride = [1, 1];
  conv3.stride = [2, 2];
  conv3.zero_pad = [0, 0];

  let mut x = x;
  x = build_proj_residual2_conv(x, online.clone(), avg_rate.clone(), conv3, [56, 56], 64, &mut params, &mut online_stats, &mut avg_stats);
  for _ in 1 .. n3 {
    x = build_residual2_conv(x, online.clone(), avg_rate.clone(), conv3, &mut params, &mut online_stats, &mut avg_stats);
  }

  let mut conv4 = Conv2dShape::default_nchw();
  conv4.src_dims = [14, 14];
  conv4.src_features = 256;
  conv4.ker_dims = [3, 3];
  conv4.features = 256;
  //conv4.stride = [1, 1];
  conv4.stride = [2, 2];
  conv4.zero_pad = [0, 0];

  let mut x = x;
  x = build_proj_residual2_conv(x, online.clone(), avg_rate.clone(), conv4, [28, 28], 128, &mut params, &mut online_stats, &mut avg_stats);
  for _ in 1 .. n4 {
    x = build_residual2_conv(x, online.clone(), avg_rate.clone(), conv4, &mut params, &mut online_stats, &mut avg_stats);
  }

  let mut conv5 = Conv2dShape::default_nchw();
  conv5.src_dims = [7, 7];
  conv5.src_features = 512;
  conv5.ker_dims = [3, 3];
  conv5.features = 512;
  //conv5.stride = [1, 1];
  conv5.stride = [2, 2];
  conv5.zero_pad = [0, 0];

  let mut x = x;
  x = build_proj_residual2_conv(x, online.clone(), avg_rate.clone(), conv5, [14, 14], 256, &mut params, &mut online_stats, &mut avg_stats);
  for _ in 1 .. n5 {
    x = build_residual2_conv(x, online.clone(), avg_rate.clone(), conv5, &mut params, &mut online_stats, &mut avg_stats);
  }

  let mut avg_pool = Pool2dShape::default_nchw();
  avg_pool.src_size = [7, 7];
  avg_pool.src_features = 512;
  avg_pool.ker_size = [7, 7];
  //avg_pool.stride = [1, 1];
  avg_pool.stride = [7, 7];
  avg_pool.zero_pad = [0, 0];

  let x = x.average_pool(avg_pool);
  let x = x.flatten();

  let x = build_linear(x, 512, 1000, &mut params);

  let logit_var = x.clone();

  let (nll, _) = x.softmax_categorical_nll(label_var.clone());
  let loss_var = nll.batch_sum();

  (image_var, label_var, logit_var, loss_var, online, avg_rate, params, online_stats, avg_stats)
}

fn build_resnet3(batch_sz: usize, n2: usize, n3: usize, n4: usize, n5: usize) -> (Val<GPUDeviceOuterBatchArray3d<u8>>, Val<GPUDeviceOuterBatchScalar<u32>>, Val<GPUDeviceOuterBatchArray1d<f32>>, Val<GPUDeviceScalar<f32>>, Val<bool>, Val<f32>, NodeVec, NodeVec, NodeVec) {
  let mut params = NodeVec::default();
  let mut online_stats = NodeVec::default();
  let mut avg_stats = NodeVec::default();

  let image_var = src(GPUDeviceOuterBatchArray3d::<u8>::zeros_init(([224, 224, 3], batch_sz)));
  let label_var = src(GPUDeviceOuterBatchScalar::<u32>::zeros_init(batch_sz));
  let online = src_init(false);
  let avg_rate = src_init(0.0);
  let epsilon: f32 = 1.0e-6;

  let x = image_var.clone().dequantize(0.0_f32, 1.0_f32);

  let mut conv1 = Conv2dShape::default_nchw();
  conv1.src_dims = [224, 224];
  conv1.src_features = 3;
  conv1.ker_dims = [3, 3];
  conv1.features = 64;
  conv1.stride = [2, 2];
  conv1.zero_pad = [1, 1];

  let x = build_batch_norm_conv(x, online.clone(), avg_rate.clone(), conv1, &mut params, &mut online_stats, &mut avg_stats);
  //let x = x.positive_clip();
  let x = x.positive_clip_inplace();

  let mut pool1 = Pool2dShape::default_nchw();
  pool1.src_size = [112, 112];
  pool1.src_features = 64;
  pool1.ker_size = [3, 3];
  pool1.stride = [2, 2];
  pool1.zero_pad = [1, 1];

  let x = x.max_pool(pool1);

  let mut conv2 = Conv2dShape::default_nchw();
  conv2.src_dims = [56, 56];
  conv2.src_features = 64;
  conv2.ker_dims = [3, 3];
  conv2.features = 64;
  conv2.stride = [1, 1];
  //conv2.stride = [2, 2];
  conv2.zero_pad = [0, 0];

  let mut x = x;
  x = build_proj_residual3_conv(x, online.clone(), avg_rate.clone(), conv2, [56, 56], 64, 4, &mut params, &mut online_stats, &mut avg_stats);
  for _ in 1 .. n2 {
    x = build_residual3_conv(x, online.clone(), avg_rate.clone(), conv2, 4, &mut params, &mut online_stats, &mut avg_stats);
  }

  let mut conv3 = Conv2dShape::default_nchw();
  conv3.src_dims = [28, 28];
  conv3.src_features = 128;
  conv3.ker_dims = [3, 3];
  conv3.features = 128;
  //conv3.stride = [1, 1];
  conv3.stride = [2, 2];
  conv3.zero_pad = [0, 0];

  let mut x = x;
  x = build_proj_residual3_conv(x, online.clone(), avg_rate.clone(), conv3, [56, 56], 64 * 4, 4, &mut params, &mut online_stats, &mut avg_stats);
  for _ in 1 .. n3 {
    x = build_residual3_conv(x, online.clone(), avg_rate.clone(), conv3, 4, &mut params, &mut online_stats, &mut avg_stats);
  }

  let mut conv4 = Conv2dShape::default_nchw();
  conv4.src_dims = [14, 14];
  conv4.src_features = 256;
  conv4.ker_dims = [3, 3];
  conv4.features = 256;
  //conv4.stride = [1, 1];
  conv4.stride = [2, 2];
  conv4.zero_pad = [0, 0];

  let mut x = x;
  x = build_proj_residual3_conv(x, online.clone(), avg_rate.clone(), conv4, [28, 28], 128 * 4, 4, &mut params, &mut online_stats, &mut avg_stats);
  for _ in 1 .. n4 {
    x = build_residual3_conv(x, online.clone(), avg_rate.clone(), conv4, 4, &mut params, &mut online_stats, &mut avg_stats);
  }

  let mut conv5 = Conv2dShape::default_nchw();
  conv5.src_dims = [7, 7];
  conv5.src_features = 512;
  conv5.ker_dims = [3, 3];
  conv5.features = 512;
  //conv5.stride = [1, 1];
  conv5.stride = [2, 2];
  conv5.zero_pad = [0, 0];

  let mut x = x;
  x = build_proj_residual3_conv(x, online.clone(), avg_rate.clone(), conv5, [14, 14], 256 * 4, 4, &mut params, &mut online_stats, &mut avg_stats);
  for _ in 1 .. n5 {
    x = build_residual3_conv(x, online.clone(), avg_rate.clone(), conv5, 4, &mut params, &mut online_stats, &mut avg_stats);
  }

  let mut avg_pool = Pool2dShape::default_nchw();
  avg_pool.src_size = [7, 7];
  avg_pool.src_features = 512 * 4;
  avg_pool.ker_size = [7, 7];
  avg_pool.stride = [7, 7];
  avg_pool.zero_pad = [0, 0];

  let x = x.average_pool(avg_pool);
  let x = x.flatten();

  let x = build_linear(x, 512 * 4, 1000, &mut params);

  let logit_var = x.clone();

  let (nll, _) = x.softmax_categorical_nll(label_var.clone());
  let loss_var = nll.batch_sum();

  (image_var, label_var, logit_var, loss_var, online, avg_rate, params, online_stats, avg_stats)
}

fn main() {
  let mut group = DistProcGroup::default();
  for node in group {
    node.spawn(|proc| {
      println!("DEBUG: hello world: {}", proc.rank());

      let args: Vec<_> = env::args().collect();
      println!("DEBUG: args: {:?}", args);

      //let train_data_path = PathBuf::from("/scratch/snx3000/peterhj/data/ilsvrc2012/ILSVRC2012_img_train.tar");
      //let val_data_path = PathBuf::from("/scratch/snx3000/peterhj/data/ilsvrc2012/ILSVRC2012_img_val.tar");
      let train_data_path =
          if args.len() > 1 {
            PathBuf::from(&args[1])
          } else {
            PathBuf::from("/scratch/snx3000/peterhj/data/ilsvrc2012/ILSVRC2012_img_train.tar")
          };
      let val_data_path =
          if args.len() > 2 {
            PathBuf::from(&args[2])
          } else {
            PathBuf::from("/scratch/snx3000/peterhj/data/ilsvrc2012/ILSVRC2012_img_val.tar")
          };

      let data_cfg = ImagenetConfig{
        train_data: Some(train_data_path),
        //wordnet_ids: Some(wordnet_ids_path),
        val_data: Some(val_data_path),
        //val_ground_truth: Some(val_ground_truth_path),
        .. Default::default()
      };

      println!("DEBUG: loading val data...");
      let val_dataset = ImagenetValData::load_index(data_cfg.clone()).unwrap();
      println!("DEBUG: val len: {}", val_dataset.len());

      println!("DEBUG: loading train data...");
      let train_dataset = ImagenetTrainData::load_index(data_cfg.clone()).unwrap();
      println!("DEBUG: train len: {}", train_dataset.len());

      /*let val_shard = val_dataset.partition(proc.rank(), proc.num_ranks());
      println!("DEBUG: rank: {} val part len: {}", proc.rank(), val_shard.len());

      let train_shard = train_dataset.partition(proc.rank(), proc.num_ranks());
      println!("DEBUG: rank: {} train part len: {}", proc.rank(), train_shard.len());*/

      /*let val_shard = val_dataset.scatter_mpi_rma();
      let train_shard = train_dataset.scatter_mpi_rma();*/

      let num_classes = 1000;

      //let num_data_workers = 1;
      let num_data_workers = 10;
      //let num_data_workers = 20;
      //let batch_sz = 16;
      let batch_sz = 32;
      //let batch_reps = 1;
      let batch_reps = 8;
      let display_interval = 1;
      //let display_interval = 100;
      //let display_interval = 500;
      //let display_interval = 5000;
      let eval_interval: Option<usize> = None;
      //let eval_interval = Some(5000);

      let minibatch_sz = batch_sz * batch_reps;

      println!("DEBUG: num data workers: {:?}", num_data_workers);
      println!("DEBUG: batch sz: {:?}", batch_sz);
      println!("DEBUG: display interval: {:?}", display_interval);
      println!("DEBUG: eval interval: {:?}", eval_interval);

      // TODO: sharding.
      let train_iter = {
          let mut worker_rngs = Vec::with_capacity(num_data_workers);
          for _ in 0 .. num_data_workers {
            worker_rngs.push(SmallRng::from_rng(thread_rng()).unwrap());
          }
          async_join_data(num_data_workers, batch_sz, |worker_rank| {
            let mut rng = worker_rngs[worker_rank].clone();
            train_dataset.clone()
              .uniform_random(&mut rng)
              .map_data(move |(value, label)| {
                /*let mut avg_value = 0;
                for &v in value.iter() {
                  avg_value += v;
                }
                /*let maybe_image = match ColorImage::decode(&value) {
                  Ok(mut image) => {
                    Some(image)
                  }
                  Err(_) => {
                    println!("WARNING: image decode error");
                    None
                  }
                };*/
                (Some(avg_value), label)*/
                let maybe_image = match ColorImage::decode(&value) {
                  Ok(mut image) => {
                    // TODO: data augmentation.
                    //println!("DEBUG: image: dims: {} {}", image.width(), image.height());
                    //println!("DEBUG: augment: pass: 0 sample: {:?}", &image.raster_line(0)[ .. 10]);
                    inception_crop_resize(224, 224, &mut image, &mut rng);
                    //println!("DEBUG: augment: pass: 1 sample: {:?}", &image.raster_line(0)[ .. 10]);
                    random_flip(&mut image, &mut rng);
                    //println!("DEBUG: augment: pass: 2 sample: {:?}", &image.raster_line(0)[ .. 10]);
                    /*scale_resize(256, &mut image);
                    center_crop(224, 224, &mut image);*/
                    Some(image)
                  }
                  Err(_) => {
                    println!("WARNING: image decode error");
                    None
                  }
                };
                (maybe_image, label)
              })
          })
          //.async_prefetch_data(batch_sz * max(2, num_data_workers))
          .batch_data(batch_sz)
      };

      // TODO
      //enable_static_graph_logging();
      //let (image_var, label_var, logit_var, loss_var, online, avg_rate, params, online_stats, avg_stats) = build_resnet2(batch_sz, 2, 2, 2, 2);
      //let (image_var, label_var, logit_var, loss_var, online, avg_rate, params, online_stats, avg_stats) = build_resnet2(batch_sz, 3, 4, 6, 3);
      let (image_var, label_var, logit_var, loss_var, online, avg_rate, params, online_stats, avg_stats) = build_resnet3(batch_sz, 3, 4, 6, 3);
      //println!("DEBUG: build adjoint...");
      let mut loss_sink = sink(loss_var.clone());
      let params = params.reversed();
      let grads = params.adjoints(&mut loss_sink);

      let init_txn = txn();
      params.persist(init_txn);
      println!("DEBUG: params count: {}", params.serialize_vec(init_txn, &mut ()));

      let batch_avg_rate = src_init(0.0_f32);
      let step_size = src_init(0.0_f32);
      let momentum = src_init(0.0_f32);

      let lr_schedule = PiecewiseSeries::new(
          0.1_f32,
          vec![
              (150_000, 0.01_f32),
              (300_000, 0.001_f32),
          ]);

      /*
      let avg_grads = grads.clone().zeros_like_vec().online_average_vec(batch_avg_rate.clone(), grads.clone());
      let grad_step = params.clone().gradient_momentum_step_vec(step_size.clone(), momentum.clone(), avg_grads.clone());
      */

      //assert_eq!(1, batch_reps);
      let batch_grads_vec = grads.clone().vectorize();
      let avg_grads_vec = zeros_like(batch_grads_vec.clone()).online_average(batch_avg_rate.clone(), batch_grads_vec.clone());
      let sum_avg_grads_vec = avg_grads_vec.clone().sum_spmd(proc.clone());
      let params_vec = params.clone().vectorize();
      let grad_vec_step = params_vec.clone().gradient_momentum_step(step_size.clone(), momentum.clone(), sum_avg_grads_vec.clone());
      let params_devec = params_vec.clone().devectorize(params.clone());

      let mut stopwatch = Stopwatch::new();

      let mut image_data = MemArray4d::<u8>::zeros([224, 224, 3, batch_sz]);
      let mut label_data = MemArray1d::<u32>::zeros(batch_sz);
      let mut logit_data = MemArray2d::<f32>::zeros([num_classes, batch_sz]);
      let mut loss: f32 = 0.0;

      let mut labels = Vec::with_capacity(batch_sz);
      let mut count = 0;
      let mut acc_ct = 0;
      let mut loss_sum: f32 = 0.0;

      // TODO: debugging.
      //enable_dynamic_graph_logging();
      DoubleCheckLogging::enable();

      for (batch_nr, batch) in train_iter.enumerate() {
        let iter_nr = batch_nr / batch_reps;
        let rep_nr = batch_nr % batch_reps;

        labels.clear();
        for (idx, (maybe_image, label)) in batch.into_iter().enumerate() {
          if let Some(ref image) = maybe_image {
            image.dump_planes(
                &mut image_data.flat_view_mut().unwrap()
                  .as_mut_slice()[idx * 224 * 224 * 3 .. (idx + 1) * 224 * 224 * 3]);
          } else {
            println!("WARNING: train: image decode error, missing image in iter: {} batch idx: {}", iter_nr, idx);
          }
          /*println!("DEBUG: train: image batch: idx: {} sample: {:?} {:?}",
              idx,
              &image_data.flat_view().unwrap()
                .as_slice()[idx * 224 * 224 * 3 .. idx * 224 * 224 * 3 + 10],
              &image_data.flat_view().unwrap()
                .as_slice()[(idx + 1) * 224 * 224 * 3 - 10 .. (idx + 1) * 224 * 224 * 3],
          );*/
          label_data.as_view_mut().as_mut_slice()[idx] = label;
          labels.push(label);
        }

        /*labels.clear();
        for _ in 0 .. batch_sz {
          labels.push(0);
        }*/

        let batch_txn = txn();
        // TODO: evaluate the batch.
        image_var.deserialize(batch_txn, &mut image_data);
        label_var.deserialize(batch_txn, &mut label_data);
        online.set(batch_txn, true);
        params.persist(batch_txn);
        loss_var.eval(batch_txn);
        grads.eval(batch_txn);
        batch_avg_rate.set(batch_txn, 1.0 / (rep_nr + 1) as f32);
        batch_grads_vec.eval(batch_txn);
        avg_grads_vec.eval(batch_txn);
        logit_var.serialize(batch_txn, &mut logit_data);
        loss_var.serialize(batch_txn, &mut loss);

        count += batch_sz;
        loss_sum += loss;
        for idx in 0 .. batch_sz {
          let k = _arg_max(&logit_data.flat_view().unwrap().as_slice()[num_classes * idx .. num_classes * (idx + 1)]);
          if k == labels[idx] as _ {
            acc_ct += 1;
          }
        }

        if rep_nr == batch_reps - 1 {
          let step_txn = txn();

          step_size.set(step_txn, -lr_schedule.at(iter_nr) / (proc.num_ranks() * batch_sz) as f32);
          momentum.set(step_txn, 0.9);
          params.persist(step_txn);
          avg_grads_vec.persist(step_txn);
          assert!(sum_avg_grads_vec.eval(step_txn));
          assert!(grad_vec_step.eval(step_txn));

          let update_txn = txn();

          params_vec.persist(update_txn);
          assert!(params_devec.eval(update_txn));

          avg_rate.set(update_txn, if iter_nr == 0 { 1.0 } else { 0.003 });
          online_stats.persist(update_txn);
          assert_eq!(avg_stats.eval(update_txn), avg_stats.len());
        }

        DoubleCheckLogging::disable();
        disable_dynamic_graph_logging();
        proc.wait_barrier();

        if rep_nr == batch_reps - 1 && (iter_nr + 1) % display_interval == 0 {
          println!("DEBUG: train: iters: {} acc: {:.4} ({}/{}) loss: {:.6} elapsed: {:.6} s",
              iter_nr + 1,
              acc_ct as f64 / count as f64,
              acc_ct, count,
              loss_sum / count as f32,
              stopwatch.click().lap_time());
          count = 0;
          acc_ct = 0;
          loss_sum = 0.0;
        }

        if eval_interval.is_some() && (iter_nr + 1) % eval_interval.unwrap() == 0 {
          println!("DEBUG: eval: evaluating...");
          let val_shard = val_dataset.clone().range_shard(proc.rank(), proc.num_ranks());
          let shard_len = val_shard.len();
          let val_iter = {
            let mut val_splits = async_split_data(num_data_workers, || {
              val_shard.one_pass()
            });
            let mut val_splits = val_splits.drain(..);
            async_join_data(num_data_workers, batch_sz, |worker_rank| {
              val_splits.next().unwrap()
                .map_data(|(value, label)| {
                  let image = match ColorImage::decode(&value) {
                    Ok(mut image) => {
                      scale_resize(256, &mut image);
                      center_crop(224, 224, &mut image);
                      Arc::new(image)
                    }
                    Err(_) => {
                      panic!("WARNING: image decode error");
                    }
                  };
                  (image, label)
                })
            })
            .round_up_data(batch_sz)
            .batch_data(batch_sz)
          };
          let mut eval_ctr = 0;
          let mut eval_acc_ct = 0;
          'eval_batch_loop: for eval_batch in val_iter {
            labels.clear();
            for (idx, ((image, label), _)) in eval_batch.into_iter().enumerate() {
              image.dump_planes(
                  &mut image_data.flat_view_mut().unwrap()
                    .as_mut_slice()[idx * 224 * 224 * 3 .. (idx + 1) * 224 * 224 * 3]);
              labels.push(label);
              label_data.as_view_mut().as_mut_slice()[idx] = label;
            }
            let batch_txn = txn();
            // TODO: evaluate the batch.
            image_var.deserialize(batch_txn, &mut image_data);
            label_var.deserialize(batch_txn, &mut label_data);
            online.set(batch_txn, false);
            params.persist(batch_txn);
            loss_var.eval(batch_txn);
            logit_var.serialize(batch_txn, &mut logit_data);
            for idx in 0 .. batch_sz {
              let k = _arg_max(&logit_data.flat_view().unwrap().as_slice()[num_classes * idx .. num_classes * (idx + 1)]);
              if k == labels[idx] as _ {
                eval_acc_ct += 1;
              }
              eval_ctr += 1;
              if eval_ctr == shard_len {
                break 'eval_batch_loop;
              }
            }
          }
          println!("DEBUG: eval: count: {} acc: {:.4} ({}/{}) elapsed: {:.6} s",
              eval_ctr,
              eval_acc_ct as f64 / eval_ctr as f64,
              eval_acc_ct, eval_ctr,
              stopwatch.click().lap_time());
        }

        //iter_nr += 1;
      }
    }).unwrap().join();
  }
}
