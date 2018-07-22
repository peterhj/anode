#![feature(alloc_system)]

extern crate anode;
extern crate colorimage;
extern crate gpudevicemem;
extern crate memarray;
extern crate minidata;
extern crate rand;
extern crate sharedmem;

use anode::*;
use anode::log::*;
use anode::ops::*;
use anode::proc_dist::*;
use anode::utils::*;
use colorimage::*;
use gpudevicemem::*;
use gpudevicemem::array::*;
use memarray::*;
use minidata::*;
//use minidata::datasets::imagenet::*;
use minidata::image::*;
use minidata::utils::*;
use rand::prelude::*;
use rand::distributions::{Distribution, Uniform, Normal};
use sharedmem::*;

use std::cmp::{max, min};
use std::env;
use std::path::{PathBuf};
use std::rc::{Rc};
use std::sync::{Arc};

pub trait KaimingConv3dInit<T, R: Rng> {
  type RValue;

  fn kaiming_conv3d_init(ker_sz: [usize; 3], src: usize, dst: usize, seed_rng: &mut R) -> Self::RValue;
}

impl<R: Rng> KaimingConv3dInit<f32, R> for GPUDeviceArray5d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn kaiming_conv3d_init(ker_sz: [usize; 3], src_ch: usize, dst_ch: usize, seed_rng: &mut R) -> Self::RValue {
    //let seed = seed_rng.next_u64();
    Rc::new(move |_, conn: GPUDeviceConn| {
      let shape = [ker_sz[0], ker_sz[1], ker_sz[2], src_ch, dst_ch];
      // TODO: seed the local rng here.
      let mut h_arr = MemArray5d::<f32>::zeros(shape);
      {
        let mean = 0.0;
        let std = (2.0 / (ker_sz[0] * ker_sz[1] * ker_sz[2] * src_ch) as f64).sqrt();
        let dist = Normal::new(mean, std);
        let mut v = h_arr.as_view_mut();
        let xs = v.flat_slice_mut().unwrap();
        for x in xs.iter_mut() {
          *x = dist.sample(&mut thread_rng()) as f32;
        }
      }
      let mut arr = GPUDeviceArray5d::<f32>::zeros(shape, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}

fn build_conv(x: Val<GPUDeviceOuterBatchArray4d<f32>>, conv_shape: Conv3dShape, params: &mut NodeVec) -> Val<GPUDeviceOuterBatchArray4d<f32>> {
  let w = src(GPUDeviceArray5d::<f32>::kaiming_conv3d_init(
      conv_shape.ker_dims,
      conv_shape.src_features,
      conv_shape.features,
      &mut thread_rng()));
  params.push_val(w.clone());
  let x = w.conv(conv_shape.clone(), x);
  x
}

fn build_upconv(x: Val<GPUDeviceOuterBatchArray4d<f32>>, conv_shape: Conv3dShape, params: &mut NodeVec) -> Val<GPUDeviceOuterBatchArray4d<f32>> {
  let w = src(GPUDeviceArray5d::<f32>::kaiming_conv3d_init(
      conv_shape.ker_dims,
      conv_shape.src_features,
      conv_shape.features,
      &mut thread_rng()));
  params.push_val(w.clone());
  let x = w.left_transpose_conv(conv_shape.clone(), x);
  x
}

//fn build_unet(batch_sz: usize) -> (Val<GPUDeviceOuterBatchArray4d<u8>>, Val<GPUDeviceOuterBatchArray4d<u32>>, Val<GPUDeviceScalar<f32>>, NodeVec) {
fn build_unet(batch_sz: usize, pz: usize) -> (Val<GPUDeviceOuterBatchArray4d<u8>>, Val<GPUDeviceOuterBatchArray4d<u32>>, Val<GPUDeviceOuterBatchArray4d<f32>>, NodeVec) {
  assert!(batch_sz >= 1);
  assert!(pz >= 1);

  let mut params = NodeVec::default();

  let image_var = src(GPUDeviceOuterBatchArray4d::<u8>::zeros_init(([128, 128, 128 / pz, 4], batch_sz)));
  let label_var = src(GPUDeviceOuterBatchArray4d::<u32>::zeros_init(([128, 128, 128 / pz, 1], batch_sz)));

  let x = image_var.clone().dequantize(0.0_f32, 1.0_f32);

  let mut conv1_0 = Conv3dShape::default_ncdhw();
  conv1_0.src_dims = [128, 128, 128 / pz];
  conv1_0.src_features = 4;
  conv1_0.ker_dims = [3, 3, 3];
  conv1_0.features = 64;
  conv1_0.stride = [1, 1, 1];
  conv1_0.zero_pad = [1, 1, 1];

  let x = build_conv(x, conv1_0, &mut params);
  let x = x.positive_clip_inplace();

  let mut conv1 = Conv3dShape::default_ncdhw();
  conv1.src_dims = [128, 128, 128 / pz];
  conv1.src_features = 64;
  conv1.ker_dims = [3, 3, 3];
  conv1.features = 64;
  conv1.stride = [1, 1, 1];
  conv1.zero_pad = [1, 1, 1];

  let x = build_conv(x, conv1, &mut params);
  let x = x.positive_clip_inplace();

  let x = build_conv(x, conv1, &mut params);
  let x = x.positive_clip_inplace();
  let x1 = x.clone();

  let mut pool1 = Pool3dShape::default_ncdhw();
  pool1.src_dims = [128, 128, 128 / pz];
  pool1.src_features = 64;
  pool1.ker_dims = [2, 2, 2];
  pool1.stride = [2, 2, 2];
  pool1.zero_pad = [0, 0, 0];

  let x = x.max_pool(pool1);

  let mut conv2_1 = Conv3dShape::default_ncdhw();
  conv2_1.src_dims = [64, 64, 64 / pz];
  conv2_1.src_features = 64;
  conv2_1.ker_dims = [3, 3, 3];
  conv2_1.features = 128;
  conv2_1.stride = [1, 1, 1];
  conv2_1.zero_pad = [1, 1, 1];

  let x = build_conv(x, conv2_1, &mut params);
  let x = x.positive_clip_inplace();

  let mut conv2 = Conv3dShape::default_ncdhw();
  conv2.src_dims = [64, 64, 64 / pz];
  conv2.src_features = 128;
  conv2.ker_dims = [3, 3, 3];
  conv2.features = 128;
  conv2.stride = [1, 1, 1];
  conv2.zero_pad = [1, 1, 1];

  let x = build_conv(x, conv2, &mut params);
  let x = x.positive_clip_inplace();
  let x2 = x.clone();

  let mut pool2 = Pool3dShape::default_ncdhw();
  pool2.src_dims = [64, 64, 64 / pz];
  pool2.src_features = 128;
  pool2.ker_dims = [2, 2, 2];
  pool2.stride = [2, 2, 2];
  pool2.zero_pad = [0, 0, 0];

  let x = x.max_pool(pool2);

  let mut conv3_1 = Conv3dShape::default_ncdhw();
  conv3_1.src_dims = [32, 32, 32 / pz];
  conv3_1.src_features = 128;
  conv3_1.ker_dims = [3, 3, 3];
  conv3_1.features = 256;
  conv3_1.stride = [1, 1, 1];
  conv3_1.zero_pad = [1, 1, 1];

  let x = build_conv(x, conv3_1, &mut params);
  let x = x.positive_clip_inplace();

  let mut conv3 = Conv3dShape::default_ncdhw();
  conv3.src_dims = [32, 32, 32 / pz];
  conv3.src_features = 256;
  conv3.ker_dims = [3, 3, 3];
  conv3.features = 256;
  conv3.stride = [1, 1, 1];
  conv3.zero_pad = [1, 1, 1];

  let x = build_conv(x, conv3, &mut params);
  let x = x.positive_clip_inplace();
  let x3 = x.clone();

  let mut pool3 = Pool3dShape::default_ncdhw();
  pool3.src_dims = [32, 32, 32 / pz];
  pool3.src_features = 256;
  pool3.ker_dims = [2, 2, 2];
  pool3.stride = [2, 2, 2];
  pool3.zero_pad = [0, 0, 0];

  let x = x.max_pool(pool3);

  let mut conv4_1 = Conv3dShape::default_ncdhw();
  conv4_1.src_dims = [16, 16, 16 / pz];
  conv4_1.src_features = 256;
  conv4_1.ker_dims = [3, 3, 3];
  conv4_1.features = 512;
  conv4_1.stride = [1, 1, 1];
  conv4_1.zero_pad = [1, 1, 1];

  let x = build_conv(x, conv4_1, &mut params);
  let x = x.positive_clip_inplace();

  let mut conv4 = Conv3dShape::default_ncdhw();
  conv4.src_dims = [16, 16, 16 / pz];
  conv4.src_features = 512;
  conv4.ker_dims = [3, 3, 3];
  conv4.features = 512;
  conv4.stride = [1, 1, 1];
  conv4.zero_pad = [1, 1, 1];

  let x = build_conv(x, conv4, &mut params);
  let x = x.positive_clip_inplace();

  let mut upconv4 = Conv3dShape::default_ncdhw();
  upconv4.src_dims = [32, 32, 32 / pz];
  upconv4.src_features = 256;
  upconv4.features = 512;
  upconv4.ker_dims = [2, 2, 2];
  upconv4.stride = [2, 2, 2];
  upconv4.zero_pad = [0, 0, 0];

  let x = x3.clone() + build_upconv(x, upconv4, &mut params);
  let x = x.positive_clip_inplace();

  let x = build_conv(x, conv3, &mut params);
  let x = x.positive_clip_inplace();

  let x = build_conv(x, conv3, &mut params);
  let x = x.positive_clip_inplace();

  let mut upconv3 = Conv3dShape::default_ncdhw();
  upconv3.src_dims = [64, 64, 64 / pz];
  upconv3.src_features = 128;
  upconv3.features = 256;
  upconv3.ker_dims = [2, 2, 2];
  upconv3.stride = [2, 2, 2];
  upconv3.zero_pad = [0, 0, 0];

  let x = x2.clone() + build_upconv(x, upconv3, &mut params);
  let x = x.positive_clip_inplace();

  let x = build_conv(x, conv2, &mut params);
  let x = x.positive_clip_inplace();

  let x = build_conv(x, conv2, &mut params);
  let x = x.positive_clip_inplace();

  let mut upconv2 = Conv3dShape::default_ncdhw();
  upconv2.src_dims = [128, 128, 128 / pz];
  upconv2.src_features = 64;
  upconv2.features = 128;
  upconv2.ker_dims = [2, 2, 2];
  upconv2.stride = [2, 2, 2];
  upconv2.zero_pad = [0, 0, 0];

  let x = x1.clone() + build_upconv(x, upconv2, &mut params);
  let x = x.positive_clip_inplace();

  let x = build_conv(x, conv1, &mut params);
  let x = x.positive_clip_inplace();

  let x = build_conv(x, conv1, &mut params);
  let x = x.positive_clip_inplace();

  let mut conv_final = Conv3dShape::default_ncdhw();
  conv_final.src_dims = [128, 128, 128 / pz];
  conv_final.src_features = 64;
  conv_final.ker_dims = [1, 1, 1];
  conv_final.features = 1;
  conv_final.stride = [1, 1, 1];
  conv_final.zero_pad = [0, 0, 0];

  let x = build_conv(x, conv_final, &mut params);

  (image_var, label_var, x, params)
}

fn main() {
  let mut group = DistProcGroup::default();
  for node in group {
    node.spawn(|proc| {
      println!("DEBUG: hello world: {}", proc.rank());

      let early_trials = 2;
      let num_trials = 20;

      /*let batch_sz = 2;
      let pz = 1;*/

      //let batch_sz = 1;
      let batch_sz = 16;
      //let pz = 1;
      let pz = 16;

      /*let batch_sz = 4;
      let pz = 1;*/

      /*let batch_sz = 32;
      let pz = 8;*/

      if proc.rank() == 0 {
      println!("DEBUG: bench: rank: {} n: {} batch sz: {} pz: {}",
          proc.rank(),
          num_trials, batch_sz, pz);
      }

      let (image_var, label_var, x, params) = build_unet(batch_sz, pz);
      let params = params.reversed();

      let mut sink_x = sink(x.clone());
      let grads = params.adjoints(&mut sink_x);

      let mut image_data = MemArray5d::<u8>::zeros([128, 128, 128 / pz, 4, batch_sz]);
      let mut label_data = MemArray5d::<u32>::zeros([128, 128, 128 / pz, 1, batch_sz]);

      let mut stopwatch = Stopwatch::new();

      for batch_nr in 0 .. num_trials + early_trials {
        let batch_txn = txn();
        // TODO: evaluate the batch.
        image_var.deserialize(batch_txn, &mut image_data);
        label_var.deserialize(batch_txn, &mut label_data);
        params.persist(batch_txn);
        x.eval(batch_txn);
        grads.eval(batch_txn);
        proc.barrier();
        if batch_nr < early_trials {
          stopwatch.click();
        }
      }

      if proc.rank() == 0 {
      println!("DEBUG: bench: rank: {} n: {} avg elapsed: {:.6} s",
          proc.rank(),
          num_trials,
          stopwatch.click().lap_time() / num_trials as f64);
      }
    }).unwrap().join();
  }
}
