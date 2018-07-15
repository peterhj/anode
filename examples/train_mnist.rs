#![feature(alloc_system)]

extern crate anode;
extern crate colorimage;
extern crate gpudevicemem;
extern crate memarray;
extern crate rand;
extern crate sharedmem;
extern crate superdata;

use anode::*;
use anode::log::*;
use anode::ops::*;
use anode::proc_dist::*;
use anode::utils::*;
use colorimage::*;
use gpudevicemem::*;
use gpudevicemem::array::*;
use memarray::*;
use rand::prelude::*;
use rand::distributions::{Distribution, Uniform, Normal};
use sharedmem::*;
use superdata::*;
use superdata::datasets::mnist::*;
use superdata::image::*;
use superdata::utils::*;

use std::cmp::{max, min};
use std::env;
use std::path::{PathBuf};
use std::rc::{Rc};
use std::sync::{Arc};

pub trait KaimingConv2dInit<T, R: Rng> {
  type RValue;

  fn kaiming_conv2d_init(ker_sz: [usize; 2], src: usize, dst: usize, seed_rng: &mut R) -> Self::RValue;
}

impl<R: Rng> KaimingConv2dInit<f32, R> for GPUDeviceArray4d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn kaiming_conv2d_init(ker_sz: [usize; 2], src_ch: usize, dst_ch: usize, seed_rng: &mut R) -> Self::RValue {
    //let seed = seed_rng.next_u64();
    Rc::new(move |_, conn: GPUDeviceConn| {
      let shape = [ker_sz[0], ker_sz[1], src_ch, dst_ch];
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

fn build_conv(x: Val<GPUDeviceOuterBatchArray3d<f32>>, conv_shape: Conv2dShape, params: &mut NodeVec) -> Val<GPUDeviceOuterBatchArray3d<f32>> {
  let w = src(GPUDeviceArray4d::<f32>::kaiming_conv2d_init(
      conv_shape.ker_dims,
      conv_shape.src_features,
      conv_shape.features,
      &mut thread_rng()));
  params.push_val(w.clone());
  let x = w.conv(conv_shape.clone(), x);
  x
}

fn build_convnet(batch_sz: usize) {
  let mut params = NodeVec::default();

  let image_var = src(GPUDeviceOuterBatchArray3d::<u8>::zeros_init(([28, 28, 1], batch_sz)));
  let label_var = src(GPUDeviceOuterBatchScalar::<u32>::zeros_init(batch_sz));

  let x = image_var.clone().dequantize(0.0_f32, 1.0_f32);

  let mut conv1 = Conv2dShape::default_nchw();
  conv1.src_dims = [28, 28];
  conv1.src_features = 1;
  conv1.ker_dims = [5, 5];
  conv1.features = 32;
  conv1.stride = [1, 1];
  conv1.zero_pad = [2, 2];

  let x = build_conv(x, conv1, &mut params);
  let x = x.positive_clip_inplace();

  let mut conv1_2 = Conv2dShape::default_nchw();
  conv1_2.src_dims = [28, 28];
  conv1_2.src_features = 32;
  conv1_2.ker_dims = [3, 3];
  conv1_2.features = 32;
  conv1_2.stride = [1, 1];
  conv1_2.zero_pad = [1, 1];

  let x = build_conv(x, conv1_2, &mut params);
  let x = x.positive_clip_inplace();

  let mut conv2_1 = Conv2dShape::default_nchw();
  conv2_1.src_dims = [28, 28];
  conv2_1.src_features = 32;
  conv2_1.ker_dims = [3, 3];
  conv2_1.features = 64;
  conv2_1.stride = [2, 2];
  conv2_1.zero_pad = [1, 1];

  let x = build_conv(x, conv2_1, &mut params);
  let x = x.positive_clip_inplace();

  let mut conv2_2 = Conv2dShape::default_nchw();
  conv2_2.src_dims = [14, 14];
  conv2_2.src_features = 64;
  conv2_2.ker_dims = [3, 3];
  conv2_2.features = 64;
  conv2_2.stride = [1, 1];
  conv2_2.zero_pad = [1, 1];

  let x = build_conv(x, conv2_2, &mut params);
  let x = x.positive_clip_inplace();

  let mut conv3_1 = Conv2dShape::default_nchw();
  conv3_1.src_dims = [14, 14];
  conv3_1.src_features = 64;
  conv3_1.ker_dims = [3, 3];
  conv3_1.features = 64;
  conv3_1.stride = [2, 2];
  conv3_1.zero_pad = [1, 1];

  let x = build_conv(x, conv3_1, &mut params);
  let x = x.positive_clip_inplace();

  let mut conv3_2 = Conv2dShape::default_nchw();
  conv3_2.src_dims = [7, 7];
  conv3_2.src_features = 64;
  conv3_2.ker_dims = [3, 3];
  conv3_2.features = 64;
  conv3_2.stride = [1, 1];
  conv3_2.zero_pad = [1, 1];

  let x = build_conv(x, conv3_2, &mut params);
  let x = x.positive_clip_inplace();

  let x = x.flatten();

  // TODO
}

fn main() {
  let mut group = DistProcGroup::default();
  for node in group {
    node.spawn(|proc| {
      println!("DEBUG: hello world: {}", proc.rank());

      let mut dataset_cfg = MnistConfig::default();
      dataset_cfg.path = Some(PathBuf::from("../datasets/mnist"));

      let train_data = dataset_cfg.open_train_data().unwrap();
      let test_data = dataset_cfg.open_test_data();

      let num_classes = 10;

      let batch_sz = 32;
      let batch_reps = 1;

      let train_iter = {
        train_data
          .uniform_random(&mut thread_rng())
          .batch_data(batch_sz)
      };

      let batch_avg_rate = src_init(0.0_f32);

      let mut stopwatch = Stopwatch::new();

      let mut image_data = MemArray4d::<u8>::zeros([28, 28, 1, batch_sz]);
      let mut label_data = MemArray1d::<u32>::zeros(batch_sz);
      let mut logit_data = MemArray2d::<f32>::zeros([num_classes, batch_sz]);
      let mut loss: f32 = 0.0;

      let mut labels = Vec::<u32>::with_capacity(batch_sz);
      let mut acc_ct = 0;
      let mut loss_sum: f32 = 0.0;

      for (batch_nr, batch) in train_iter.enumerate() {
        let iter_nr = batch_nr / batch_reps;
        let rep_nr = batch_nr % batch_reps;

        let batch_txn = txn();
        /*image_var.deserialize(batch_txn, &mut image_data);
        label_var.deserialize(batch_txn, &mut label_data);
        params.persist(batch_txn);
        loss_var.eval(batch_txn);
        grads.eval(batch_txn);
        batch_avg_rate.set(batch_txn, 1.0 / (rep_nr + 1) as f32);
        batch_grads_vec.eval(batch_txn);
        avg_grads_vec.eval(batch_txn);
        logit_var.serialize(batch_txn, &mut logit_data);
        loss_var.serialize(batch_txn, &mut loss);
        loss_sum += loss;*/

        if rep_nr == batch_reps - 1 {
          let step_txn = txn();
          // TODO

          let update_txn = txn();
          /*params_vec.persist(update_txn);
          assert!(params_devec.eval(update_txn));*/
        }
      }
    }).unwrap().join();
  }
}
